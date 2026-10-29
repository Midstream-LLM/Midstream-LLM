"""Generate text from a Jishui CUDA checkpoint (PyTorch counterpart of generate_mlx.py).

Sampling is intentionally shared with the MLX path (``sample_next_token``) so a
fixed prompt/seed/parameter set is comparable across checkpoints and backends.
Only the model weights are read; no optimizer or training state is touched.

``--prompt-file`` runs a prompt suite (see ``tests/prompts/classical-inference-suite.json``)
and enforces the suite's structural checks: prompts must round-trip through the
tokenizer, and outputs must not contain U+FFFD, blocked special tokens, or be empty.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tokenizers import Tokenizer

from .checkpoints import latest_checkpoint
from .config import ModelConfig
from .generate_mlx import UNTRAINED_SPECIAL_TOKENS, sample_next_token
from .model_torch import JishuiForCausalLM

DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
REPLACEMENT_CHAR = "\ufffd"
# Measured on ccc-bbpe-32k: traditional text costs 1.0-1.64x the tokens of the
# equivalent simplified text because the tokenizer was trained on CCC (mostly
# simplified). A ratio outside this band means tokenizer or data drift.
TRADITIONAL_RATIO_BAND = (1.0, 1.7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a Jishui CUDA checkpoint")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/jishui-200m-cuda-4b"))
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint directory, or 'latest' under --run-dir",
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/ccc-bbpe-32k"))
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to complete; repeat the option for multiple prompts",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="JSON prompt suite: a list of strings, or {\"cases\": [...]} with id/prompt/script/genre",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write run metadata and every generation result to this JSON file",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--dtype",
        default=None,
        choices=tuple(DTYPE_BY_NAME),
        help="Weight dtype; defaults to float16 on CUDA and float32 on CPU",
    )
    parser.add_argument(
        "--top-logits",
        type=int,
        default=0,
        help="Print the N most likely next tokens after the prompt (diagnostics)",
    )
    parser.add_argument(
        "--allow-all-special-tokens",
        action="store_true",
        help="Do not mask pad/unk/chat tokens that were absent from pretraining",
    )
    parser.add_argument("--show-special-tokens", action="store_true")
    return parser.parse_args()


def load_cases(args: argparse.Namespace) -> list[dict]:
    if args.prompt_file and args.prompts:
        raise ValueError("use either --prompt or --prompt-file, not both")
    if args.prompt_file:
        raw = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
        entries = raw["cases"] if isinstance(raw, dict) else raw
        cases = []
        for index, entry in enumerate(entries, 1):
            if isinstance(entry, str):
                entry = {"prompt": entry}
            if "prompt" not in entry:
                raise ValueError(f"case {index} in {args.prompt_file} has no 'prompt'")
            cases.append({
                "id": entry.get("id", f"case-{index}"),
                "prompt": entry["prompt"],
                "pair": entry.get("pair"),
                "script": entry.get("script", "n/a"),
                "genre": entry.get("genre", "n/a"),
                "max_new_tokens": entry.get("max_new_tokens"),
            })
        if not cases:
            raise ValueError(f"prompt file contains no cases: {args.prompt_file}")
        return cases
    prompts = args.prompts or ["太史公曰："]
    return [
        {"id": f"cli-{index}", "prompt": prompt, "pair": None, "script": "n/a",
         "genre": "n/a", "max_new_tokens": None}
        for index, prompt in enumerate(prompts, 1)
    ]


def check_prompt(tokenizer: Tokenizer, case: dict) -> int:
    """Enforce suite criterion 1: every prompt must survive a tokenizer round-trip."""
    prompt = case["prompt"]
    if prompt == "":
        return 0
    if REPLACEMENT_CHAR in prompt:
        raise ValueError(f"{case['id']}: prompt contains U+FFFD")
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids:
        raise ValueError(f"{case['id']}: prompt encoded to zero tokens")
    if tokenizer.decode(ids, skip_special_tokens=False) != prompt:
        raise ValueError(f"{case['id']}: prompt does not round-trip through the tokenizer")
    return len(ids)


def generated_span(text: str, prompt: str) -> str:
    """Text added by the model: drop the prompt echo when the decode starts with it."""
    return text[len(prompt):] if prompt and text.startswith(prompt) else text


def distinct_4gram_ratio(text: str) -> float | None:
    """Fraction of distinct 4-character n-grams; low values expose repetition loops."""
    if len(text) < 5:
        return None
    grams = [text[index:index + 4] for index in range(len(text) - 3)]
    return round(len(set(grams)) / len(grams), 3)


def analyze_output(
    text: str,
    prompt: str,
    generated_ids: Sequence[int],
    blocked_token_ids: Sequence[int],
    stop_reason: str,
) -> tuple[list[str], list[str]]:
    """Return (failures, notes) for one generation.

    Failures are structural defects. Notes are expected behaviour that a human
    should still look at: a truncated generation can end mid-character (a
    byte-level BPE artifact), and a model that judges the prompt a complete
    document emits <|endoftext|> immediately.
    """
    flags: list[str] = []
    notes: list[str] = []
    span = generated_span(text, prompt)

    if not generated_ids:
        flags.append("zero-tokens")
    if set(generated_ids) & set(blocked_token_ids):
        flags.append("blocked-special-leaked")

    stripped = span.rstrip(REPLACEMENT_CHAR)
    if REPLACEMENT_CHAR in stripped:
        flags.append("replacement-char")  # mojibake inside the text, not at the cut
    elif span != stripped:
        (notes if stop_reason == "length" else flags).append(
            "trailing-partial-char" if stop_reason == "length" else "replacement-char"
        )

    if stop_reason == "eod" and not span.strip():
        notes.append("immediate-eod")

    ratio = distinct_4gram_ratio(span)
    # Heuristic, not a law: only meaningful on longer spans. Measured span 0.59
    # (99 chars, visibly looping) vs >=0.95 elsewhere in the 2026-10-03 run.
    if ratio is not None and len(span) >= 40 and ratio < 0.7:
        notes.append(f"low-distinctness={ratio:.2f}")
    return flags, notes



def pair_summary(results: list[dict]) -> list[str]:
    """Report criterion 3: simplified/traditional prompt token-cost ratio per pair."""
    groups: dict[str, dict[str, dict]] = {}
    for result in results:
        if result["pair"]:
            groups.setdefault(result["pair"], {})[result["script"]] = result
    lines = []
    for pair in sorted(groups):
        variants = groups[pair]
        simplified, traditional = variants.get("simplified"), variants.get("traditional")
        if not simplified or not traditional:
            continue
        low, high = TRADITIONAL_RATIO_BAND
        ratio = traditional["prompt_tokens"] / max(simplified["prompt_tokens"], 1)
        marker = "" if low <= ratio <= high else "  <- out of expected band"
        lines.append(
            f"  {pair}: simp={simplified['prompt_tokens']} trad={traditional['prompt_tokens']}"
            f" ratio={ratio:.2f}{marker}"
        )
    return lines


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def resolve_checkpoint(run_dir: Path, value: str) -> Path:
    if value == "latest":
        return latest_checkpoint(run_dir)
    checkpoint = Path(value)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    return checkpoint


def resolve_dtype(name: str | None, device: torch.device) -> torch.dtype:
    if name is None:
        return torch.float32 if device.type == "cpu" else torch.float16
    dtype = DTYPE_BY_NAME[name]
    # V100 (Volta) has no bf16 tensor cores: an fp32 emulation that is slower
    # than fp16 and is banned on this project's V100 path.
    if dtype is torch.bfloat16 and device.type == "cuda":
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise RuntimeError("bfloat16 is not hardware-accelerated on this GPU; use float16")
    if dtype is torch.bfloat16 and device.type == "cpu":
        raise RuntimeError("bfloat16 CPU inference is not supported; use float32")
    return dtype


def load_checkpoint_metadata(checkpoint: Path) -> tuple[dict, ModelConfig]:
    state_path = checkpoint / "train_state.json"
    weights_path = checkpoint / "training.pt"
    if not state_path.exists() or not weights_path.exists():
        if (checkpoint / "model.safetensors").exists():
            raise FileNotFoundError(
                f"{checkpoint} is an MLX checkpoint; use jishui.generate_mlx or generate_mlx.py"
            )
        raise FileNotFoundError(f"incomplete CUDA checkpoint: {checkpoint}")
    with state_path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    return state, ModelConfig.from_dict(state["model"])


def load_model(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    state, model_config = load_checkpoint_metadata(checkpoint)
    model = JishuiForCausalLM(model_config)
    bundle = torch.load(
        checkpoint / "training.pt", map_location="cpu", weights_only=True, mmap=True
    )
    if bundle.get("format_version") != 1:
        raise ValueError("unsupported CUDA checkpoint format")
    model.load_state_dict(bundle["model"], strict=True)
    del bundle  # drop the optimizer/scaler references before moving to the device
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, state


@torch.inference_mode()
def next_token_logits(
    model: JishuiForCausalLM, context: Sequence[int], device: torch.device
) -> np.ndarray:
    tokens = torch.as_tensor([list(context)], dtype=torch.long, device=device)
    logits = model(tokens)[0, -1]
    return logits.float().cpu().numpy()


@torch.inference_mode()
def generate_one(
    model: JishuiForCausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    repetition_window: int,
    seed: int,
    blocked_token_ids: Sequence[int],
    eos_token_id: int | None,
    top_logits: int = 0,
) -> tuple[list[int], str, float]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not prompt_ids:
        if eos_token_id is None:
            raise ValueError("an empty prompt requires an EOD token in the tokenizer")
        prompt_ids = [eos_token_id]

    generated: list[int] = []
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    stop_reason = "length"
    limit = model.config.max_position_embeddings
    for index in range(max_new_tokens):
        history = [*prompt_ids, *generated]
        context = history[-limit:]
        logits = next_token_logits(model, context, device)
        if index == 0 and top_logits > 0:
            order = np.argsort(logits)[::-1][:top_logits]
            print("[top-logits after prompt]")
            for rank, token_id in enumerate(order, 1):
                text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
                print(f"  {rank:>2}. {tokenizer.id_to_token(int(token_id))!r} {text!r} "
                      f"logit={logits[token_id]:.3f}")
        token_id = sample_next_token(
            logits,
            rng,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            history=history,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            blocked_token_ids=blocked_token_ids,
        )
        generated.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            stop_reason = "eod"
            break
    return [*prompt_ids, *generated], stop_reason, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    cases = load_cases(args)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    checkpoint = resolve_checkpoint(args.run_dir, args.checkpoint)

    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    blocked_token_ids = []
    if not args.allow_all_special_tokens:
        blocked_token_ids = [
            token_id
            for token in UNTRAINED_SPECIAL_TOKENS
            if (token_id := tokenizer.token_to_id(token)) is not None
        ]

    for case in cases:
        case["prompt_tokens"] = check_prompt(tokenizer, case)

    load_started = time.perf_counter()
    model, state = load_model(checkpoint, device, dtype)
    load_seconds = time.perf_counter() - load_started
    parameters = sum(parameter.numel() for parameter in model.parameters())

    print(
        f"checkpoint={checkpoint} step={state.get('step')} "
        f"tokens={state.get('tokens_seen', 0):,} parameters={parameters:,} "
        f"device={device} dtype={str(dtype).replace('torch.', '')} load={load_seconds:.2f}s"
    )
    print(
        f"sampling: temperature={args.temperature:g} top_p={args.top_p:g} "
        f"top_k={args.top_k} repetition_penalty={args.repetition_penalty:g} seed={args.seed}"
    )
    if args.prompt_file:
        print(f"suite={args.prompt_file} cases={len(cases)}")

    results = []
    for index, case in enumerate(cases, 1):
        max_new_tokens = case["max_new_tokens"] or args.max_new_tokens
        output_ids, stop_reason, elapsed = generate_one(
            model,
            tokenizer,
            case["prompt"],
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            seed=args.seed + index - 1,
            blocked_token_ids=blocked_token_ids,
            eos_token_id=eos_token_id,
            top_logits=args.top_logits,
        )
        generated_ids = output_ids[case["prompt_tokens"] or 1:]
        generated_count = len(generated_ids)
        text = tokenizer.decode(
            output_ids,
            skip_special_tokens=not args.show_special_tokens,
        )
        flags, notes = analyze_output(
            text, case["prompt"], generated_ids, blocked_token_ids, stop_reason
        )
        results.append({
            "id": case["id"], "pair": case["pair"], "script": case["script"],
            "genre": case["genre"], "prompt": case["prompt"],
            "prompt_tokens": case["prompt_tokens"], "output": text,
            "generated_tokens": generated_count, "stop": stop_reason,
            "distinct_4gram": distinct_4gram_ratio(generated_span(text, case["prompt"])),
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(generated_count / max(elapsed, 1e-9), 2),
            "flags": flags, "notes": notes,
        })
        print("\n" + "=" * 80)
        print(f"[{index}/{len(cases)}] id={case['id']} script={case['script']} "
              f"genre={case['genre']} prompt_tokens={case['prompt_tokens']}")
        print(f"  prompt={case['prompt']!r}")
        print(text)
        print(
            f"[generated={generated_count} stop={stop_reason} elapsed={elapsed:.2f}s "
            f"speed={generated_count / max(elapsed, 1e-9):.2f} tok/s"
            f"{' flags=' + ','.join(flags) if flags else ''}"
            f"{' notes=' + ','.join(notes) if notes else ''}]"
        )

    failed = [result for result in results if result["flags"]]
    noted = [result for result in results if result["notes"]]
    ratios = pair_summary(results)
    if ratios:
        print("\n" + "=" * 80)
        print("[prompt token cost, simplified vs traditional]")
        for line in ratios:
            print(line)
    print(
        f"\nsummary: cases={len(results)} flagged={len(failed)} noted={len(noted)} "
        f"total_generated={sum(result['generated_tokens'] for result in results)}"
    )

    if args.output_json:
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "checkpoint": str(checkpoint),
            "step": state.get("step"),
            "tokens_seen": state.get("tokens_seen"),
            "parameters": parameters,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "load_seconds": round(load_seconds, 3),
            "sampling": {
                "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "repetition_window": args.repetition_window,
                "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                "blocked_special_tokens": [tokenizer.id_to_token(t) for t in blocked_token_ids],
            },
            "suite": str(args.prompt_file) if args.prompt_file else None,
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.output_json}")

    if failed:
        raise SystemExit(f"suite failed structural checks: "
                         f"{', '.join(result['id'] for result in failed)}")


if __name__ == "__main__":
    main()
