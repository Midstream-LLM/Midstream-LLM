from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from tokenizers import Tokenizer

from .checkpoints import latest_checkpoint
from .config import ModelConfig


DTYPE_NAMES = {"float32", "float16", "bfloat16"}
UNTRAINED_SPECIAL_TOKENS = (
    "<|pad|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a Jishui MLX checkpoint")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/jishui-200m-stage0-1b"))
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint directory or 'latest' under --run-dir",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer/ccc-bbpe-32k"),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to complete; repeat the option for multiple prompts",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-all-special-tokens",
        action="store_true",
        help="Do not mask pad/unk/chat tokens that were absent from pretraining",
    )
    parser.add_argument("--show-special-tokens", action="store_true")
    return parser.parse_args()


def sample_next_token(
    logits: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    history: Sequence[int] = (),
    repetition_penalty: float = 1.0,
    repetition_window: int = 0,
    blocked_token_ids: Sequence[int] = (),
) -> int:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if repetition_penalty < 1:
        raise ValueError("repetition_penalty must be at least 1")
    if repetition_window < 0:
        raise ValueError("repetition_window must be non-negative")

    scores = np.asarray(logits, dtype=np.float64).copy()
    if scores.ndim != 1:
        raise ValueError(f"expected one-dimensional logits, got shape {scores.shape}")

    if repetition_penalty > 1 and repetition_window:
        for token_id in set(history[-repetition_window:]):
            if 0 <= token_id < len(scores):
                if scores[token_id] < 0:
                    scores[token_id] *= repetition_penalty
                else:
                    scores[token_id] /= repetition_penalty

    for token_id in blocked_token_ids:
        if 0 <= token_id < len(scores):
            scores[token_id] = -np.inf

    if not np.isfinite(scores).any():
        raise ValueError("all token logits were masked or non-finite")
    if temperature == 0:
        return int(np.argmax(scores))

    scores /= temperature
    token_ids = np.arange(len(scores), dtype=np.int64)
    if 0 < top_k < len(scores):
        kept = np.argpartition(scores, -top_k)[-top_k:]
        scores = scores[kept]
        token_ids = token_ids[kept]

    order = np.argsort(scores)[::-1]
    scores = scores[order]
    token_ids = token_ids[order]
    probabilities = np.exp(scores - scores[0])
    probabilities /= probabilities.sum()

    if top_p < 1:
        keep_count = int(np.searchsorted(np.cumsum(probabilities), top_p, side="left")) + 1
        scores = scores[:keep_count]
        token_ids = token_ids[:keep_count]
        probabilities = np.exp(scores - scores[0])
        probabilities /= probabilities.sum()

    return int(rng.choice(token_ids, p=probabilities))


def resolve_checkpoint(run_dir: Path, value: str) -> Path:
    if value == "latest":
        return latest_checkpoint(run_dir)
    checkpoint = Path(value)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    return checkpoint


def load_checkpoint_metadata(checkpoint: Path) -> tuple[dict, ModelConfig, str]:
    state_path = checkpoint / "train_state.json"
    weights_path = checkpoint / "model.safetensors"
    if not state_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"incomplete checkpoint: {checkpoint}")
    with state_path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    model_config = ModelConfig.from_dict(state["model"])
    dtype = state.get("training", {}).get("dtype", "float16")
    if dtype not in DTYPE_NAMES:
        raise ValueError(f"unsupported checkpoint dtype: {dtype}")
    return state, model_config, dtype


def generate_one(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    repetition_window: int,
    seed: int,
    blocked_token_ids: Sequence[int],
    eos_token_id: int | None,
) -> tuple[list[int], str, float]:
    import mlx.core as mx

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not prompt_ids:
        if eos_token_id is None:
            raise ValueError("an empty prompt requires an EOD token in the tokenizer")
        prompt_ids = [eos_token_id]

    generated: list[int] = []
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    stop_reason = "length"
    for _ in range(max_new_tokens):
        history = [*prompt_ids, *generated]
        context = history[-model.config.max_position_embeddings :]
        logits = model(mx.array([context], dtype=mx.int32))[0, -1]
        mx.eval(logits)
        token_id = sample_next_token(
            np.asarray(logits),
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
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    prompts = args.prompts or ["太史公曰："]

    import mlx.core as mx

    from .model import JishuiForCausalLM

    checkpoint = resolve_checkpoint(args.run_dir, args.checkpoint)
    state, model_config, dtype_name = load_checkpoint_metadata(checkpoint)
    dtype_by_name = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    blocked_token_ids = []
    if not args.allow_all_special_tokens:
        blocked_token_ids = [
            token_id
            for token in UNTRAINED_SPECIAL_TOKENS
            if (token_id := tokenizer.token_to_id(token)) is not None
        ]

    load_started = time.perf_counter()
    model = JishuiForCausalLM(model_config)
    model.set_dtype(dtype_by_name[dtype_name])
    model.load_weights(str(checkpoint / "model.safetensors"), strict=True)
    model.eval()
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - load_started

    print(
        f"checkpoint={checkpoint} step={state['step']} tokens={state['tokens_seen']:,} "
        f"parameters={model.num_parameters():,} dtype={dtype_name} load={load_seconds:.2f}s"
    )
    print(
        f"sampling: temperature={args.temperature:g} top_p={args.top_p:g} "
        f"top_k={args.top_k} repetition_penalty={args.repetition_penalty:g} seed={args.seed}"
    )
    for index, prompt in enumerate(prompts, 1):
        output_ids, stop_reason, elapsed = generate_one(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            seed=args.seed + index - 1,
            blocked_token_ids=blocked_token_ids,
            eos_token_id=eos_token_id,
        )
        prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False).ids) or 1
        generated_count = len(output_ids) - prompt_token_count
        text = tokenizer.decode(
            output_ids,
            skip_special_tokens=not args.show_special_tokens,
        )
        print("\n" + "=" * 80)
        print(f"[{index}] prompt={prompt!r}")
        print(text)
        print(
            f"[generated={generated_count} stop={stop_reason} elapsed={elapsed:.2f}s "
            f"speed={generated_count / max(elapsed, 1e-9):.2f} tok/s]"
        )


if __name__ == "__main__":
    main()
