"""Single-GPU supervised fine-tuning on a compact SFT export.

Reads the layout written by scripts/sft/export_compact_sft.py (input_ids.npy,
loss_mask.npy, sample_offsets.npy, {train,val}_indices.npy, records.parquet) and
starts from a pretraining CUDA checkpoint's weights. Loss is token-weighted over
supervised (assistant + <|im_end|>) tokens only. Micro-batches are built by a
token budget from length-sorted pools so padding stays small; the whole batch
plan is a pure function of (seed, epoch), so --resume restarts the exact step.
Checkpoints use the pretraining format, so jishui.generate_torch can load them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoints import latest_checkpoint
from .checkpoints_torch import load_checkpoint, load_model_weights, save_checkpoint
from .config import ModelConfig
from .model_torch import JishuiForCausalLM

LOG = logging.getLogger("jishui.sft_torch")
IM_START, IM_END, PAD = 3, 4, 2


class CompactSFT:
    def __init__(self, root: Path):
        self.root = root
        self.ids = np.load(root / "input_ids.npy", mmap_mode="r")
        self.mask = np.load(root / "loss_mask.npy", mmap_mode="r")
        self.offsets = np.load(root / "sample_offsets.npy")
        self.splits = {s: np.load(root / f"{s}_indices.npy") for s in ("train", "val", "test")}
        self.lengths = np.diff(self.offsets)
        if len(self.ids) != len(self.mask) or int(self.offsets[-1]) != len(self.ids):
            raise ValueError("compact SFT arrays disagree")
        self.tasks = None
        parquet = root / "records.parquet"
        if parquet.exists():
            import pyarrow.parquet as pq
            self.tasks = pq.read_table(parquet, columns=["task"])["task"].to_pylist()

    def fingerprint(self) -> str:
        manifest = self.root / "manifest.json"
        return hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else "unknown"

    def sample(self, i: int):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.ids[a:b], self.mask[a:b]


def batch_plan(lengths: np.ndarray, indices: np.ndarray, token_budget: int, seed: int, epoch: int,
               pool: int = 4096) -> list[list[int]]:
    """Shuffle, sort within pools by length, cut by padded-token budget, shuffle batches."""
    rng = np.random.default_rng([seed, epoch])
    order = indices[rng.permutation(len(indices))]
    batches = []
    for start in range(0, len(order), pool):
        chunk = sorted(order[start:start + pool].tolist(), key=lambda i: lengths[i])
        current, longest = [], 0
        for i in chunk:
            n = int(lengths[i]) - 1
            if current and max(longest, n) * (len(current) + 1) > token_budget:
                batches.append(current)
                current, longest = [], 0
            current.append(i)
            longest = max(longest, n)
        if current:
            batches.append(current)
    return [batches[k] for k in rng.permutation(len(batches))]


def collate(data: CompactSFT, batch: list[int], device):
    width = max(int(data.lengths[i]) for i in batch) - 1
    x = np.full((len(batch), width), PAD, dtype=np.int64)
    y = np.full((len(batch), width), -100, dtype=np.int64)
    for r, i in enumerate(batch):
        ids, mask = data.sample(i)
        ids = ids.astype(np.int64)
        n = len(ids) - 1
        x[r, :n] = ids[:-1]
        y[r, :n] = np.where(mask[1:] == 1, ids[1:], -100)
    return torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def summed_loss(model, x, y):
    with autocast(x.device):
        logits = model(x)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                           ignore_index=-100, reduction="sum")


@torch.no_grad()
def evaluate(model, data: CompactSFT, indices: np.ndarray, token_budget: int, device) -> dict:
    """Token-weighted loss over supervised tokens, overall and per task."""
    model.eval()
    total, count = defaultdict(float), defaultdict(int)
    for batch in batch_plan(data.lengths, indices, token_budget, seed=0, epoch=0):
        x, y = collate(data, batch, device)
        with autocast(device):
            logits = model(x)
        rows = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none").sum(1)
        counts = (y != -100).sum(1)
        for i, loss, n in zip(batch, rows.tolist(), counts.tolist()):
            task = data.tasks[i] if data.tasks else "all"
            for key in (task, "all"):
                total[key] += loss
                count[key] += n
    model.train()
    return {k: total[k] / count[k] for k in sorted(total)}


@torch.no_grad()
def generate(model, prompt_ids: list[int], device, max_new: int = 160) -> list[int]:
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new):
        context = ids[-model.config.max_position_embeddings:]
        with autocast(device):
            logits = model(torch.as_tensor([context], device=device))[0, -1]
        token = int(logits.float().argmax())
        if token == IM_END:
            break
        ids.append(token)
    model.train()
    return ids[len(prompt_ids):]


def sample_generations(model, data: CompactSFT, tokenizer, device, per_task: int = 2) -> list[dict]:
    """Greedy answers for the first val samples of each task, prompt = everything before the last assistant."""
    if tokenizer is None:
        return []
    seen, out = defaultdict(int), []
    for i in data.splits["val"].tolist():
        task = data.tasks[i] if data.tasks else "all"
        if seen[task] >= per_task:
            continue
        ids, mask = data.sample(i)
        first = int(np.argmax(mask == 1))
        prompt = ids[:first].tolist()
        reference = ids[first:].tolist()
        seen[task] += 1
        out.append(dict(task=task, sample_index=i, prompt=tokenizer.decode(prompt, skip_special_tokens=False),
                        reference=tokenizer.decode([t for t in reference if t != IM_END]),
                        generated=tokenizer.decode(generate(model, prompt, device))))
    return out


def lr_at(step: int, total: int, peak: float, floor: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


PATH_KEYS = {"data_dir", "run_dir", "init_from", "tokenizer"}
# Operational switches a strict config may not pin and the CLI may always set.
RUNTIME_KEYS = {"config", "resume", "max_steps", "device", "run_dir"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_strict_config(path: Path) -> dict:
    """{"sft": {argparse keys}, "expect": {...}}. Every recipe key must be given."""
    raw = json.loads(Path(path).read_text())
    if set(raw) - {"sft", "expect", "comment"}:
        raise ValueError("strict SFT config takes only sft, expect and comment")
    return raw


def build_parser(required: bool = True):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   help="strict JSON recipe; recipe flags may not also be passed on the command line")
    p.add_argument("--data-dir", type=Path, required=required)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--init-from", type=Path, help="pretraining checkpoint directory (weights only)")
    p.add_argument("--resume", action="store_true", help="exact resume from run-dir/latest checkpoint")
    p.add_argument("--tokenizer", type=Path, help="tokenizer.json for sample generations")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--micro-tokens", type=int, default=4096, help="padded tokens per micro-batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--save-interval", type=int, default=400)
    p.add_argument("--keep-checkpoints", type=int, default=2)
    p.add_argument("--max-steps", type=int, help="stop early (smoke tests); schedule still spans all epochs")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def parse_args(argv=None):
    argv = list(argv) if argv is not None else None
    first, _ = build_parser(required=False).parse_known_args(argv)
    if first.config is None:
        args = build_parser().parse_args(argv)
        args.expect = None
        return args
    p = build_parser(required=False)
    strict = load_strict_config(first.config)
    recipe = dict(strict["sft"])
    known = {a.dest for a in p._actions} - {"help"}
    unknown = set(recipe) - known
    if unknown:
        raise ValueError(f"unknown SFT config keys: {sorted(unknown)}")
    missing = known - RUNTIME_KEYS - set(recipe)
    if missing:
        raise ValueError(f"strict config must pin every recipe key; missing {sorted(missing)}")
    explicit = p.parse_args(argv)
    defaults = p.parse_args(["--run-dir", str(explicit.run_dir)])
    for key in known - RUNTIME_KEYS:
        if getattr(explicit, key) != getattr(defaults, key):
            raise ValueError(f"--{key.replace('_', '-')} is fixed by {first.config}; do not override it")
    for key, value in recipe.items():
        setattr(explicit, key, Path(value) if key in PATH_KEYS and value is not None else value)
    explicit.expect = strict.get("expect", {})
    return explicit


def check_expectations(args, data, total_steps, steps_per_epoch, parent=None):
    """Fail before any update when the data, tokenizer, parent or plan differ from the config."""
    expect = args.expect or {}
    checks = {"data_fingerprint": data.fingerprint(), "total_steps": total_steps,
              "steps_per_epoch": steps_per_epoch, "train_samples": int(len(data.splits["train"])),
              "val_samples": int(len(data.splits["val"])), "test_samples": int(len(data.splits["test"]))}
    if args.tokenizer:
        checks["tokenizer_sha256"] = sha256_file(args.tokenizer)
    if parent is not None:
        checks["parent_step"] = parent.get("step")
        checks["parent_model"] = parent.get("model")
        if args.init_from is not None:
            checks["parent_weights_sha256"] = sha256_file(args.init_from / "training.pt")
    # On --resume the parent was verified when the run started; config_sha256
    # in the saved hyperparameters ties the resumed run to the same file.
    parent_keys = {"parent_step", "parent_model", "parent_weights_sha256"}
    unknown = set(expect) - set(checks) - (parent_keys if parent is None else set())
    if unknown:
        raise ValueError(f"unknown expect keys: {sorted(unknown)}")
    for key, wanted in expect.items():
        if key not in checks:
            continue
        if checks[key] != wanted:
            raise ValueError(f"expectation failed for {key}: config {wanted!r}, actual {checks[key]!r}")
    if expect:
        LOG.info("strict config checks passed: %s", sorted(expect))


def run(args) -> dict:
    device = torch.device(args.device)
    data = CompactSFT(args.data_dir)
    hyper = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
             if k not in ("resume", "max_steps", "device", "run_dir", "init_from", "expect")}
    if args.config is not None:
        hyper["config_sha256"] = sha256_file(args.config)
    plans = [batch_plan(data.lengths, data.splits["train"], args.micro_tokens, args.seed, e) for e in range(args.epochs)]
    steps_per_epoch = [math.ceil(len(p) / args.grad_accum) for p in plans]
    total_steps = sum(steps_per_epoch)

    if args.resume:
        check_expectations(args, data, total_steps, steps_per_epoch)
        checkpoint = latest_checkpoint(args.run_dir)
        state = json.loads((checkpoint / "train_state.json").read_text())
        config = ModelConfig.from_dict(state["model"])
    else:
        if args.init_from is None:
            raise ValueError("--init-from is required for a new run")
        if (args.run_dir / "checkpoints").exists():
            raise FileExistsError(f"{args.run_dir} already has checkpoints; use --resume")
        parent = json.loads((args.init_from / "train_state.json").read_text())
        check_expectations(args, data, total_steps, steps_per_epoch, parent)
        config = ModelConfig.from_dict(parent["model"])
    if args.gradient_checkpointing:
        config = ModelConfig.from_dict({**config.to_dict(), "gradient_checkpointing": True})
    torch.manual_seed(args.seed)
    model = JishuiForCausalLM(config).to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.95), eps=1e-5, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=256.0)

    if args.resume:
        state = load_checkpoint(checkpoint, model, optimizer, scaler)
        if state["sft"]["hyperparameters"] != hyper or state["sft"]["data_fingerprint"] != data.fingerprint():
            raise ValueError("resume must use the same data and hyperparameters")
        step = state["step"]
    else:
        parent_state = load_model_weights(args.init_from, model)
        step = 0
        state = {"model": config.to_dict(), "step": 0, "sft": {
            "hyperparameters": hyper, "data_fingerprint": data.fingerprint(),
            "parent_checkpoint": str(args.init_from.resolve()), "parent_step": parent_state.get("step"),
            "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
            "train_samples": int(len(data.splits["train"])), "history": []}}
        args.run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    log_path = args.run_dir / "metrics.jsonl"

    def record(entry):
        state["sft"]["history"].append(entry)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        LOG.info(json.dumps(entry, ensure_ascii=False))

    def save():
        state["step"] = step
        save_checkpoint(args.run_dir, state, model, optimizer, scaler, keep=args.keep_checkpoints)

    if step == 0:
        record({"step": 0, "val_loss": evaluate(model, data, data.splits["val"], args.micro_tokens, device)})

    model.train()
    stop = args.max_steps if args.max_steps is not None else total_steps
    window = defaultdict(float)
    started = time.time()
    while step < min(stop, total_steps):
        epoch, local = step, 0
        for e, n in enumerate(steps_per_epoch):
            if epoch < n:
                epoch, local = e, epoch
                break
            epoch -= n
        micro = plans[epoch][local * args.grad_accum:(local + 1) * args.grad_accum]
        lr = lr_at(step, total_steps, args.lr, args.min_lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        targets = sum(int(sum(int(data.mask[int(data.offsets[i]) + 1:int(data.offsets[i + 1])].sum())
                              for i in batch)) for batch in micro)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for batch in micro:
            x, y = collate(data, batch, device)
            loss = summed_loss(model, x, y) / targets
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
            window["padded"] += x.numel()
        scaler.unscale_(optimizer)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        step += 1
        window["loss"] += loss_sum
        window["targets"] += targets
        window["steps"] += 1
        if step % 10 == 0 or step == total_steps:
            elapsed = time.time() - started
            record({"step": step, "epoch": epoch, "lr": lr, "loss": window["loss"] / window["steps"],
                    "grad_norm": norm, "loss_scale": scaler.get_scale(),
                    "supervised_tokens_per_s": window["targets"] / elapsed,
                    "padded_tokens_per_s": window["padded"] / elapsed,
                    "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None})
            window.clear()
            started = time.time()
        if step % args.eval_interval == 0 or step == total_steps:
            record({"step": step, "val_loss": evaluate(model, data, data.splits["val"], args.micro_tokens, device)})
        if step % args.save_interval == 0 or step == total_steps:
            save()
    if step == total_steps:
        samples = sample_generations(model, data, tokenizer, device)
        (args.run_dir / "val_generations.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2))
        state["sft"]["complete"] = True
        (args.run_dir / "final.json").write_text(json.dumps(
            {"step": step, "checkpoint": str(latest_checkpoint(args.run_dir)),
             "final_val_loss": state["sft"]["history"][-1].get("val_loss")}, ensure_ascii=False, indent=2))
    elif args.max_steps is not None and step % args.save_interval:
        save()
    return state


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
