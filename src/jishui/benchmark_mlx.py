from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from .config import load_run_config
from .model import JishuiForCausalLM
from .train_mlx import DTYPES, make_loss_and_grad, make_optimizer, optimizer_gradients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoint-free MLX training throughput benchmark"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/jishui-200m-mlx.json"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--microsteps", type=int, default=16)
    parser.add_argument("--warmup-microsteps", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--start-file", type=Path)
    parser.add_argument("--barrier-timeout", type=float, default=300.0)
    return parser.parse_args()


def load_records(path: Path, seq_len: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<u2")
    width = seq_len + 1
    if values.size == 0 or values.size % width:
        raise ValueError(
            f"{path} has {values.size} uint16 values; expected a multiple of {width}"
        )
    return values.reshape(-1, width).astype(np.int32, copy=False)


def wait_at_barrier(
    ready_path: Path | None, start_path: Path | None, timeout: float
) -> None:
    if ready_path is None and start_path is None:
        return
    if ready_path is None or start_path is None:
        raise ValueError("--ready-file and --start-file must be used together")
    if timeout <= 0:
        raise ValueError("--barrier-timeout must be positive")
    descriptor = os.open(ready_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"pid={os.getpid()}\n")
    deadline = time.monotonic() + timeout
    while not start_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {start_path}")
        time.sleep(0.01)


def run_update(
    model: JishuiForCausalLM,
    optimizer: optim.Optimizer,
    loss_and_grad,
    records: np.ndarray,
    first_microstep: int,
    microsteps: int,
) -> float:
    accumulated_grads = None
    accumulated_loss = 0.0
    for offset in range(microsteps):
        record = records[(first_microstep + offset) % len(records)]
        inputs = mx.array(record[:-1][None, :])
        targets = mx.array(record[1:][None, :])
        loss, grads = loss_and_grad(inputs, targets)
        mx.eval(loss, grads)
        accumulated_loss += float(loss.item())
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = tree_map(
                lambda current, new: current + new, accumulated_grads, grads
            )
            mx.eval(accumulated_grads)

    grads = tree_map(lambda value: value / microsteps, accumulated_grads)
    grads, grad_norm = optim.clip_grad_norm(grads, 1.0)
    optimizer.update(model, optimizer_gradients(model, grads))
    mx.eval(model.parameters(), optimizer.state, grad_norm)
    return accumulated_loss / microsteps


def main() -> None:
    args = parse_args()
    model_config, train_config = load_run_config(args.config)
    accumulation = args.grad_accum_steps or train_config.grad_accum_steps
    if min(args.microsteps, args.warmup_microsteps, accumulation) < 1:
        raise ValueError("microstep and accumulation counts must be positive")
    if args.microsteps % accumulation:
        raise ValueError("--microsteps must be divisible by --grad-accum-steps")
    records = load_records(args.records, train_config.seq_len)

    mx.random.seed(train_config.seed)
    model = JishuiForCausalLM(model_config)
    model.set_dtype(DTYPES[train_config.dtype])
    mx.eval(model.parameters())
    optimizer = make_optimizer(train_config)
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)
    loss_and_grad = make_loss_and_grad(model, train_config)

    warmup_done = 0
    while warmup_done < args.warmup_microsteps:
        count = min(accumulation, args.warmup_microsteps - warmup_done)
        run_update(model, optimizer, loss_and_grad, records, warmup_done, count)
        warmup_done += count

    wait_at_barrier(args.ready_file, args.start_file, args.barrier_timeout)
    started = time.perf_counter()
    last_loss = math.nan
    for first in range(0, args.microsteps, accumulation):
        last_loss = run_update(
            model, optimizer, loss_and_grad, records, first, accumulation
        )
    elapsed = time.perf_counter() - started
    tokens = args.microsteps * train_config.seq_len
    print(
        json.dumps(
            {
                "worker": "mlx",
                "microsteps": args.microsteps,
                "tokens": tokens,
                "seconds": elapsed,
                "tokens_per_second": tokens / elapsed,
                "loss": last_loss,
                "grad_accum_steps": accumulation,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
