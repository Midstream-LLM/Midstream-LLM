from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .config import ModelConfig, TrainConfig, load_run_config
from .checkpoints import latest_checkpoint, prune_checkpoints
from .data import (
    PackedBatchIterator,
    PretrainIndex,
    SequentialBatchIterator,
    load_or_build_index,
    load_sampling_config,
)
from .model import JishuiForCausalLM, causal_lm_loss


LOG = logging.getLogger("jishui.train_mlx")
DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Jishui with Apple MLX")
    parser.add_argument("--config", type=Path, default=Path("configs/jishui-200m-mlx.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/jishui-200m-mlx"))
    parser.add_argument("--index-cache", type=Path)
    parser.add_argument("--resume", help="checkpoint directory or 'latest'")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--dtype", choices=tuple(DTYPES))
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--save-interval", type=int)
    return parser.parse_args()


def apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    updates = {}
    for name in (
        "max_steps",
        "seq_len",
        "batch_size",
        "grad_accum_steps",
        "dtype",
        "log_interval",
        "eval_interval",
        "save_interval",
    ):
        value = getattr(args, name)
        if value is not None:
            updates[name] = value
    return replace(config, **updates)


def make_schedule(config: TrainConfig):
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    cosine = optim.cosine_decay(
        config.learning_rate,
        decay_steps,
        end=config.min_learning_rate,
    )
    if config.warmup_steps == 0:
        return cosine
    # The optimizer evaluates a schedule at step zero before its first update.
    # Start at one warmup fraction so the first batch is not discarded.
    warmup = optim.linear_schedule(
        config.learning_rate / config.warmup_steps,
        config.learning_rate,
        max(1, config.warmup_steps - 1),
    )
    return optim.join_schedules([warmup, cosine], [config.warmup_steps])


def make_optimizer(config: TrainConfig) -> optim.AdamW:
    return optim.AdamW(
        learning_rate=make_schedule(config),
        betas=[config.beta1, config.beta2],
        # 1e-8 underflows in fp16 and can turn the first Adam update into NaN.
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        bias_correction=True,
    )


def save_checkpoint(
    run_dir: Path,
    step: int,
    tokens_seen: int,
    model: JishuiForCausalLM,
    optimizer: optim.Optimizer,
    sampler: PackedBatchIterator,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Path:
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_root / f"step_{step:08d}"
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = checkpoint_root / f".step_{step:08d}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    mx.eval(model.parameters(), optimizer.state)
    model.save_weights(str(temporary / "model.safetensors"))
    optimizer_flat = tree_flatten(optimizer.state, destination={})
    mx.savez(str(temporary / "optimizer.npz"), **optimizer_flat)
    state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "sampler": sampler.state_dict(),
        "model": model_config.to_dict(),
        "training": train_config.to_dict(),
    }
    with (temporary / "train_state.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    temporary.rename(destination)
    with (run_dir / "latest.json").open("w", encoding="utf-8") as handle:
        json.dump({"checkpoint": str(destination), "step": step}, handle, indent=2)
    prune_checkpoints(
        run_dir,
        train_config.max_checkpoints,
        optimizer_checkpoints=train_config.optimizer_checkpoints,
    )
    return destination


def load_checkpoint(
    checkpoint: Path,
    model: JishuiForCausalLM,
    optimizer: optim.Optimizer,
    sampler: PackedBatchIterator,
    model_config: ModelConfig,
) -> tuple[int, int]:
    with (checkpoint / "train_state.json").open(encoding="utf-8") as handle:
        state = json.load(handle)
    if state["model"] != model_config.to_dict():
        raise ValueError("checkpoint model config does not match the requested model")
    model.load_weights(str(checkpoint / "model.safetensors"), strict=True)
    optimizer_path = checkpoint / "optimizer.npz"
    if optimizer_path.exists():
        optimizer_values = list(mx.load(str(optimizer_path)).items())
        optimizer.state = tree_unflatten(optimizer_values)
    else:
        # Older retained snapshots may intentionally omit Adam state to save
        # disk. The model and sampler can still be restored, but optimization
        # resumes with fresh moments rather than being mathematically exact.
        LOG.warning("checkpoint %s has no optimizer.npz; resetting Adam state", checkpoint)
        optimizer.state = {}
    optimizer.init(model.trainable_parameters())
    sampler.load_state_dict(state["sampler"])
    mx.eval(model.parameters(), optimizer.state)
    return int(state["step"]), int(state["tokens_seen"])


def evaluate(
    model: JishuiForCausalLM,
    index: PretrainIndex,
    config: TrainConfig,
) -> float:
    model.eval()
    iterator = SequentialBatchIterator(
        index,
        split="val",
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        eod_token_id=config.eod_token_id,
    )
    loss_fn = lambda inputs, targets: causal_lm_loss(model, inputs, targets)
    total = 0.0
    for _ in range(config.eval_batches):
        batch = next(iterator)
        inputs = mx.array(batch[:, :-1])
        targets = mx.array(batch[:, 1:])
        loss = loss_fn(inputs, targets)
        mx.eval(loss)
        total += float(loss.item())
    model.train()
    return total / config.eval_batches


def make_loss_and_grad(model: JishuiForCausalLM, config: TrainConfig):
    loss_fn = lambda inputs, targets: causal_lm_loss(model, inputs, targets)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    if not config.compile_step:
        return loss_and_grad
    # Compile one microbatch at a time. Compiling the entire accumulation loop
    # would retain all intermediate activations and defeat memory savings.
    return mx.compile(loss_and_grad, inputs=[model.state], outputs=[model.state])


def train(
    model: JishuiForCausalLM,
    optimizer: optim.Optimizer,
    sampler: PackedBatchIterator,
    index: PretrainIndex,
    model_config: ModelConfig,
    config: TrainConfig,
    run_dir: Path,
    start_step: int = 0,
    tokens_seen: int = 0,
) -> None:
    loss_and_grad = make_loss_and_grad(model, config)
    model.train()
    last_log_time = time.perf_counter()
    interval_tokens = 0
    last_checkpoint_step = start_step

    for step in range(start_step + 1, config.max_steps + 1):
        accumulated_grads = None
        accumulated_loss = 0.0
        for _ in range(config.grad_accum_steps):
            batch = next(sampler)
            inputs = mx.array(batch[:, :-1])
            targets = mx.array(batch[:, 1:])
            loss, grads = loss_and_grad(inputs, targets)
            mx.eval(loss, grads)
            accumulated_loss += float(loss.item())
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(
                    lambda current, new: current + new,
                    accumulated_grads,
                    grads,
                )
                mx.eval(accumulated_grads)

        grads = tree_map(lambda value: value / config.grad_accum_steps, accumulated_grads)
        grads, grad_norm = optim.clip_grad_norm(grads, config.grad_clip)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, grad_norm)

        step_tokens = config.batch_size * config.seq_len * config.grad_accum_steps
        tokens_seen += step_tokens
        interval_tokens += step_tokens
        mean_loss = accumulated_loss / config.grad_accum_steps
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {mean_loss}")

        if step % config.log_interval == 0:
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1.0e-9)
            learning_rate = float(optimizer.learning_rate.item())
            LOG.info(
                "step=%d loss=%.4f ppl=%.2f lr=%.3e grad_norm=%.3f "
                "tokens=%d tok/s=%.0f",
                step,
                mean_loss,
                math.exp(min(mean_loss, 20.0)),
                learning_rate,
                float(grad_norm.item()),
                tokens_seen,
                interval_tokens / elapsed,
            )
            last_log_time = now
            interval_tokens = 0

        if config.eval_interval > 0 and step % config.eval_interval == 0:
            validation_loss = evaluate(model, index, config)
            LOG.info(
                "validation step=%d loss=%.4f ppl=%.2f",
                step,
                validation_loss,
                math.exp(min(validation_loss, 20.0)),
            )

        if config.save_interval > 0 and step % config.save_interval == 0:
            checkpoint = save_checkpoint(
                run_dir,
                step,
                tokens_seen,
                model,
                optimizer,
                sampler,
                model_config,
                config,
            )
            last_checkpoint_step = step
            LOG.info("saved checkpoint %s", checkpoint)

    if last_checkpoint_step != config.max_steps:
        checkpoint = save_checkpoint(
            run_dir,
            config.max_steps,
            tokens_seen,
            model,
            optimizer,
            sampler,
            model_config,
            config,
        )
        LOG.info("saved final checkpoint %s", checkpoint)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    model_config, train_config = load_run_config(args.config)
    train_config = apply_overrides(train_config, args)
    if train_config.seq_len > model_config.max_position_embeddings:
        raise ValueError(
            f"seq_len={train_config.seq_len} exceeds model maximum "
            f"{model_config.max_position_embeddings}"
        )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    index_cache = args.index_cache or args.run_dir / "cache" / "pretrain_index.npz"
    index, rebuilt = load_or_build_index(args.data_dir, index_cache)
    LOG.info("pretrain index: %s (%s)", index_cache, "rebuilt" if rebuilt else "cached")
    LOG.info("index repairs: %s", json.dumps(index.metadata, ensure_ascii=False))
    LOG.info("usable train tokens by category: %s", index.token_totals("train"))

    sampling = load_sampling_config(args.data_dir)
    sampler = PackedBatchIterator(
        index,
        sampling,
        seq_len=train_config.seq_len,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
        mode=train_config.sampling_mode,
        eod_token_id=train_config.eod_token_id,
    )
    probabilities = {
        int(category): round(float(probability), 6)
        for category, probability in zip(
            sampler.category_ids, sampler.category_probabilities, strict=True
        )
    }
    LOG.info("sampling mode=%s probabilities=%s", train_config.sampling_mode, probabilities)

    mx.random.seed(train_config.seed)
    model = JishuiForCausalLM(model_config)
    model.set_dtype(DTYPES[train_config.dtype])
    mx.eval(model.parameters())
    LOG.info(
        "model=%s parameters=%s dtype=%s device=%s",
        model_config.name,
        f"{model.num_parameters():,}",
        train_config.dtype,
        mx.default_device(),
    )
    optimizer = make_optimizer(train_config)
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)

    start_step = 0
    tokens_seen = 0
    if args.resume:
        checkpoint = latest_checkpoint(args.run_dir) if args.resume == "latest" else Path(args.resume)
        start_step, tokens_seen = load_checkpoint(
            checkpoint, model, optimizer, sampler, model_config
        )
        LOG.info("resumed %s at step=%d tokens=%d", checkpoint, start_step, tokens_seen)
    if start_step >= train_config.max_steps:
        raise ValueError(
            f"checkpoint step {start_step} is not below max_steps {train_config.max_steps}"
        )

    resolved = {"model": model_config.to_dict(), "training": train_config.to_dict()}
    with (args.run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved, handle, ensure_ascii=False, indent=2)
    train(
        model,
        optimizer,
        sampler,
        index,
        model_config,
        train_config,
        args.run_dir,
        start_step=start_step,
        tokens_seen=tokens_seen,
    )


if __name__ == "__main__":
    main()
