"""Pretraining, new-corpus initialization and exact continuation.

One GPU: ``python -m jishui.pretrain_torch``. Several GPUs (shuffled_blocks
only): ``torchrun --nproc_per_node=N -m jishui.pretrain_torch``. Every rank
draws the same global micro-batch (batch_size * world_size rows) from one
deterministic sampler and keeps its own row slice, so sampler state, token
counts and checkpoints are identical to a single process reading that batch.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import logging
import math
import os
import signal
import time
from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoints import latest_checkpoint
from .checkpoints_torch import import_mlx, load_checkpoint, load_model_weights, save_checkpoint
from .config import load_run_config
from .data import PackedBatchIterator, SequentialBatchIterator, ShuffledDocumentIterator, load_or_build_index, load_sampling_config
from .block_data import SampleSpec, TokenBatch, ShuffledBlockIterator, FixedValidation
from .model_torch import JishuiForCausalLM, causal_lm_loss
from .schedules_torch import continuation_lr, make_continuation_schedule, make_pretraining_schedule
from .tracking import ExperimentTracker

LOG = logging.getLogger("jishui.pretrain_torch")


def natural_sampling(index) -> dict:
    totals = index.token_totals("train")
    total = sum(totals.values())
    if total <= 0:
        raise ValueError("empty training set")
    return {"target_mix": {str(k): v / total for k, v in totals.items()}}


ATTENTION_MODES = ("causal", "document")


def world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def rows(batch, rank, size):
    """This rank's contiguous slice of a global TokenBatch."""
    if size == 1:
        return batch
    n = len(batch.tokens) // size
    part = slice(rank*n, (rank+1)*n)
    return TokenBatch(batch.tokens[part], batch.document_ids[part], batch.target_counts[part],
                      batch.target_starts[part], batch.spec)


def segment_ids(batch):
    """Attention segment of each input position = document of its target.

    EOD belongs to the document it ends, so the EOD input position joins the
    next document and acts as its BOS; a document never attends to another.
    Padding (-1) only attends to padding and is never a target.
    """
    return batch.document_ids[:, 1:batch.spec.seq_len+1]


def tensor_batch(batch, device, document_attention=False):
    if isinstance(batch, TokenBatch):
        targets, _ = batch.targets(1)
        segments = (torch.as_tensor(np.ascontiguousarray(segment_ids(batch)), device=device)
                    if document_attention else None)
        return (torch.as_tensor(batch.input_ids, device=device),
                torch.as_tensor(targets, device=device), segments)
    if document_attention:
        raise ValueError("document attention requires shuffled_blocks batches")
    tokens = torch.as_tensor(batch.astype(np.int64), device=device)
    return tokens[:, :-1], tokens[:, 1:], None


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate(model, index, config, device, category=None) -> float:
    model.eval()
    iterator = SequentialBatchIterator(index, "val", config.seq_len, config.batch_size,
                                       config.eod_token_id, category=category)
    losses = []
    for _ in range(config.eval_batches):
        x, y, _ = tensor_batch(next(iterator), device)
        with autocast(device):
            loss = causal_lm_loss(model, x, y)
        losses.append(float(loss))
    model.train()
    result = sum(losses) / len(losses)
    if not math.isfinite(result):
        raise FloatingPointError("non-finite validation loss")
    return result


def validation_report(model, index, config, device, step, max_categories=None, fixed=None,
                      document_attention=False):
    if fixed is not None:
        rank, size = world()
        sources = sorted(fixed.source_tokens)
        totals = torch.zeros(len(sources), 2, dtype=torch.float64, device=device)
        model.eval()
        try:
            with torch.no_grad():
                # Samples are sharded round-robin; per-source sums are reduced,
                # so the result equals a single-process pass over all samples.
                for i, (source, batch) in enumerate(fixed):
                    if i % size != rank:
                        continue
                    x, y, seg = tensor_batch(batch, device, document_attention)
                    count = int((y != -100).sum())
                    with autocast(device):
                        loss = causal_lm_loss(model, x, y, seg)
                    totals[sources.index(source), 0] += float(loss)*count
                    totals[sources.index(source), 1] += count
        finally:
            model.train()
        if size > 1:
            dist.all_reduce(totals)
        sums = {s: float(totals[i, 0]) for i, s in enumerate(sources) if totals[i, 1] > 0}
        counts = {s: int(totals[i, 1]) for i, s in enumerate(sources) if totals[i, 1] > 0}
        means = {source: sums[source]/counts[source] for source in sums}
        overall = fixed.weighted_mean(means)
        if not all(math.isfinite(v) for v in [overall, *means.values()]):
            raise FloatingPointError("non-finite validation loss")
        by_source = {index.source_names[s]: v for s, v in means.items()}
        LOG.info("validation step=%d loss=%.4f ppl=%.2f protocol=%s evaluated_tokens=%d per_source=%s",
                 step, overall, math.exp(overall), fixed.metadata["fingerprint"], sum(counts.values()),
                 json.dumps(by_source, sort_keys=True))
        return {"step": step, "loss": overall, "per_category_loss": {}, "per_source_loss": by_source,
                "evaluation": fixed.metadata, "evaluated_tokens": sum(counts.values())}
    overall = evaluate(model, index, config, device)
    categories = {}
    category_ids = index.category_ids
    if max_categories is not None:
        totals = index.token_totals("val")
        category_ids = sorted(totals, key=lambda key: (-totals[key], key))[:max_categories]
    for category in category_ids:
        if np.any((index.splits == 1) & (index.categories == category)):
            categories[category] = evaluate(model, index, config, device, category)
    LOG.info("validation step=%d loss=%.4f ppl=%.2f per_category=%s", step, overall,
             math.exp(overall), " ".join(f"c{k}={math.exp(v):.1f}" for k, v in categories.items()))
    return {"step": step, "loss": overall, "per_category_loss": categories}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--index-cache", type=Path, help="default: a cache inside the run directory")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", help="exact MLX/CUDA checkpoint recovery, or latest")
    start.add_argument("--init-from", type=Path, help="initialize a NEW pretraining run from CUDA weights only")
    start.add_argument("--migrate-from", type=Path,
                       help="NEW run: retain full CUDA state; migrate unread document suffix to shuffled blocks")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="CPU float32 is for small validation runs")
    parser.add_argument("--steps", type=int, help="stop and save after this many updates (smoke test)")
    parser.add_argument("--evaluate-first", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--expected-val-loss", type=float)
    parser.add_argument("--val-tolerance", type=float, default=0.03)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        help="override tracking mode (default comes from config)")
    return parser.parse_args()


def run(args, tracker):
    rank, size = world()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if getattr(args, "device", "cuda") == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run in the host GPU environment")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = json.loads(args.config.read_text())
    model_config, config = load_run_config(args.config)
    if config.dtype != ("float16" if device.type == "cuda" else "float32"):
        raise ValueError("use float16 AMP on CUDA or float32 for CPU validation")
    if config.seq_len > model_config.max_position_embeddings or config.eval_batches < 1:
        raise ValueError("invalid sequence length or evaluation batches")
    pretraining = raw.get("pretraining")
    migrating = getattr(args, "migrate_from", None) is not None
    new_corpus = pretraining is not None
    continuation = raw.get("continuation")
    if new_corpus and continuation is not None:
        raise ValueError("choose pretraining or continuation, not both")
    if new_corpus:
        if pretraining.get("schedule") != "cosine" or pretraining.get("sampling") not in {"natural", "target", "document_stream", "shuffled_blocks"}:
            raise ValueError("unsupported pretraining schedule or sampling")
        if type(pretraining.get("eval_categories", 12)) is not int or pretraining.get("eval_categories", 12) < 0:
            raise ValueError("eval_categories must be a nonnegative integer")
    elif not args.resume or continuation is None:
        raise ValueError("legacy continuation requires --resume; use a pretraining config for a new run")
    elif continuation.get("schedule") not in {"cosine_from_imported_lr", "cosine_warm_restarts"}:
        raise ValueError("unsupported continuation schedule")
    if getattr(args, "init_from", None) and not new_corpus:
        raise ValueError("--init-from requires a pretraining configuration")
    block_mode = new_corpus and pretraining["sampling"] == "shuffled_blocks"
    attention = (pretraining or {}).get("attention", "causal")
    if attention not in ATTENTION_MODES:
        raise ValueError(f"pretraining.attention must be one of {ATTENTION_MODES}")
    document_attention = attention == "document"
    if document_attention and not block_mode:
        raise ValueError("document attention requires shuffled_blocks sampling (document ids)")
    if size > 1 and not block_mode:
        raise ValueError("multi-GPU training requires shuffled_blocks sampling")
    if migrating and (not block_mode or args.resume or getattr(args, "init_from", None)):
        raise ValueError("--migrate-from requires a new shuffled-block pretraining run")
    spec = SampleSpec(config.seq_len, tuple((pretraining or {}).get("prediction_offsets", [1])),
                      config.eod_token_id)
    if spec.prediction_offsets != (1,):
        raise ValueError("MTP data interface is available, but this trainer currently implements NTP only")
    target_tokens = int((pretraining if new_corpus else continuation)["target_tokens"])
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    step_tokens = config.seq_len * config.batch_size * config.grad_accum_steps * size
    checkpoint = (latest_checkpoint(args.run_dir) if args.resume == "latest" else Path(args.resume)) if args.resume else None
    if checkpoint is None and ((args.run_dir / "checkpoints").exists() or (args.run_dir / "resolved_config.json").exists()):
        raise ValueError("run directory already contains a run; use --resume latest or a new directory")
    if migrating:
        checkpoint = Path(args.migrate_from)
    source_meta = json.loads((checkpoint / "train_state.json").read_text()) if checkpoint else None
    if source_meta and not migrating and source_meta.get("distributed", {}).get("world_size", 1) != size:
        raise ValueError("resume changed world_size; the global batch and sampler would change")
    if source_meta:
        for name in ("seq_len", "batch_size", "grad_accum_steps", "eod_token_id", "beta1", "beta2", "adam_eps", "weight_decay", "grad_clip"):
            if source_meta["training"][name] != getattr(config, name):
                raise ValueError(f"continuation changed {name}; explicit migration is required")
    data_policy = raw.get("data", {})
    if new_corpus:
        from .prepare_corpus import validate_training_data
        if data_policy.get("category_mode") != "labels" or type(data_policy.get("before_year")) is not int:
            raise ValueError("new corpus requires category_mode=labels and an explicit before_year")
        validate_training_data(args.data_dir, before_year=data_policy["before_year"],
                               vocab_size=model_config.vocab_size,
                               tokenizer_sha256=data_policy.get("tokenizer_sha256"),
                               date_policy=data_policy.get("date_policy", "strict_upper"))
    # All ranks have validated the run directory; only now may rank 0 write.
    barrier()
    torch.manual_seed(config.seed)
    model = JishuiForCausalLM(model_config).to(device)
    if checkpoint is None:
        model.initialize_pretraining()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
        betas=(config.beta1, config.beta2), eps=config.adam_eps, weight_decay=config.weight_decay, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", init_scale=config.loss_scale, enabled=device.type == "cuda")
    gpu_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    LOG.info("device=%s torch=%s parameters=%d world_size=%d attention=%s", gpu_name, torch.__version__,
             sum(p.numel() for p in model.parameters()), size, attention)
    native = checkpoint is not None and (checkpoint / "training.pt").exists()
    if new_corpus and checkpoint and not native:
        raise ValueError("new-corpus resume requires its own PyTorch checkpoint")
    if checkpoint:
        state = (load_checkpoint(checkpoint, model, optimizer, scaler) if native
                 else import_mlx(checkpoint, model, optimizer))
    else:
        state = {"step": 0, "tokens_seen": 0, "origin": {"format": "random", "seed": config.seed}}
        if getattr(args, "init_from", None):
            parent = load_model_weights(args.init_from, model)
            state["origin"] = {"format": "weights_only", "checkpoint": str(args.init_from.resolve()),
                               "parent_step": parent["step"], "parent_tokens": parent["tokens_seen"]}
            state["tracking"] = parent.get("tracking")
    step, tokens_seen = int(state["step"]), int(state["tokens_seen"])
    final_step = step + math.ceil((target_tokens - tokens_seen) / step_tokens)
    if block_mode and native:
        # Partial epoch-tail blocks reduce counted tokens, but must not move the
        # saved cosine schedule on resume. Any final extra update uses the floor LR.
        final_step = state["schedule"]["end_step"]
    if target_tokens <= tokens_seen:
        raise ValueError("checkpoint already reached the target token budget")
    config = replace(config, max_steps=final_step, compile_step=False)
    index_path = args.index_cache or args.run_dir / "pretrain_index.npz"
    if rank == 0:
        index, rebuilt = load_or_build_index(args.data_dir, index_path,
                                            category_mode=data_policy.get("category_mode", "legacy"))
    barrier()
    if rank != 0:
        index, rebuilt = load_or_build_index(args.data_dir, index_path,
                                            category_mode=data_policy.get("category_mode", "legacy"))
        if rebuilt:
            raise RuntimeError("non-zero rank rebuilt the index; rank 0 cache is inconsistent")
    LOG.info("index=%s rebuilt=%s usable_train_tokens=%s", args.index_cache, rebuilt, index.token_totals("train"))
    if new_corpus:
        schedule = make_pretraining_schedule(config, final_step)
        if pretraining["sampling"] in {"document_stream", "shuffled_blocks"}:
            sampling = {"mode": pretraining["sampling"]}
            if block_mode:
                sampling["sample_spec"] = spec.identity()
        else:
            sampling = natural_sampling(index) if pretraining["sampling"] == "natural" else load_sampling_config(args.data_dir)
        if native:
            checks = [("schedule", schedule), ("data_policy", data_policy), ("index_metadata", index.metadata)]
            if migrating:
                if state.get("sampling") != {"mode": "document_stream"}:
                    raise ValueError("migration only accepts the old document_stream sampler")
                allowed = {"sampling", "prediction_offsets", "validation", "eval_categories"}
                if ({k: v for k, v in state["pretraining"].items() if k not in allowed} !=
                        {k: v for k, v in pretraining.items() if k not in allowed}):
                    raise ValueError("migration changed pretraining budget or schedule")
            else:
                checks += [("pretraining", pretraining), ("sampling", sampling)]
            for key, expected in checks:
                if state.get(key) != expected:
                    raise ValueError(f"resume changed {key}; use --init-from for a new training run")
            if source_meta["training"]["seed"] != config.seed or source_meta["training"]["dtype"] != config.dtype:
                raise ValueError("resume changed seed or dtype")
    elif native:
        if state["continuation"] != continuation or state["schedule"]["end_step"] != final_step:
            raise ValueError("resume must retain the checkpoint continuation and schedule")
        sampling, schedule = state["sampling"], state["schedule"]
        expected_schedule = make_continuation_schedule(continuation, config,
            schedule["start_step"], final_step, schedule["start_lr"])
        saved_schedule = dict(schedule)
        saved_schedule.setdefault("mode", "cosine_from_imported_lr")
        if expected_schedule != saved_schedule:
            raise ValueError("resume changed learning-rate settings; retain the saved schedule")
        # Reject data changes even if only file size/mtime changed; do not silently
        # resume RNG against a different document ordering or corpus.
        if state["index_metadata"] != index.metadata:
            raise ValueError("data index changed since CUDA checkpoint")
    else:
        if continuation["sampling"] == "natural":
            sampling = natural_sampling(index)
        elif continuation["sampling"] == "target":
            sampling = load_sampling_config(args.data_dir)
        else:
            raise ValueError("continuation sampling must be natural or target")
        schedule = make_continuation_schedule(continuation, config, step, final_step,
                                              state["imported_learning_rate"])
    if block_mode:
        # Every rank draws the global micro-batch; only rank 0 writes the cache.
        sampler = ShuffledBlockIterator(index, spec, config.batch_size*size, config.seed,
                                        args.run_dir / "sample_indices" if rank == 0 else None)
    elif sampling.get("mode") == "document_stream":
        sampler = ShuffledDocumentIterator(index, config.seq_len, config.batch_size, config.seed, config.eod_token_id)
    else:
        sampler = PackedBatchIterator(index, sampling, config.seq_len, config.batch_size,
                                      config.seed, mode="target", eod_token_id=config.eod_token_id)
    if checkpoint:
        if migrating:
            sampler.migrate_document_stream(state["sampler"])
            state["data_migration"] = {"checkpoint": str(checkpoint.resolve()), "step": step,
                "tokens_seen": tokens_seen, "from": "document_stream", "to": "shuffled_blocks",
                "first_epoch": sampler.first_epoch, "first_offset": sampler.first_offset,
                "remaining_epoch_targets": sampler.stream.size-sampler.first_offset,
                "preserved": ["model", "optimizer", "scaler", "rng", "schedule", "token_budget"]}
        else:
            sampler.load_state_dict(state["sampler"])
    fixed = None
    validation = (pretraining or {}).get("validation")
    if validation:
        if validation.get("mode") != "source_stratified_blocks" or set(validation) - {"mode", "seed"}:
            raise ValueError("unsupported validation configuration")
        fixed = FixedValidation(index, spec, config.eval_batches, config.batch_size,
                                validation.get("seed", config.seed))
        if native and not migrating and state.get("evaluation") != fixed.metadata:
            raise ValueError("resume changed validation sample protocol")
        if rank == 0:
            (args.run_dir / "validation_protocol.json").write_text(json.dumps(fixed.metadata, indent=2))
        LOG.info("validation protocol=%s sources=%s missing_training_sources=%s", fixed.metadata["fingerprint"],
                 list(fixed.metadata["sources"]), fixed.metadata["missing_training_sources"])
    LOG.info("resumed=%s step=%d tokens=%d final_step=%d target_tokens=%d sampling=%s schedule=%s",
             checkpoint, step, tokens_seen, final_step, target_tokens, sampling.get("target_mix", sampling), schedule)
    resolved = {"model": model_config.to_dict(), "training": config.to_dict(),
                "continuation": continuation, "sampling": sampling, "schedule": schedule,
                "origin": state.get("origin"), "index_metadata": index.metadata,
                "distributed": {"world_size": size, "global_micro_batch": config.batch_size*size,
                                "tokens_per_update": step_tokens}}
    if new_corpus:
        resolved.update(pretraining=pretraining, data_policy=data_policy)
    if fixed:
        resolved["evaluation"] = fixed.metadata
    if state.get("data_migration"):
        resolved["data_migration"] = state["data_migration"]
    if rank == 0:
        (args.run_dir / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2))
    tracker.start(dict(resolved, runtime={"torch": str(torch.__version__),
        "cuda": torch.version.cuda, "gpu": gpu_name,
        "parameters": sum(p.numel() for p in model.parameters())}), step, tokens_seen,
        parent=state.get("tracking"))
    if args.evaluate_first or args.validate_only or args.expected_val_loss is not None:
        report = validation_report(model, index, config, device, step,
                                   max_categories=pretraining.get("eval_categories", 12) if new_corpus else None,
                                   fixed=fixed, document_attention=document_attention)
        if rank == 0:
            (args.run_dir / "initial_validation.json").write_text(json.dumps(report, indent=2))
        tracker.validation(report, tokens_seen)
        if args.expected_val_loss is not None and abs(report["loss"] - args.expected_val_loss) > args.val_tolerance:
            raise RuntimeError(f"checkpoint validation mismatch: {report['loss']} vs {args.expected_val_loss}")
    if args.validate_only:
        return "validated"

    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        LOG.info("stop requested: finishing current update and saving full checkpoint")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    last_saved = step
    limit = step + args.steps if args.steps else (math.inf if block_mode else final_step)
    window = dict(state.get("log_window", {"loss_sum": 0.0, "tokens": 0, "steps": 0}))

    def save():
        if rank == 0:
            snapshot = dict(resolved, step=step, tokens_seen=tokens_seen, sampler=sampler.state_dict(),
                            tracking=tracker.identity, log_window=dict(window))
            saved = save_checkpoint(args.run_dir, snapshot, model, optimizer, scaler, config.max_checkpoints)
            LOG.info("saved checkpoint %s", saved)
            tracker.checkpoint(saved, step, tokens_seen)
        barrier()

    def any_rank(flag):
        if size == 1:
            return flag
        value = torch.tensor([int(flag)], device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return bool(value.item())

    # A new run must be independently resumable even before its first periodic
    # save. This also makes the migrated source an explicit, complete checkpoint.
    if not (args.run_dir / "checkpoints" / f"step_{step:08d}").exists():
        save()

    # DDP wraps after every load so checkpoints keep the plain module's keys.
    trainable = (DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
                 if size > 1 else model)
    model.train()
    start_time, start_tokens = time.monotonic(), tokens_seen
    while step < limit and tokens_seen < target_tokens and not stop:
        before_sampler = copy.deepcopy(sampler.state_dict())
        update_start = time.monotonic()
        for attempt in range(8):
            sampler.load_state_dict(before_sampler)
            optimizer.zero_grad(set_to_none=True)
            batches = [next(sampler) for _ in range(config.grad_accum_steps)]
            # Global token count is known on every rank (all draw the same batch).
            update_tokens = sum(int(batch.targets(1)[1].sum()) if isinstance(batch, TokenBatch)
                                else config.seq_len*config.batch_size for batch in batches)
            local = [rows(batch, rank, size) for batch in batches]
            counts = [int(batch.targets(1)[1].sum()) if isinstance(batch, TokenBatch)
                      else config.seq_len*config.batch_size for batch in local]
            total_loss = 0.0
            for i, (batch, count) in enumerate(zip(local, counts)):
                x, y, seg = tensor_batch(batch, device, document_attention)
                sync = nullcontext() if size == 1 or i == len(local)-1 else trainable.no_sync()
                with sync:
                    with autocast(device):
                        loss = causal_lm_loss(trainable, x, y, seg)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss before update {step + 1}")
                    total_loss += float(loss.detach()) * count
                    # DDP averages gradients over ranks; multiplying by size makes
                    # the result the token-weighted mean over the global batch.
                    scaler.scale(loss * (count * size / update_tokens)).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            lr = continuation_lr(schedule, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if not torch.isfinite(grad_norm):
                # unscale_ recorded the overflow. Retry identical samples with a
                # lower loss scale; do not advance optimizer, tokens, or schedule.
                scaler.update(new_scale=scaler.get_scale() / 2)
                LOG.warning("nonfinite gradient; retry=%d scale=%g", attempt + 1, scaler.get_scale())
                tracker.log({"optimizer/overflow_retry": attempt + 1,
                             "optimizer/loss_scale": scaler.get_scale()}, step, tokens_seen)
                continue
            scaler.step(optimizer)
            scaler.update()
            break
        else:
            raise FloatingPointError("gradient overflow persisted after 8 retries")
        if size > 1:
            reduced = torch.tensor([total_loss], dtype=torch.float64, device=device)
            dist.all_reduce(reduced)
            total_loss = float(reduced.item())
        step += 1
        tokens_seen += update_tokens
        window["loss_sum"] += total_loss
        window["tokens"] += update_tokens
        window["steps"] += 1
        if step % config.log_interval == 0:
            elapsed = time.monotonic() - update_start
            mean_loss = window["loss_sum"] / window["tokens"]
            average_rate = (tokens_seen - start_tokens) / (time.monotonic() - start_time)
            cycle = sum(step - 1 >= boundary for boundary in schedule.get("cycle_boundaries", [schedule["start_step"]])[1:-1]) + 1
            tracker.log({"train/loss": mean_loss, "train/perplexity": math.exp(min(mean_loss, 20)),
                "optimizer/learning_rate": lr, "optimizer/grad_norm": float(grad_norm),
                "optimizer/loss_scale": scaler.get_scale(), "optimizer/cycle": cycle,
                "perf/tokens_per_second": update_tokens / elapsed,
                "perf/average_tokens_per_second": average_rate,
                "perf/gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
                "perf/gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30 if device.type == "cuda" else 0,
                "perf/eta_hours": max(0, target_tokens - tokens_seen) / average_rate / 3600,
                "progress/target_fraction": tokens_seen / target_tokens}, step, tokens_seen)
            LOG.info("step=%d loss=%.4f ppl=%.2f lr=%.3e grad_norm=%.3f tokens=%d tok/s=%.0f "
                     "peak_vram_gib=%.2f scale=%g eta_hours=%.1f loss_steps=%d loss_tokens=%d", step, mean_loss,
                     math.exp(min(mean_loss, 20)), lr, float(grad_norm), tokens_seen,
                     update_tokens / elapsed, torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
                     scaler.get_scale(), max(0, target_tokens - tokens_seen) / average_rate / 3600,
                     window["steps"], window["tokens"])
            window = {"loss_sum": 0.0, "tokens": 0, "steps": 0}
        if config.eval_interval > 0 and step % config.eval_interval == 0:
            tracker.validation(validation_report(model, index, config, device, step,
                max_categories=pretraining.get("eval_categories", 12) if new_corpus else None, fixed=fixed,
                document_attention=document_attention), tokens_seen)
        if config.save_interval > 0 and step % config.save_interval == 0:
            save()
            last_saved = step
        # A signal may reach ranks at different updates; stop together.
        stop = any_rank(stop)
    if step != last_saved:
        save()
    LOG.info("%s step=%d tokens=%d", "target reached" if tokens_seen >= target_tokens else "stopped", step, tokens_seen)
    return "completed" if tokens_seen >= target_tokens else "stopped"


def init_distributed(device):
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size > 1 and not dist.is_initialized():
        if device == "cuda":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl" if device == "cuda" else "gloo")
    return world()


def main():
    args = parse_args()
    rank, size = init_distributed(args.device)
    try:
        if rank == 0:
            args.run_dir.mkdir(parents=True, exist_ok=True)
        barrier()
        with (args.run_dir / "trainer.lock" if rank == 0 else Path(os.devnull)).open("a") as lock:
            if rank == 0:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("another trainer is already using this run directory") from None
            handlers = ([logging.StreamHandler(), logging.FileHandler(args.run_dir / "trainlog.txt")]
                        if rank == 0 else [logging.StreamHandler()])
            logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                                format=f"%(asctime)s %(levelname)s rank{rank} %(name)s: %(message)s",
                                handlers=handlers, force=True)
            if rank == 0:
                (args.run_dir / "trainer.pid").write_text(str(os.getpid()))
            # Only rank 0 reports metrics; the others never create a W&B run.
            tracker = ExperimentTracker(args.run_dir, json.loads(args.config.read_text()).get("tracking"),
                                        mode=args.wandb_mode if rank == 0 else "disabled")
            status, exit_code = "failed", 1
            try:
                status = run(args, tracker)
                exit_code = 0
            except Exception:
                LOG.exception("training failed; resume from the latest complete checkpoint")
                raise
            finally:
                if rank == 0:
                    (args.run_dir / "trainer.pid").unlink(missing_ok=True)
                try:
                    tracker.finish(status, exit_code)
                except Exception:
                    LOG.exception("W&B finalization failed; local checkpoints and logs remain available")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
