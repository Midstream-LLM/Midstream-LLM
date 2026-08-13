"""Train Jishui with MLX as optimizer owner and ANE as a gradient worker.

The production mode resumes the ordinary MLX checkpoint format, draws the
same number of sampler microbatches per optimizer step, splits those records
between MLX and ANE, then performs one weighted gradient reduction and one MLX
AdamW update.  Checkpoints remain readable by the original MLX trainer.

Passing ``--records`` without ``--resume`` keeps a bounded fixed-record mode
for protocol and performance gates.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from .ane_profiles import resolve_ane_backward_scales
from .config import load_run_config
from .checkpoints import latest_checkpoint
from .data import PackedBatchIterator, load_or_build_index, load_sampling_config
from .hybrid_protocol import (
    HYBRID_STATUS_GRAD_READY,
    HYBRID_STATUS_READY,
    HYBRID_STATUS_STOPPED,
    initialize_control_file,
    initialize_tensor_file,
    publish_generation,
    publish_stop,
    read_parameter_tree,
    wait_for_status,
    write_parameter_tree,
)
from .model import JishuiForCausalLM
from .train_mlx import (
    DTYPES,
    evaluate,
    load_checkpoint,
    make_loss_and_grad,
    make_optimizer,
    optimizer_gradients,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/jishui-200m-mlx.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/jishui-200m-hybrid"))
    parser.add_argument("--index-cache", type=Path)
    parser.add_argument("--resume", help="MLX checkpoint directory or 'latest'")
    parser.add_argument(
        "--records",
        type=Path,
        help="fixed uint16 records for a bounded gate instead of the real sampler",
    )
    parser.add_argument(
        "--native-binary", type=Path, default=Path("native/ane/training_dynamic/train")
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/private/tmp/jishui-hybrid"))
    parser.add_argument(
        "--generations",
        type=int,
        help="stop after this many optimizer updates (default: 3 fixed, config max real)",
    )
    parser.add_argument("--max-steps", type=int, help="override the absolute training target")
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--mlx-microsteps", type=int)
    parser.add_argument("--ane-microsteps", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--metal-norm", action="store_true")
    parser.add_argument("--metal-silu", action="store_true")
    parser.add_argument("--metal-io", action="store_true")
    parser.add_argument("--npu-fuse-qkv-bwd", action="store_true")
    parser.add_argument(
        "--legacy-sdpa-bwd",
        action="store_true",
        help="use the pre-optimization native SDPA backward I/O path",
    )
    parser.add_argument(
        "--ffn-bwd-scale",
        type=float,
        help=(
            "override dffn scaling before native W2 backward "
            "(Jishui-200M/2048 default: 128)"
        ),
    )
    parser.add_argument(
        "--ffn-w13-bwd-scale",
        type=float,
        help=(
            "override dh1/dh3 scaling before native W1/W3 backward "
            "(Jishui-200M/2048 default: 32)"
        ),
    )
    parser.add_argument(
        "--sdpa-bwd-scale",
        type=float,
        help=(
            "override da scaling through native SDPA backward "
            "(Jishui-200M/2048 default: 256)"
        ),
    )
    parser.add_argument(
        "--qkv-bwd-scale",
        type=float,
        help=(
            "override dq/dk/dv scaling through native projection backward "
            "(Jishui-200M/2048 default: 128)"
        ),
    )
    parser.add_argument(
        "--wo-bwd-scale",
        type=float,
        help=(
            "override dx2 scaling through native Wo backward "
            "(Jishui-200M/2048 default: 128)"
        ),
    )
    parser.add_argument(
        "--native-trace-dir",
        type=Path,
        help="write opt-in native fp32 parity diagnostics",
    )
    parser.add_argument(
        "--gradient-shadow",
        action="store_true",
        help="recompute ANE records on MLX and report full-gradient agreement",
    )
    parser.add_argument(
        "--gradient-cosine-min",
        type=float,
        default=0.999,
        help="minimum full-gradient cosine accepted by --gradient-shadow",
    )
    parser.add_argument(
        "--gradient-relative-l2-max",
        type=float,
        default=0.05,
        help="maximum relative L2 accepted by --gradient-shadow",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="do not execute one MLX gradient before starting the native worker",
    )
    parser.add_argument(
        "--no-save-final",
        action="store_true",
        help="do not save a final checkpoint (intended only for bounded gates)",
    )
    return parser.parse_args()


def _record_array(path: Path, seq_len: int, vocab_size: int) -> np.memmap:
    width = seq_len + 1
    size = path.stat().st_size
    if size == 0 or size % (width * np.dtype("<u2").itemsize):
        raise ValueError(f"records file is not a multiple of {width} uint16 values: {path}")
    records = np.memmap(path, dtype="<u2", mode="r", shape=(size // (width * 2), width))
    if np.any(records >= vocab_size):
        raise ValueError("records contain a token id outside the model vocabulary")
    return records


def _weighted_gradients(
    mlx_gradients: object,
    native_gradients: object,
    mlx_microsteps: int,
    native_microsteps: int,
) -> object:
    total = mlx_microsteps + native_microsteps
    return tree_map(
        lambda mlx_value, native_value: (
            mlx_value.astype(mx.float32) * mlx_microsteps
            + native_value.astype(mx.float32) * native_microsteps
        )
        / total,
        mlx_gradients,
        native_gradients,
    )


def _mlx_average_gradient(loss_and_grad, records: np.ndarray) -> tuple[float, object]:
    accumulated = None
    total_loss = 0.0
    for record in records:
        values = np.asarray(record, dtype=np.int32)
        inputs = mx.array(values[:-1][None, :])
        targets = mx.array(values[1:][None, :])
        loss, gradients = loss_and_grad(inputs, targets)
        mx.eval(loss, gradients)
        total_loss += float(loss.item())
        if accumulated is None:
            accumulated = gradients
        else:
            accumulated = tree_map(
                lambda current, new: current + new,
                accumulated,
                gradients,
            )
            mx.eval(accumulated)
    if accumulated is None:
        raise ValueError("MLX must own at least one microstep")
    averaged = tree_map(lambda value: value / len(records), accumulated)
    return total_loss / len(records), averaged


def _fixed_step_records(
    records: np.ndarray, update: int, total_microsteps: int
) -> np.ndarray:
    indexes = (update * total_microsteps + np.arange(total_microsteps)) % len(records)
    return np.asarray(records[indexes], dtype=np.int32)


def _sample_step_records(
    sampler: PackedBatchIterator, total_microsteps: int
) -> np.ndarray:
    return np.concatenate([next(sampler) for _ in range(total_microsteps)], axis=0)


def _gradient_group(name: str) -> str:
    if name == "embed_tokens.weight":
        return "tied_embedding"
    if name.startswith("norm."):
        return "final_norm"
    if ".input_norm." in name or ".post_attention_norm." in name:
        return "layer_norms"
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if f".attention.{projection}." in name:
            return f"attention_{projection}"
    for projection in ("gate_proj", "up_proj", "down_proj"):
        if f".mlp.{projection}." in name:
            return f"mlp_{projection}"
    return "other"


def _finalize_gradient_metrics(values: dict[str, float]) -> dict[str, float]:
    denominator = math.sqrt(values["reference_sq"] * values["candidate_sq"])
    return {
        "cosine": values["dot"] / max(denominator, 1.0e-30),
        "relative_l2": math.sqrt(values["difference_sq"]) / max(
            math.sqrt(values["reference_sq"]), 1.0e-30
        ),
        "reference_norm": math.sqrt(values["reference_sq"]),
        "candidate_norm": math.sqrt(values["candidate_sq"]),
        "difference_norm": math.sqrt(values["difference_sq"]),
    }


def _compare_gradients(reference: object, candidate: object) -> dict[str, object]:
    reference_flat = tree_flatten(reference)
    candidate_flat = tree_flatten(candidate)
    if [name for name, _ in reference_flat] != [name for name, _ in candidate_flat]:
        raise ValueError("gradient shadow trees have different leaves")

    dot = 0.0
    reference_sq = 0.0
    candidate_sq = 0.0
    difference_sq = 0.0
    maximum = 0.0
    worst_name = ""
    worst_relative_l2 = 0.0
    grouped: dict[str, dict[str, float]] = {}
    layered: dict[str, dict[str, float]] = {}
    tensors: list[dict[str, float | str]] = []
    for (name, reference_value), (_, candidate_value) in zip(
        reference_flat, candidate_flat, strict=True
    ):
        reference_f32 = reference_value.astype(mx.float32)
        candidate_f32 = candidate_value.astype(mx.float32)
        difference = candidate_f32 - reference_f32
        probes = (
            mx.sum(reference_f32 * candidate_f32),
            mx.sum(reference_f32 * reference_f32),
            mx.sum(candidate_f32 * candidate_f32),
            mx.sum(difference * difference),
            mx.max(mx.abs(difference)),
        )
        mx.eval(*probes)
        tensor_reference_sq = float(probes[1].item())
        tensor_candidate_sq = float(probes[2].item())
        tensor_difference_sq = float(probes[3].item())
        tensor_dot = float(probes[0].item())
        tensor_relative_l2 = math.sqrt(tensor_difference_sq) / max(
            math.sqrt(tensor_reference_sq), 1.0e-12
        )
        if tensor_relative_l2 > worst_relative_l2:
            worst_relative_l2 = tensor_relative_l2
            worst_name = name
        tensor_denominator = math.sqrt(tensor_reference_sq * tensor_candidate_sq)
        tensors.append(
            {
                "name": name,
                "cosine": tensor_dot / max(tensor_denominator, 1.0e-30),
                "relative_l2": tensor_relative_l2,
                "difference_norm": math.sqrt(tensor_difference_sq),
                "reference_norm": math.sqrt(tensor_reference_sq),
            }
        )
        for collection, group in (
            (grouped, _gradient_group(name)),
            (
                layered,
                name.split(".", 2)[1]
                if name.startswith("layers.")
                else name.split(".", 1)[0],
            ),
        ):
            accumulator = collection.setdefault(
                group,
                {
                    "dot": 0.0,
                    "reference_sq": 0.0,
                    "candidate_sq": 0.0,
                    "difference_sq": 0.0,
                },
            )
            accumulator["dot"] += tensor_dot
            accumulator["reference_sq"] += tensor_reference_sq
            accumulator["candidate_sq"] += tensor_candidate_sq
            accumulator["difference_sq"] += tensor_difference_sq
        dot += tensor_dot
        reference_sq += tensor_reference_sq
        candidate_sq += tensor_candidate_sq
        difference_sq += tensor_difference_sq
        maximum = max(maximum, float(probes[4].item()))

    denominator = math.sqrt(reference_sq * candidate_sq)
    group_metrics = {
        name: _finalize_gradient_metrics(values)
        for name, values in sorted(grouped.items())
    }
    layer_metrics = [
        {"name": name, **_finalize_gradient_metrics(values)}
        for name, values in layered.items()
    ]
    layer_metrics.sort(key=lambda value: float(value["difference_norm"]), reverse=True)
    tensors.sort(key=lambda value: float(value["difference_norm"]), reverse=True)
    return {
        "cosine": dot / max(denominator, 1.0e-30),
        "relative_l2": math.sqrt(difference_sq) / max(
            math.sqrt(reference_sq), 1.0e-30
        ),
        "max_abs": maximum,
        "reference_norm": math.sqrt(reference_sq),
        "candidate_norm": math.sqrt(candidate_sq),
        "worst_tensor": worst_name,
        "worst_tensor_relative_l2": worst_relative_l2,
        "groups": group_metrics,
        "largest_difference_tensors": tensors[:10],
        "largest_difference_layers": layer_metrics[:10],
    }


def run(args: argparse.Namespace) -> dict:
    model_config, train_config = load_run_config(args.config)
    backward_scales = resolve_ane_backward_scales(
        model_config,
        {
            "ffn_bwd_scale": args.ffn_bwd_scale,
            "ffn_w13_bwd_scale": args.ffn_w13_bwd_scale,
            "wo_bwd_scale": args.wo_bwd_scale,
            "sdpa_bwd_scale": args.sdpa_bwd_scale,
            "qkv_bwd_scale": args.qkv_bwd_scale,
        },
    )
    overrides = {}
    for name in ("max_steps", "save_interval", "eval_interval"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if overrides:
        train_config = replace(train_config, **overrides)
    if train_config.batch_size != 1 or train_config.seq_len != model_config.max_position_embeddings:
        raise ValueError("hybrid training currently requires batch_size=1 and full model sequence")
    if args.ane_microsteps < 1:
        raise ValueError("--ane-microsteps must be positive")
    if not -1.0 <= args.gradient_cosine_min <= 1.0:
        raise ValueError("--gradient-cosine-min must be between -1 and 1")
    if not math.isfinite(args.gradient_relative_l2_max) or (
        args.gradient_relative_l2_max < 0.0
    ):
        raise ValueError("--gradient-relative-l2-max must be finite and non-negative")
    fixed_mode = args.records is not None
    if fixed_mode and args.resume:
        raise ValueError("--records fixed gate and --resume real training are mutually exclusive")
    if not fixed_mode and not args.resume:
        raise ValueError("real sampler mode requires --resume CHECKPOINT or --resume latest")

    if fixed_mode:
        fixed_records = _record_array(
            args.records, train_config.seq_len, model_config.vocab_size
        )
        mlx_microsteps = args.mlx_microsteps or 1
        updates_to_run = args.generations or 3
    else:
        fixed_records = None
        mlx_microsteps = (
            args.mlx_microsteps
            if args.mlx_microsteps is not None
            else train_config.grad_accum_steps - args.ane_microsteps
        )
        updates_to_run = args.generations
    if mlx_microsteps < 1:
        raise ValueError("--mlx-microsteps must be positive")
    total_microsteps = mlx_microsteps + args.ane_microsteps
    if not fixed_mode and total_microsteps != train_config.grad_accum_steps:
        raise ValueError(
            "real training must preserve grad_accum_steps: "
            f"mlx={mlx_microsteps} ane={args.ane_microsteps} "
            f"total={total_microsteps} configured={train_config.grad_accum_steps}"
        )
    if updates_to_run is not None and updates_to_run < 1:
        raise ValueError("--generations must be positive")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.work_dir / "weights.fp16"
    gradients_path = args.work_dir / "gradients.fp16"
    control_path = args.work_dir / "control.bin"
    records_path = args.work_dir / "records.uint16"
    parity_path = args.work_dir / "native.parity.jsonl"

    sampler: PackedBatchIterator | None = None
    index = None
    if not fixed_mode:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        index_cache = args.index_cache or args.run_dir / "cache" / "pretrain_index.npz"
        index, rebuilt = load_or_build_index(args.data_dir, index_cache)
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
        print(
            "hybrid sampler index=%s status=%s"
            % (index_cache, "rebuilt" if rebuilt else "cached"),
            flush=True,
        )

    mx.random.seed(train_config.seed)
    model = JishuiForCausalLM(model_config)
    model.set_dtype(DTYPES[train_config.dtype])
    mx.eval(model.parameters())
    optimizer = make_optimizer(train_config)
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)

    start_step = 0
    tokens_seen = 0
    resumed_checkpoint: Path | None = None
    if not fixed_mode:
        assert sampler is not None
        resumed_checkpoint = (
            latest_checkpoint(args.run_dir)
            if args.resume == "latest"
            else Path(args.resume)
        )
        start_step, tokens_seen = load_checkpoint(
            resumed_checkpoint, model, optimizer, sampler, model_config
        )
        print(
            "hybrid resumed checkpoint=%s step=%d tokens=%d"
            % (resumed_checkpoint, start_step, tokens_seen),
            flush=True,
        )
    model.train()

    configured_target = train_config.max_steps
    target_step = (
        start_step + updates_to_run
        if updates_to_run is not None
        else configured_target
    )
    if target_step > configured_target:
        raise ValueError(
            f"requested target step {target_step} exceeds configured max_steps "
            f"{configured_target}"
        )
    if target_step <= start_step:
        raise ValueError(
            f"target step {target_step} must be greater than checkpoint step {start_step}"
        )

    loss_and_grad = make_loss_and_grad(model, train_config)
    if args.no_compile and train_config.compile_step:
        uncompiled_config = replace(train_config, compile_step=False)
        loss_and_grad = make_loss_and_grad(model, uncompiled_config)

    def next_step_records(update: int) -> np.ndarray:
        if fixed_records is not None:
            return _fixed_step_records(fixed_records, update, total_microsteps)
        assert sampler is not None
        return _sample_step_records(sampler, total_microsteps)

    pending_records = next_step_records(0)
    if np.any(pending_records < 0) or np.any(pending_records >= model_config.vocab_size):
        raise ValueError("sampler produced a token id outside the model vocabulary")

    # Compile and exercise the MLX graph before ANE is launched.  Apart from
    # removing first-use compiler noise from the overlap measurement, this
    # avoids making two independent Metal clients initialize at the same time.
    warmup_record = pending_records[args.ane_microsteps]
    warmup_inputs = mx.array(warmup_record[:-1][None, :])
    warmup_targets = mx.array(warmup_record[1:][None, :])
    if not args.no_warmup:
        warmup_started = time.monotonic()
        warmup_loss, warmup_grads = loss_and_grad(warmup_inputs, warmup_targets)
        mx.eval(warmup_loss, warmup_grads)
        print(
            "hybrid MLX warmup loss=%.6f seconds=%.3f"
            % (float(warmup_loss.item()), time.monotonic() - warmup_started),
            flush=True,
        )
        del warmup_loss, warmup_grads

    record_map = np.memmap(
        records_path,
        dtype="<u2",
        mode="w+",
        shape=(args.ane_microsteps, train_config.seq_len + 1),
    )
    record_map[:] = pending_records[: args.ane_microsteps]
    record_map.flush()

    native_args = [
        str(args.native_binary.resolve()),
        "--scratch",
        "--records",
        "--data",
        str(records_path.resolve()),
        "--steps",
        "1",
        "--accum",
        "1",
        "--log-interval",
        str(args.ane_microsteps),
        "--no-checkpoint",
        "--hybrid-weights",
        str(weights_path),
        "--hybrid-gradients",
        str(gradients_path),
        "--hybrid-control",
        str(control_path),
        "--hybrid-timeout",
        str(max(1, int(args.timeout))),
        "--parity-report",
        str(parity_path),
        "--loss-scale",
        str(train_config.loss_scale),
    ]
    for enabled, flag in (
        (args.metal_norm, "--metal-norm"),
        (args.metal_silu, "--metal-silu"),
        (args.metal_io, "--metal-io"),
        (args.npu_fuse_qkv_bwd, "--npu-fuse-qkv-bwd"),
        (args.legacy_sdpa_bwd, "--legacy-sdpa-bwd"),
    ):
        if enabled:
            native_args.append(flag)
    if args.native_trace_dir is not None:
        native_args.extend(["--trace-dir", str(args.native_trace_dir.resolve())])
    if backward_scales["ffn_bwd_scale"] != 1.0:
        native_args.extend(
            ["--ffn-bwd-scale", str(backward_scales["ffn_bwd_scale"])]
        )
    if backward_scales["ffn_w13_bwd_scale"] != 1.0:
        native_args.extend(
            [
                "--ffn-w13-bwd-scale",
                str(backward_scales["ffn_w13_bwd_scale"]),
            ]
        )
    if backward_scales["sdpa_bwd_scale"] != 1.0:
        native_args.extend(
            ["--sdpa-bwd-scale", str(backward_scales["sdpa_bwd_scale"])]
        )
    if backward_scales["qkv_bwd_scale"] != 1.0:
        native_args.extend(
            ["--qkv-bwd-scale", str(backward_scales["qkv_bwd_scale"])]
        )
    if backward_scales["wo_bwd_scale"] != 1.0:
        native_args.extend(
            ["--wo-bwd-scale", str(backward_scales["wo_bwd_scale"])]
        )

    process: subprocess.Popen | None = None
    started = time.monotonic()
    generation_results: list[dict[str, float | int]] = []
    last_loss = math.nan
    last_grad_norm = math.nan
    last_gradient_shadow: dict[str, float | str] | None = None
    last_checkpoint_step = start_step
    with weights_path.open("w+b") as weight_handle, gradients_path.open(
        "w+b"
    ) as gradient_handle, control_path.open("w+b") as control_handle:
        weight_map = initialize_tensor_file(weight_handle, model_config, 1)
        gradient_map = initialize_tensor_file(gradient_handle, model_config, 2)
        control_map = initialize_control_file(control_handle)
        try:
            write_parameter_tree(weight_map, model.parameters(), model_config)
            weight_map.flush()
            process = subprocess.Popen(native_args, cwd=args.native_binary.parent)
            wait_for_status(control_map, HYBRID_STATUS_READY, args.timeout)
            print("hybrid native status=READY", flush=True)

            for update, step in enumerate(range(start_step + 1, target_step + 1)):
                generation = update + 1
                if update:
                    pending_records = next_step_records(update)
                    if np.any(pending_records < 0) or np.any(
                        pending_records >= model_config.vocab_size
                    ):
                        raise ValueError(
                            "sampler produced a token id outside the model vocabulary"
                        )
                record_map[:] = pending_records[: args.ane_microsteps]
                record_map.flush()
                publish_generation(
                    control_map,
                    generation,
                    record_base=0,
                    microsteps=args.ane_microsteps,
                )
                print(
                    "hybrid step=%d generation=%d published mlx=%d ane=%d"
                    % (step, generation, mlx_microsteps, args.ane_microsteps),
                    flush=True,
                )

                # Start the MLX GPU work immediately after publishing the
                # snapshot so it overlaps the native ANE worker.
                t0 = time.monotonic()
                mlx_loss, mlx_grads = _mlx_average_gradient(
                    loss_and_grad,
                    pending_records[args.ane_microsteps :],
                )
                mlx_seconds = time.monotonic() - t0
                print(
                    "hybrid step=%d mlx-gradient seconds=%.3f"
                    % (step, mlx_seconds),
                    flush=True,
                )
                native_state = wait_for_status(
                    control_map, HYBRID_STATUS_GRAD_READY, args.timeout
                )
                if native_state.generation != generation:
                    raise RuntimeError(
                        "native returned the wrong generation: "
                        f"expected={generation} got={native_state.generation}"
                    )
                print(
                    "hybrid step=%d native-gradient-ready seconds=%.3f"
                    % (step, time.monotonic() - t0),
                    flush=True,
                )
                native_grads = read_parameter_tree(
                    gradient_map, mlx_grads, model_config
                )
                mx.eval(native_grads)
                if args.gradient_shadow:
                    shadow_loss, shadow_grads = _mlx_average_gradient(
                        loss_and_grad,
                        pending_records[: args.ane_microsteps],
                    )
                    last_gradient_shadow = _compare_gradients(
                        shadow_grads, native_grads
                    )
                    last_gradient_shadow["mlx_loss"] = shadow_loss
                    print(
                        "hybrid gradient-shadow "
                        + json.dumps(last_gradient_shadow, sort_keys=True),
                        flush=True,
                    )
                    if (
                        last_gradient_shadow["cosine"] < args.gradient_cosine_min
                        or last_gradient_shadow["relative_l2"]
                        > args.gradient_relative_l2_max
                    ):
                        raise RuntimeError(
                            "native gradient shadow failed: cosine=%.9f (min %.9f) "
                            "relative_l2=%.9f (max %.9f)"
                            % (
                                last_gradient_shadow["cosine"],
                                args.gradient_cosine_min,
                                last_gradient_shadow["relative_l2"],
                                args.gradient_relative_l2_max,
                            )
                        )
                    del shadow_grads
                combined = _weighted_gradients(
                    mlx_grads,
                    native_grads,
                    mlx_microsteps,
                    args.ane_microsteps,
                )
                combined, grad_norm = optim.clip_grad_norm(
                    combined, train_config.grad_clip
                )
                optimizer.update(model, optimizer_gradients(model, combined))
                mx.eval(model.parameters(), optimizer.state, grad_norm)
                write_parameter_tree(weight_map, model.parameters(), model_config)
                weight_map.flush()
                elapsed = time.monotonic() - t0
                tokens_seen += total_microsteps * train_config.seq_len
                last_loss = mlx_loss
                last_grad_norm = float(grad_norm.item())
                generation_results.append(
                    {
                        "generation": generation,
                        "step": step,
                        "mlx_loss": mlx_loss,
                        "native_generation": native_state.generation,
                        "combined_grad_norm": last_grad_norm,
                        "seconds": elapsed,
                    }
                )
                learning_rate = float(optimizer.learning_rate.item())
                print(
                    "hybrid step=%d loss=%.6f ppl=%.2f lr=%.3e "
                    "grad_norm=%.4f tokens=%d tok/s=%.1f seconds=%.3f"
                    % (
                        step,
                        mlx_loss,
                        math.exp(min(mlx_loss, 20.0)),
                        learning_rate,
                        last_grad_norm,
                        tokens_seen,
                        total_microsteps * train_config.seq_len / elapsed,
                        elapsed,
                    ),
                    flush=True,
                )

                if not fixed_mode and train_config.eval_interval > 0 and (
                    step % train_config.eval_interval == 0
                ):
                    assert index is not None
                    validation_loss = evaluate(model, index, train_config)
                    print(
                        "hybrid step=%d val_loss=%.6f val_ppl=%.2f"
                        % (
                            step,
                            validation_loss,
                            math.exp(min(validation_loss, 20.0)),
                        ),
                        flush=True,
                    )

                if (
                    not fixed_mode
                    and train_config.save_interval > 0
                    and step % train_config.save_interval == 0
                ):
                    assert sampler is not None
                    checkpoint = save_checkpoint(
                        args.run_dir,
                        step,
                        tokens_seen,
                        model,
                        optimizer,
                        sampler,
                        model_config,
                        train_config,
                    )
                    last_checkpoint_step = step
                    print(f"hybrid saved checkpoint={checkpoint}", flush=True)

            if (
                not fixed_mode
                and not args.no_save_final
                and last_checkpoint_step != target_step
            ):
                assert sampler is not None
                checkpoint = save_checkpoint(
                    args.run_dir,
                    target_step,
                    tokens_seen,
                    model,
                    optimizer,
                    sampler,
                    model_config,
                    train_config,
                )
                last_checkpoint_step = target_step
                print(f"hybrid saved final checkpoint={checkpoint}", flush=True)

            publish_stop(control_map)
            wait_for_status(control_map, HYBRID_STATUS_STOPPED, args.timeout)
            return {
                "mode": "fixed" if fixed_mode else "resume",
                "resumed_checkpoint": (
                    str(resumed_checkpoint) if resumed_checkpoint is not None else None
                ),
                "start_step": start_step,
                "end_step": target_step,
                "tokens_seen": tokens_seen,
                "mlx_microsteps": mlx_microsteps,
                "ane_microsteps": args.ane_microsteps,
                "loss_scale": train_config.loss_scale,
                "ane_backward_scales": backward_scales,
                "last_mlx_loss": last_loss,
                "last_grad_norm": last_grad_norm,
                "last_gradient_shadow": last_gradient_shadow,
                "last_checkpoint_step": last_checkpoint_step,
                "generations": generation_results,
                "wall_seconds": time.monotonic() - started,
                "weights": str(weights_path),
                "gradients": str(gradients_path),
            }
        finally:
            if process is not None:
                # A coordinator exception must wake the native wait loop before
                # waiting for the child.  Otherwise a failed MLX step can leave
                # the worker blocked until its full protocol timeout expires.
                if process.poll() is None:
                    try:
                        publish_stop(control_map)
                    except (OSError, ValueError):
                        pass
                try:
                    process.wait(timeout=min(max(1.0, args.timeout), 15.0))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            control_map.close()
            gradient_map.close()
            weight_map.close()
            record_map.flush()
            record_map._mmap.close()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
