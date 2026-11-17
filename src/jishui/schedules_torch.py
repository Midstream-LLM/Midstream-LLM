"""Learning rates indexed by completed optimizer updates, including restarts."""
from __future__ import annotations

import math


def make_pretraining_schedule(config, end_step):
    if (config.warmup_steps < 0 or config.warmup_steps >= end_step - 1
            or not 0 < config.min_learning_rate <= config.learning_rate):
        raise ValueError("pretraining needs positive learning rates and at least two decay updates")
    return {"mode": "cosine", "start_step": 0, "end_step": end_step,
            "warmup_steps": config.warmup_steps, "peak_lr": config.learning_rate,
            "end_lr": config.min_learning_rate}


def make_continuation_schedule(continuation, config, start_step, end_step, imported_lr):
    mode = continuation["schedule"]
    common = {"mode": mode, "start_step": start_step, "end_step": end_step,
              "start_lr": imported_lr, "end_lr": config.min_learning_rate}
    if mode == "cosine_from_imported_lr":
        return common
    if mode != "cosine_warm_restarts":
        raise ValueError("unsupported continuation schedule")
    cycles = continuation.get("cycles", 3)
    if type(cycles) is not int or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    updates = end_step - start_step
    width, extra = divmod(updates, cycles)
    if width < 2 or config.warmup_steps < 0 or config.warmup_steps == 1:
        raise ValueError("cycles need at least two updates; warmup must be zero or >=2")
    if (not all(math.isfinite(v) for v in (imported_lr, config.learning_rate, config.min_learning_rate))
            or not 0 < config.min_learning_rate <= imported_lr <= config.learning_rate):
        raise ValueError("restart schedule requires 0 < min_lr <= imported_lr <= peak_lr")
    boundaries = [start_step]
    for i in range(cycles):
        boundaries.append(boundaries[-1] + width + int(i < extra))
    if config.warmup_steps > boundaries[1] - start_step - 2:
        raise ValueError("first cycle must leave at least two cosine decay updates after warmup")
    return dict(common, peak_lr=config.learning_rate, warmup_steps=config.warmup_steps,
                cycle_boundaries=boundaries)


def continuation_lr(schedule: dict, completed_step: int) -> float:
    """LR for the NEXT update. No scheduler object is reset when resuming.

    For restarts, each cycle's first/last update uses exactly peak/min LR.
    Only the first cycle has a linear transition from the imported LR. Later
    cycles reset LR to the peak, retaining all model and optimizer state.
    """
    if schedule.get("mode") == "cosine":
        warmup = schedule["warmup_steps"]
        if completed_step < warmup:
            return schedule["peak_lr"] * (completed_step + 1) / warmup
        progress = min(1.0, max(0.0, (completed_step - warmup) /
                               (schedule["end_step"] - warmup - 1)))
        return schedule["end_lr"] + 0.5 * (schedule["peak_lr"] - schedule["end_lr"]) * (1 + math.cos(math.pi * progress))
    if schedule.get("mode", "cosine_from_imported_lr") == "cosine_from_imported_lr":
        progress = min(1.0, max(0.0, (completed_step - schedule["start_step"]) /
                               max(1, schedule["end_step"] - schedule["start_step"])))
        return schedule["end_lr"] + 0.5 * (schedule["start_lr"] - schedule["end_lr"]) * (1 + math.cos(math.pi * progress))
    if schedule["mode"] != "cosine_warm_restarts":
        raise ValueError("unsupported continuation schedule")
    if completed_step >= schedule["end_step"]:
        return schedule["end_lr"]
    completed_step = max(schedule["start_step"], completed_step)
    boundaries = schedule["cycle_boundaries"]
    for cycle, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        if begin <= completed_step < end:
            offset = completed_step - begin
            warmup = schedule["warmup_steps"] if cycle == 0 else 0
            if offset < warmup:
                return schedule["start_lr"] + (schedule["peak_lr"] - schedule["start_lr"]) * offset / (warmup - 1)
            progress = (offset - warmup) / (end - begin - warmup - 1)
            return schedule["end_lr"] + 0.5 * (schedule["peak_lr"] - schedule["end_lr"]) * (1 + math.cos(math.pi * progress))
    raise ValueError("invalid cycle boundaries")
