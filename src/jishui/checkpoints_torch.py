"""Strict MLX import and atomic, complete PyTorch continuation checkpoints."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .checkpoints import prune_checkpoints


def check_model_config(saved: dict, requested: dict) -> None:
    # Checkpointing changes memory use, not the model or parameter layout.
    saved, requested = dict(saved), dict(requested)
    saved.pop("gradient_checkpointing", None)
    requested.pop("gradient_checkpointing", None)
    if saved != requested:
        raise ValueError("checkpoint architecture differs from requested model")


def import_mlx(path: Path, model, optimizer) -> dict:
    state = json.loads((path / "train_state.json").read_text())
    check_model_config(state["model"], model.config.to_dict())
    parameters = dict(model.named_parameters())
    with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as weights:
        if set(weights.keys()) != set(parameters):
            raise ValueError("MLX parameter names do not match the PyTorch model")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = weights.get_tensor(name)
                if value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError(f"invalid MLX weight: {name}")
                parameter.copy_(value)
    with np.load(path / "optimizer.npz", allow_pickle=False) as values:
        expected = {f"{name}.{moment}" for name in parameters for moment in ("m", "v")}
        if set(values.files) != expected | {"step", "learning_rate"}:
            raise ValueError("incomplete or unknown MLX Adam state")
        step = int(values["step"])
        if step != state["step"]:
            raise ValueError("MLX optimizer and training step disagree")
        lr = float(values["learning_rate"])
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("invalid saved learning rate")
        for name, parameter in parameters.items():
            moments = {}
            for old, new in (("m", "exp_avg"), ("v", "exp_avg_sq")):
                value = values[f"{name}.{old}"]
                if (value.shape != tuple(parameter.shape) or not np.isfinite(value).all()
                        or (old == "v" and np.any(value < 0))):
                    raise ValueError(f"invalid MLX moment: {name}.{old}")
                moments[new] = torch.tensor(value, device=parameter.device, dtype=torch.float32)
            moments["step"] = torch.tensor(float(step), device=(parameter.device
                if optimizer.defaults.get("fused") else "cpu"))
            optimizer.state[parameter] = moments
    for group in optimizer.param_groups:
        group["lr"] = lr
    state["imported_learning_rate"] = lr
    state["origin"] = {
        "format": "mlx", "checkpoint": str(path.resolve()), "step": step,
        "precision_change": "fp16 weights/Adam -> fp32 master weights/Adam with fp16 CUDA AMP",
    }
    return state


def save_checkpoint(run_dir: Path, state: dict, model, optimizer, scaler, keep: int = 3) -> Path:
    root = run_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step_{state['step']:08d}"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = root / f".step_{state['step']:08d}.{os.getpid()}.tmp"
    temporary.mkdir(exist_ok=False)
    bundle = {
        "format_version": 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "state": state,
    }
    with (temporary / "training.pt").open("wb") as handle:
        torch.save(bundle, handle)
        handle.flush()
        os.fsync(handle.fileno())
    with (temporary / "train_state.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(destination)
    latest = run_dir / ".latest.json.tmp"
    latest.write_text(json.dumps({"checkpoint": str(destination), "step": state["step"]}, indent=2))
    latest.replace(run_dir / "latest.json")
    prune_checkpoints(run_dir, keep)
    return destination


def load_checkpoint(path: Path, model, optimizer, scaler) -> dict:
    bundle = torch.load(path / "training.pt", map_location="cpu", weights_only=True, mmap=True)
    if bundle["format_version"] != 1:
        raise ValueError("unsupported CUDA checkpoint format")
    state = bundle["state"]
    check_model_config(state["model"], model.config.to_dict())
    model.load_state_dict(bundle["model"], strict=True)
    optimizer.load_state_dict(bundle["optimizer"])
    scaler.load_state_dict(bundle["scaler"])
    torch.set_rng_state(bundle["rng"])
    if bundle["cuda_rng"]:
        torch.cuda.set_rng_state_all(bundle["cuda_rng"])
    return state


def load_model_weights(path: Path, model) -> dict:
    """Start a new run from weights without inheriting its data, Adam or counters."""
    bundle = torch.load(path / "training.pt", map_location="cpu", weights_only=True, mmap=True)
    if bundle["format_version"] != 1:
        raise ValueError("unsupported CUDA checkpoint format")
    state = bundle["state"]
    check_model_config(state["model"], model.config.to_dict())
    model.load_state_dict(bundle["model"], strict=True)
    return state
