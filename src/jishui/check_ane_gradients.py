from __future__ import annotations

import argparse
import json
import math
import mmap
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from .ane_profiles import resolve_ane_backward_scales
from .ane_reference import (
    NativeBackwardScales,
    NativeConstants,
    embedding_lookup_backward,
    native_classifier_backward,
    native_forward_residuals,
    native_layer_backward,
    native_layer_forward,
)
from .config import load_run_config
from .hybrid_protocol import iter_tensor_slices, tensor_file_bytes
from .model import JishuiForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare native ANE gradients with a layerwise mixed-precision VJP."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/jishui-200m-mlx.json")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--loss-scale", type=float)
    parser.add_argument("--ffn-bwd-scale", type=float)
    parser.add_argument("--ffn-w13-bwd-scale", type=float)
    parser.add_argument("--wo-bwd-scale", type=float)
    parser.add_argument("--sdpa-bwd-scale", type=float)
    parser.add_argument("--qkv-bwd-scale", type=float)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


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


def _layer_group(name: str) -> str:
    if name.startswith("layers."):
        return name.split(".", 2)[1]
    return name.split(".", 1)[0]


def _empty_totals() -> dict[str, float]:
    return {
        "dot": 0.0,
        "reference_sq": 0.0,
        "candidate_sq": 0.0,
        "difference_sq": 0.0,
    }


def _finalize(values: dict[str, float]) -> dict[str, float]:
    reference_norm = math.sqrt(values["reference_sq"])
    candidate_norm = math.sqrt(values["candidate_sq"])
    denominator = reference_norm * candidate_norm
    return {
        "cosine": values["dot"] / max(denominator, 1.0e-30),
        "relative_l2": math.sqrt(values["difference_sq"])
        / max(reference_norm, 1.0e-30),
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "difference_norm": math.sqrt(values["difference_sq"]),
    }


class GradientMetrics:
    def __init__(self, native: dict[str, np.ndarray], top_k: int):
        self.native = native
        self.top_k = top_k
        self.total = _empty_totals()
        self.groups: dict[str, dict[str, float]] = defaultdict(_empty_totals)
        self.layers: dict[str, dict[str, float]] = defaultdict(_empty_totals)
        self.tensors: list[dict[str, float | str]] = []
        self.maximum = 0.0
        self.worst_name = ""
        self.worst_relative_l2 = 0.0
        self.seen: set[str] = set()

    @staticmethod
    def _merge(destination: dict[str, float], source: dict[str, float]) -> None:
        for name, value in source.items():
            destination[name] += value

    def add(self, name: str, reference: mx.array) -> None:
        if name in self.seen:
            raise ValueError(f"duplicate reference gradient: {name}")
        try:
            native_value = self.native[name]
        except KeyError as error:
            raise ValueError(f"native gradient file has no tensor named {name}") from error
        if tuple(reference.shape) != tuple(native_value.shape):
            raise ValueError(
                f"gradient shape mismatch for {name}: "
                f"reference={reference.shape} native={native_value.shape}"
            )
        candidate = mx.array(native_value).astype(mx.float32)
        reference = reference.astype(mx.float32)
        difference = candidate - reference
        probes = (
            mx.sum(reference * candidate),
            mx.sum(reference * reference),
            mx.sum(candidate * candidate),
            mx.sum(difference * difference),
            mx.max(mx.abs(difference)),
        )
        mx.eval(*probes)
        values = {
            "dot": float(probes[0].item()),
            "reference_sq": float(probes[1].item()),
            "candidate_sq": float(probes[2].item()),
            "difference_sq": float(probes[3].item()),
        }
        reference_norm = math.sqrt(values["reference_sq"])
        candidate_norm = math.sqrt(values["candidate_sq"])
        relative_l2 = math.sqrt(values["difference_sq"]) / max(
            reference_norm, 1.0e-12
        )
        tensor = {
            "name": name,
            "cosine": values["dot"]
            / max(reference_norm * candidate_norm, 1.0e-30),
            "relative_l2": relative_l2,
            "reference_norm": reference_norm,
            "candidate_norm": candidate_norm,
            "difference_norm": math.sqrt(values["difference_sq"]),
        }
        self.tensors.append(tensor)
        self._merge(self.total, values)
        self._merge(self.groups[_gradient_group(name)], values)
        self._merge(self.layers[_layer_group(name)], values)
        self.maximum = max(self.maximum, float(probes[4].item()))
        if relative_l2 > self.worst_relative_l2:
            self.worst_relative_l2 = relative_l2
            self.worst_name = name
        self.seen.add(name)

    def result(self) -> dict[str, object]:
        missing = sorted(set(self.native) - self.seen)
        if missing:
            raise ValueError(
                f"reference did not produce {len(missing)} native tensors; first={missing[0]}"
            )
        tensors = sorted(
            self.tensors, key=lambda value: float(value["difference_norm"]), reverse=True
        )
        layers = [
            {"name": name, **_finalize(values)} for name, values in self.layers.items()
        ]
        layers.sort(key=lambda value: float(value["difference_norm"]), reverse=True)
        return {
            **_finalize(self.total),
            "max_abs": self.maximum,
            "worst_tensor": self.worst_name,
            "worst_tensor_relative_l2": self.worst_relative_l2,
            "groups": {
                name: _finalize(values) for name, values in sorted(self.groups.items())
            },
            "largest_difference_tensors": tensors[: self.top_k],
            "largest_difference_layers": layers[: self.top_k],
        }


def _resolve_model_path(path: Path) -> Path:
    return path / "model.safetensors" if path.is_dir() else path


def _load_record(path: Path, width: int, index: int, vocab_size: int) -> np.ndarray:
    if index < 0:
        raise ValueError("--record-index must be non-negative")
    item_bytes = np.dtype("<u2").itemsize
    size = path.stat().st_size
    if size == 0 or size % (width * item_bytes):
        raise ValueError(f"records file is not a multiple of {width} uint16 values")
    count = size // (width * item_bytes)
    if index >= count:
        raise ValueError(f"record index {index} is outside a {count}-record file")
    records = np.memmap(path, dtype="<u2", mode="r", shape=(count, width))
    record = np.asarray(records[index], dtype=np.int32)
    if np.any(record >= vocab_size):
        raise ValueError("record contains a token outside the model vocabulary")
    return record


def run(args: argparse.Namespace) -> dict[str, object]:
    model_config, train_config = load_run_config(args.config)
    resolved_scales = resolve_ane_backward_scales(
        model_config,
        {
            "ffn_bwd_scale": args.ffn_bwd_scale,
            "ffn_w13_bwd_scale": args.ffn_w13_bwd_scale,
            "wo_bwd_scale": args.wo_bwd_scale,
            "sdpa_bwd_scale": args.sdpa_bwd_scale,
            "qkv_bwd_scale": args.qkv_bwd_scale,
        },
    )
    scales = NativeBackwardScales(
        loss=(train_config.loss_scale if args.loss_scale is None else args.loss_scale),
        ffn_w2=resolved_scales["ffn_bwd_scale"],
        ffn_w13=resolved_scales["ffn_w13_bwd_scale"],
        wo=resolved_scales["wo_bwd_scale"],
        sdpa=resolved_scales["sdpa_bwd_scale"],
        qkv=resolved_scales["qkv_bwd_scale"],
    )
    record = _load_record(
        args.records,
        model_config.max_position_embeddings + 1,
        args.record_index,
        model_config.vocab_size,
    )
    inputs = mx.array(record[:-1])
    targets = mx.array(record[1:])

    model = JishuiForCausalLM(model_config)
    model.set_dtype(mx.float16)
    model.load_weights(str(_resolve_model_path(args.checkpoint)), strict=True)
    mx.eval(model.parameters())
    model.train()

    if args.gradients.stat().st_size != tensor_file_bytes(model_config):
        raise ValueError("native gradient tensor file has the wrong size")
    constants = NativeConstants.from_config(model_config)
    started = time.monotonic()
    print("ANE reference: replaying native-boundary forward", flush=True)
    residuals = native_forward_residuals(model, inputs, model_config, constants)
    forward_seconds = time.monotonic() - started
    print(
        "ANE reference: forward complete seconds=%.3f peak_gib=%.3f"
        % (forward_seconds, mx.get_peak_memory() / 2**30),
        flush=True,
    )

    with args.gradients.open("rb") as gradient_handle:
        mapping = mmap.mmap(gradient_handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            native = {
                spec.name: values
                for spec, values in iter_tensor_slices(mapping, model_config)
            }
            metrics = GradientMetrics(native, args.top_k)
            (
                loss,
                upstream,
                classifier_embedding_gradient,
                final_norm_weight,
                final_norm_bias,
            ) = native_classifier_backward(
                model,
                residuals[-1],
                targets,
                model_config,
                scales.loss,
            )
            metrics.add("norm.weight", final_norm_weight / scales.loss)
            metrics.add("norm.bias", final_norm_bias / scales.loss)

            backward_started = time.monotonic()
            for layer_index in range(model_config.num_hidden_layers - 1, -1, -1):
                _, cache = native_layer_forward(
                    model.layers[layer_index],
                    residuals[layer_index],
                    model_config,
                    constants,
                )
                next_upstream, layer_gradients = native_layer_backward(
                    model.layers[layer_index],
                    cache,
                    upstream,
                    model_config,
                    constants,
                    scales,
                )
                mx.eval(next_upstream, layer_gradients)
                prefix = f"layers.{layer_index}."
                for local_name, gradient in layer_gradients.items():
                    metrics.add(prefix + local_name, gradient / scales.loss)
                upstream = mx.stop_gradient(next_upstream)
                del cache, layer_gradients, next_upstream
                if layer_index % 5 == 0 or layer_index == model_config.num_hidden_layers - 1:
                    print(
                        "ANE reference: backward layer=%d peak_gib=%.3f"
                        % (layer_index, mx.get_peak_memory() / 2**30),
                        flush=True,
                    )

            lookup_gradient = embedding_lookup_backward(
                inputs,
                upstream,
                tuple(model.embed_tokens.weight.shape),
            )
            embedding_gradient = classifier_embedding_gradient + lookup_gradient
            mx.eval(embedding_gradient)
            metrics.add("embed_tokens.weight", embedding_gradient / scales.loss)
            result = metrics.result()
        finally:
            mapping.close()

    result.update(
        {
            "loss": loss,
            "forward_seconds": forward_seconds,
            "backward_seconds": time.monotonic() - backward_started,
            "peak_gib": mx.get_peak_memory() / 2**30,
            "record_index": args.record_index,
            "scales": {
                "loss": scales.loss,
                "ffn_w2": scales.ffn_w2,
                "ffn_w13": scales.ffn_w13,
                "wo": scales.wo,
                "sdpa": scales.sdpa,
                "qkv": scales.qkv,
            },
        }
    )
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
