from __future__ import annotations

import mmap
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from .config import ModelConfig


HYBRID_MAGIC = b"JSHHYB1\0"
HYBRID_VERSION = 1
HYBRID_HEADER_BYTES = 4096
HYBRID_KIND_WEIGHTS = 1
HYBRID_KIND_GRADIENTS = 2
HYBRID_CONTROL_BYTES = 4096
HYBRID_CMD_RUN = 1
HYBRID_CMD_STOP = 2
HYBRID_STATUS_READY = 1
HYBRID_STATUS_RUNNING = 2
HYBRID_STATUS_GRAD_READY = 3
HYBRID_STATUS_STOPPED = 4
HYBRID_STATUS_ERROR = 255
_HEADER = struct.Struct("<8s4IQ6I")
_CONTROL = struct.Struct("<8s6IQ")


@dataclass(frozen=True)
class HybridControlState:
    command: int
    status: int
    generation: int
    microsteps: int
    record_base: int


def initialize_control_file(handle: BinaryIO) -> mmap.mmap:
    """Create a zeroed, versioned control page for one native worker."""
    handle.truncate(HYBRID_CONTROL_BYTES)
    mapping = mmap.mmap(handle.fileno(), HYBRID_CONTROL_BYTES, access=mmap.ACCESS_WRITE)
    mapping[:] = b"\0" * HYBRID_CONTROL_BYTES
    _CONTROL.pack_into(
        mapping,
        0,
        HYBRID_MAGIC,
        HYBRID_VERSION,
        HYBRID_CONTROL_BYTES,
        0,
        0,
        0,
        0,
        0,
    )
    mapping.flush()
    return mapping


def _validate_control(mapping: mmap.mmap) -> None:
    if len(mapping) != HYBRID_CONTROL_BYTES:
        raise ValueError("invalid hybrid control mapping size")
    magic, version, size, *_ = _CONTROL.unpack_from(mapping, 0)
    if magic != HYBRID_MAGIC or version != HYBRID_VERSION or size != HYBRID_CONTROL_BYTES:
        raise ValueError("hybrid control header does not match")


def control_state(mapping: mmap.mmap) -> HybridControlState:
    _validate_control(mapping)
    _, _, _, command, status, generation, microsteps, record_base = _CONTROL.unpack_from(
        mapping, 0
    )
    return HybridControlState(command, status, generation, microsteps, record_base)


def publish_generation(
    mapping: mmap.mmap,
    generation: int,
    record_base: int,
    microsteps: int = 1,
) -> None:
    """Publish weights and ask the native worker to consume one generation.

    Callers must flush the weight mapping before this function. The worker
    checks command and generation with acquire loads, so publishing RUN last
    prevents it from observing a new command before the payload is complete.
    """
    _validate_control(mapping)
    if generation < 1 or generation > 0xFFFFFFFF:
        raise ValueError("generation must fit uint32 and be positive")
    if record_base < 0 or record_base > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("record_base must fit uint64")
    if microsteps < 1 or microsteps > 0xFFFFFFFF:
        raise ValueError("microsteps must fit uint32 and be positive")
    current = control_state(mapping)
    if generation <= current.generation:
        raise ValueError("generation must increase monotonically")
    # Write the payload and generation while command remains idle, flush it,
    # then publish RUN as the final 32-bit field. The native side may still be
    # publishing STOPPED from the previous generation.
    _CONTROL.pack_into(
        mapping,
        0,
        HYBRID_MAGIC,
        HYBRID_VERSION,
        HYBRID_CONTROL_BYTES,
        0,
        HYBRID_STATUS_READY,
        generation,
        microsteps,
        record_base,
    )
    mapping.flush()
    struct.pack_into("<I", mapping, 16, HYBRID_CMD_RUN)
    mapping.flush()


def publish_stop(mapping: mmap.mmap) -> None:
    """Ask the native worker to leave its generation loop."""
    _validate_control(mapping)
    state = control_state(mapping)
    _CONTROL.pack_into(
        mapping,
        0,
        HYBRID_MAGIC,
        HYBRID_VERSION,
        HYBRID_CONTROL_BYTES,
        HYBRID_CMD_STOP,
        state.status,
        state.generation,
        state.microsteps,
        state.record_base,
    )
    mapping.flush()


def wait_for_status(
    mapping: mmap.mmap,
    expected: int | set[int],
    timeout: float,
) -> HybridControlState:
    expected_statuses = {expected} if isinstance(expected, int) else set(expected)
    deadline = time.monotonic() + timeout
    while True:
        state = control_state(mapping)
        if state.status in expected_statuses:
            return state
        if state.status == HYBRID_STATUS_ERROR:
            raise RuntimeError("native hybrid worker reported an error")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for hybrid status {sorted(expected_statuses)}; "
                f"last status={state.status} generation={state.generation}"
            )
        time.sleep(0.005)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


def tensor_specs(config: ModelConfig) -> tuple[TensorSpec, ...]:
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    dim = config.hidden_size
    hidden = config.intermediate_size
    specs = [TensorSpec("embed_tokens.weight", (config.vocab_size, dim))]
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        specs.extend(
            (
                TensorSpec(f"{prefix}.input_norm.weight", (dim,)),
                TensorSpec(f"{prefix}.input_norm.bias", (dim,)),
                TensorSpec(f"{prefix}.attention.q_proj.weight", (q_dim, dim)),
                TensorSpec(f"{prefix}.attention.k_proj.weight", (kv_dim, dim)),
                TensorSpec(f"{prefix}.attention.v_proj.weight", (kv_dim, dim)),
                TensorSpec(f"{prefix}.attention.o_proj.weight", (dim, q_dim)),
                TensorSpec(f"{prefix}.post_attention_norm.weight", (dim,)),
                TensorSpec(f"{prefix}.post_attention_norm.bias", (dim,)),
                TensorSpec(f"{prefix}.mlp.gate_proj.weight", (hidden, dim)),
                TensorSpec(f"{prefix}.mlp.up_proj.weight", (hidden, dim)),
                TensorSpec(f"{prefix}.mlp.down_proj.weight", (dim, hidden)),
            )
        )
    specs.extend((TensorSpec("norm.weight", (dim,)), TensorSpec("norm.bias", (dim,))))
    return tuple(specs)


def parameter_count(config: ModelConfig) -> int:
    return sum(spec.size for spec in tensor_specs(config))


def tensor_file_bytes(config: ModelConfig) -> int:
    return HYBRID_HEADER_BYTES + parameter_count(config) * np.dtype("<f2").itemsize


def _header(config: ModelConfig, kind: int) -> bytes:
    return _HEADER.pack(
        HYBRID_MAGIC,
        HYBRID_VERSION,
        HYBRID_HEADER_BYTES,
        kind,
        0,
        parameter_count(config),
        config.num_hidden_layers,
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_key_value_heads,
    )


def initialize_tensor_file(handle: BinaryIO, config: ModelConfig, kind: int) -> mmap.mmap:
    if kind not in {HYBRID_KIND_WEIGHTS, HYBRID_KIND_GRADIENTS}:
        raise ValueError(f"invalid hybrid tensor kind: {kind}")
    handle.truncate(tensor_file_bytes(config))
    mapping = mmap.mmap(handle.fileno(), tensor_file_bytes(config), access=mmap.ACCESS_WRITE)
    mapping[: _HEADER.size] = _header(config, kind)
    mapping[_HEADER.size : HYBRID_HEADER_BYTES] = b"\0" * (
        HYBRID_HEADER_BYTES - _HEADER.size
    )
    return mapping


def tensor_view(mapping: mmap.mmap, config: ModelConfig) -> np.ndarray:
    return np.ndarray(
        (parameter_count(config),),
        dtype="<f2",
        buffer=mapping,
        offset=HYBRID_HEADER_BYTES,
    )


def validate_parameter_tree(parameters: object, config: ModelConfig) -> list[tuple[str, mx.array]]:
    flat = tree_flatten(parameters)
    specs = tensor_specs(config)
    actual = [(name, tuple(value.shape)) for name, value in flat]
    expected = [(spec.name, spec.shape) for spec in specs]
    if actual != expected:
        mismatch = next(
            (
                (index, want, got)
                for index, (want, got) in enumerate(zip(expected, actual, strict=False))
                if want != got
            ),
            None,
        )
        raise ValueError(
            "MLX parameter layout does not match the native hybrid protocol: "
            f"expected={len(expected)} actual={len(actual)} first_mismatch={mismatch}"
        )
    return flat


def write_parameter_tree(
    mapping: mmap.mmap, parameters: object, config: ModelConfig
) -> None:
    destination = tensor_view(mapping, config)
    offset = 0
    for (_, value), spec in zip(
        validate_parameter_tree(parameters, config), tensor_specs(config), strict=True
    ):
        next_offset = offset + spec.size
        destination[offset:next_offset] = np.asarray(value).reshape(-1)
        offset = next_offset
    if offset != destination.size:
        raise AssertionError(f"wrote {offset} values into a {destination.size}-value map")


def read_parameter_tree(
    mapping: mmap.mmap, template: object, config: ModelConfig
) -> object:
    source = tensor_view(mapping, config)
    flat_template = validate_parameter_tree(template, config)
    values: list[mx.array] = []
    offset = 0
    for (_, _), spec in zip(flat_template, tensor_specs(config), strict=True):
        next_offset = offset + spec.size
        # MLX must own the imported gradient after the native worker starts
        # writing the next slot, so force a copy here.
        array = mx.array(source[offset:next_offset].reshape(spec.shape))
        values.append(array)
        offset = next_offset
    if offset != source.size:
        raise AssertionError(f"read {offset} values from a {source.size}-value map")

    # Replacing leaves on the original template preserves empty/non-trainable
    # subtrees (notably `attention.rope`).  Rebuilding solely from dotted leaf
    # names would drop those empty dictionaries and make the result structurally
    # incompatible with MLX's gradient tree even though all tensors match.
    values_iter = iter(values)
    rebuilt = tree_map(lambda _: next(values_iter), template)
    try:
        next(values_iter)
    except StopIteration:
        return rebuilt
    raise AssertionError("gradient template contained fewer leaves than expected")


def iter_tensor_slices(
    mapping: mmap.mmap, config: ModelConfig
) -> Iterable[tuple[TensorSpec, np.ndarray]]:
    values = tensor_view(mapping, config)
    offset = 0
    for spec in tensor_specs(config):
        next_offset = offset + spec.size
        yield spec, values[offset:next_offset].reshape(spec.shape)
        offset = next_offset
