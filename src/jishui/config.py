from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    norm_type: str = "layernorm"
    norm_eps: float = 1.0e-5
    norm_bias: bool = True
    rope_theta: float = 10_000.0
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim: "
                f"{self.hidden_size} != {self.num_attention_heads} * {self.head_dim}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type must be layernorm or rmsnorm")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        return cls(**value)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    dtype: str = "float16"
    seq_len: int = 2048
    batch_size: int = 1
    grad_accum_steps: int = 8
    max_steps: int = 10_000
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1.0e-5
    grad_clip: float = 1.0
    log_interval: int = 1
    eval_interval: int = 200
    eval_batches: int = 10
    save_interval: int = 200
    max_checkpoints: int = 3
    # By default every retained checkpoint is an exact-resume point. Lowering
    # this is supported for storage-constrained experiments, but is opt-in.
    optimizer_checkpoints: int = 3
    sampling_mode: str = "target"
    eod_token_id: int = 2
    compile_step: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if min(self.seq_len, self.batch_size, self.grad_accum_steps, self.max_steps) < 1:
            raise ValueError("sequence, batch, accumulation, and step values must be positive")
        if self.sampling_mode not in {"target", "weights"}:
            raise ValueError("sampling_mode must be target or weights")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.max_checkpoints < 1:
            raise ValueError("max_checkpoints must be positive")
        if self.optimizer_checkpoints < 0 or self.optimizer_checkpoints > self.max_checkpoints:
            raise ValueError("optimizer_checkpoints must be between 0 and max_checkpoints")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainConfig":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_run_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return ModelConfig.from_dict(raw["model"]), TrainConfig.from_dict(raw["training"])
