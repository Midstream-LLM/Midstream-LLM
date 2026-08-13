from __future__ import annotations

import math
from collections.abc import Mapping

from .config import ModelConfig


ANE_BACKWARD_SCALE_FIELDS = (
    "ffn_bwd_scale",
    "ffn_w13_bwd_scale",
    "wo_bwd_scale",
    "sdpa_bwd_scale",
    "qkv_bwd_scale",
)

IDENTITY_ANE_BACKWARD_SCALES = {
    name: 1.0 for name in ANE_BACKWARD_SCALE_FIELDS
}

# Validated against the step-7200 fp16 checkpoint on three fixed records. The
# profile is sequence-specific because the native loss is averaged over SEQ.
JISHUI_200M_2048_BACKWARD_SCALES = {
    "ffn_bwd_scale": 128.0,
    "ffn_w13_bwd_scale": 32.0,
    "wo_bwd_scale": 128.0,
    "sdpa_bwd_scale": 256.0,
    "qkv_bwd_scale": 128.0,
}


def _is_jishui_200m_2048(config: ModelConfig) -> bool:
    return (
        config.vocab_size == 32768
        and config.hidden_size == 704
        and config.num_hidden_layers == 30
        and config.num_attention_heads == 11
        and config.num_key_value_heads == 11
        and config.head_dim == 64
        and config.intermediate_size == 1856
        and config.max_position_embeddings == 2048
        and config.norm_type == "layernorm"
    )


def default_ane_backward_scales(config: ModelConfig) -> dict[str, float]:
    if _is_jishui_200m_2048(config):
        return dict(JISHUI_200M_2048_BACKWARD_SCALES)
    return dict(IDENTITY_ANE_BACKWARD_SCALES)


def resolve_ane_backward_scales(
    config: ModelConfig,
    overrides: Mapping[str, float | None],
) -> dict[str, float]:
    unknown = set(overrides) - set(ANE_BACKWARD_SCALE_FIELDS)
    if unknown:
        raise ValueError(f"unknown ANE backward scale: {sorted(unknown)[0]}")
    values = default_ane_backward_scales(config)
    for name in ANE_BACKWARD_SCALE_FIELDS:
        override = overrides.get(name)
        if override is not None:
            values[name] = float(override)
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("native backward scales must be finite and positive")
    return values
