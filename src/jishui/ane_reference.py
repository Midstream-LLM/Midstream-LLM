from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .config import ModelConfig


@dataclass(frozen=True)
class NativeBackwardScales:
    loss: float = 256.0
    ffn_w2: float = 1.0
    ffn_w13: float = 1.0
    wo: float = 1.0
    sdpa: float = 1.0
    qkv: float = 1.0

    def __post_init__(self) -> None:
        values = (self.loss, self.ffn_w2, self.ffn_w13, self.wo, self.sdpa, self.qkv)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("native backward scales must be finite and positive")


@dataclass(frozen=True)
class NativeConstants:
    causal_mask: mx.array
    rope_cos_f16: mx.array
    rope_sin_f16: mx.array
    rope_cos_f32: mx.array
    rope_sin_f32: mx.array

    @classmethod
    def from_config(cls, config: ModelConfig) -> "NativeConstants":
        sequence = config.max_position_embeddings
        half_dim = config.head_dim // 2
        positions = np.arange(sequence, dtype=np.float32)[:, None]
        indexes = np.arange(half_dim, dtype=np.float32)[None, :]
        exponents = np.float32(2.0) * indexes / np.float32(config.head_dim)
        frequencies = np.float32(1.0) / np.power(
            np.float32(config.rope_theta), exponents
        ).astype(np.float32)
        angles = (positions * frequencies).astype(np.float32)
        cos_f32 = np.cos(angles).astype(np.float32)
        sin_f32 = np.sin(angles).astype(np.float32)
        cos_full = np.concatenate((cos_f32, cos_f32), axis=-1).astype(np.float16)
        sin_full = np.concatenate((sin_f32, sin_f32), axis=-1).astype(np.float16)
        mask = np.triu(
            np.full((sequence, sequence), -65504.0, dtype=np.float16), k=1
        )[None, None, :, :]
        return cls(
            causal_mask=mx.array(mask),
            rope_cos_f16=mx.array(cos_full)[None, :, :],
            rope_sin_f16=mx.array(sin_full)[None, :, :],
            rope_cos_f32=mx.array(cos_f32),
            rope_sin_f32=mx.array(sin_f32),
        )


@dataclass(frozen=True)
class NativeLayerCache:
    layer_input: mx.array
    xnorm: mx.array
    q: mx.array
    k: mx.array
    v: mx.array
    attention_output: mx.array
    x2: mx.array
    x2norm: mx.array
    h1: mx.array
    h3: mx.array
    gate: mx.array


def _half(value: mx.array) -> mx.array:
    return value.astype(mx.float16)


def _half_matmul(left: mx.array, right: mx.array) -> mx.array:
    return mx.matmul(_half(left), _half(right)).astype(mx.float16)


def layer_norm_forward(module, value: mx.array, eps: float) -> mx.array:
    value = value.astype(mx.float32)
    mean = mx.mean(value, axis=-1, keepdims=True)
    centered = value - mean
    inverse = mx.rsqrt(mx.mean(centered * centered, axis=-1, keepdims=True) + eps)
    return (
        centered
        * inverse
        * module.weight.astype(mx.float32)
        + module.bias.astype(mx.float32)
    )


def layer_norm_backward(
    upstream: mx.array,
    value: mx.array,
    weight: mx.array,
    eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    upstream = upstream.astype(mx.float32)
    value = value.astype(mx.float32)
    weight = weight.astype(mx.float32)
    mean = mx.mean(value, axis=-1, keepdims=True)
    centered = value - mean
    inverse = mx.rsqrt(mx.mean(centered * centered, axis=-1, keepdims=True) + eps)
    normalized = centered * inverse
    weighted = upstream * weight
    dx = inverse * (
        weighted
        - mx.mean(weighted, axis=-1, keepdims=True)
        - normalized * mx.mean(weighted * normalized, axis=-1, keepdims=True)
    )
    dweight = mx.sum(upstream * normalized, axis=0)
    dbias = mx.sum(upstream, axis=0)
    return dx, dweight, dbias


def _rope_forward(value: mx.array, constants: NativeConstants) -> mx.array:
    half_dim = value.shape[-1] // 2
    first = value[..., :half_dim]
    second = value[..., half_dim:]
    rotated = mx.concatenate((-second, first), axis=-1).astype(mx.float16)
    direct = (value * constants.rope_cos_f16).astype(mx.float16)
    crossed = (rotated * constants.rope_sin_f16).astype(mx.float16)
    return (direct + crossed).astype(mx.float16)


def _rope_backward(
    value: mx.array,
    config: ModelConfig,
    constants: NativeConstants,
) -> mx.array:
    sequence, width = value.shape
    heads = width // config.head_dim
    reshaped = value.reshape(sequence, heads, config.head_dim).astype(mx.float32)
    half_dim = config.head_dim // 2
    first = reshaped[..., :half_dim]
    second = reshaped[..., half_dim:]
    cos = constants.rope_cos_f32[:, None, :]
    sin = constants.rope_sin_f32[:, None, :]
    return mx.concatenate(
        (first * cos + second * sin, -first * sin + second * cos), axis=-1
    ).reshape(sequence, width)


def _repeat_kv(value: mx.array, repeats: int) -> mx.array:
    return value if repeats == 1 else mx.repeat(value, repeats, axis=0)


def _reduce_kv(value: mx.array, kv_heads: int, repeats: int) -> mx.array:
    if repeats == 1:
        return value.astype(mx.float32)
    _, sequence, head_dim = value.shape
    return mx.sum(
        value.astype(mx.float32).reshape(kv_heads, repeats, sequence, head_dim),
        axis=1,
    )


def native_layer_forward(
    layer,
    value: mx.array,
    config: ModelConfig,
    constants: NativeConstants,
) -> tuple[mx.array, NativeLayerCache]:
    sequence = value.shape[0]
    attention = layer.attention
    xnorm = layer_norm_forward(layer.input_norm, value, config.norm_eps)
    xnorm_half = xnorm.astype(mx.float16)

    q = _half_matmul(xnorm_half, attention.q_proj.weight.T)
    k = _half_matmul(xnorm_half, attention.k_proj.weight.T)
    v = _half_matmul(xnorm_half, attention.v_proj.weight.T)
    q = q.reshape(sequence, config.num_attention_heads, config.head_dim).transpose(
        1, 0, 2
    )
    k = k.reshape(sequence, config.num_key_value_heads, config.head_dim).transpose(
        1, 0, 2
    )
    v = v.reshape(sequence, config.num_key_value_heads, config.head_dim).transpose(
        1, 0, 2
    )
    q = _rope_forward(q, constants)
    k = _rope_forward(k, constants)

    repeats = config.num_attention_heads // config.num_key_value_heads
    attention_k = _repeat_kv(k, repeats)
    attention_v = _repeat_kv(v, repeats)
    scale = mx.array(config.head_dim**-0.5, dtype=mx.float16)
    scores = _half_matmul(q, attention_k.transpose(0, 2, 1))
    scores = (scores * scale).astype(mx.float16)
    probabilities = mx.softmax(
        (scores[None, :, :, :] + constants.causal_mask).astype(mx.float16),
        axis=-1,
    ).astype(mx.float16)[0]
    attention_heads = _half_matmul(probabilities, attention_v)
    attention_output_half = attention_heads.transpose(1, 0, 2).reshape(
        sequence, -1
    )
    projected = _half_matmul(attention_output_half, attention.o_proj.weight.T)
    x2 = value.astype(mx.float32) + projected.astype(mx.float32)

    x2norm = layer_norm_forward(layer.post_attention_norm, x2, config.norm_eps)
    x2norm_half = x2norm.astype(mx.float16)
    h1 = _half_matmul(x2norm_half, layer.mlp.gate_proj.weight.T)
    h3 = _half_matmul(x2norm_half, layer.mlp.up_proj.weight.T)
    silu = (h1 * mx.sigmoid(h1)).astype(mx.float16)
    gate = (silu * h3).astype(mx.float16)
    ffn_output = _half_matmul(gate, layer.mlp.down_proj.weight.T)
    next_value = (x2.astype(mx.float16) + ffn_output).astype(mx.float16).astype(
        mx.float32
    )

    cache = NativeLayerCache(
        layer_input=value.astype(mx.float32),
        xnorm=xnorm,
        q=q.transpose(1, 0, 2).reshape(sequence, -1).astype(mx.float32),
        k=k.transpose(1, 0, 2).reshape(sequence, -1).astype(mx.float32),
        v=v.transpose(1, 0, 2).reshape(sequence, -1).astype(mx.float32),
        attention_output=attention_output_half.astype(mx.float32),
        x2=x2,
        x2norm=x2norm,
        h1=h1.astype(mx.float32),
        h3=h3.astype(mx.float32),
        gate=gate.astype(mx.float32),
    )
    return next_value, cache


def _sdpa_backward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    upstream: mx.array,
    config: ModelConfig,
    constants: NativeConstants,
    scale: float,
) -> tuple[mx.array, mx.array, mx.array]:
    sequence = q.shape[0]
    q_heads = q.reshape(sequence, config.num_attention_heads, config.head_dim).transpose(
        1, 0, 2
    )
    k_heads = k.reshape(sequence, config.num_key_value_heads, config.head_dim).transpose(
        1, 0, 2
    )
    v_heads = v.reshape(sequence, config.num_key_value_heads, config.head_dim).transpose(
        1, 0, 2
    )
    repeats = config.num_attention_heads // config.num_key_value_heads
    tiled_k = _repeat_kv(k_heads, repeats).astype(mx.float16)
    tiled_v = _repeat_kv(v_heads, repeats).astype(mx.float16)
    q_heads = q_heads.astype(mx.float16)
    scaled_upstream = (upstream * scale).astype(mx.float16)
    scaled_upstream = scaled_upstream.reshape(
        sequence, config.num_attention_heads, config.head_dim
    ).transpose(1, 0, 2)

    attention_scale = mx.array(config.head_dim**-0.5, dtype=mx.float16)
    scores = _half_matmul(q_heads, tiled_k.transpose(0, 2, 1))
    scores = (scores * attention_scale).astype(mx.float16)
    probabilities = mx.softmax(
        (scores[None, :, :, :] + constants.causal_mask).astype(mx.float16),
        axis=-1,
    ).astype(mx.float16)[0]
    dv_full = _half_matmul(probabilities.transpose(0, 2, 1), scaled_upstream)
    dprobabilities = _half_matmul(scaled_upstream, tiled_v.transpose(0, 2, 1))
    product = (probabilities * dprobabilities).astype(mx.float16)
    product_sum = mx.sum(product, axis=-1, keepdims=True).astype(mx.float16)
    centered = (dprobabilities - product_sum).astype(mx.float16)
    dscores = (probabilities * centered).astype(mx.float16)
    dscores = (dscores * attention_scale).astype(mx.float16)
    dq_full = _half_matmul(dscores, tiled_k)
    dk_full = _half_matmul(dscores.transpose(0, 2, 1), q_heads)

    dk_heads = _reduce_kv(dk_full, config.num_key_value_heads, repeats)
    dv_heads = _reduce_kv(dv_full, config.num_key_value_heads, repeats)
    inverse = mx.array(1.0 / scale, dtype=mx.float32)
    dq = dq_full.transpose(1, 0, 2).reshape(sequence, -1).astype(mx.float32)
    dk = dk_heads.transpose(1, 0, 2).reshape(sequence, -1)
    dv = dv_heads.transpose(1, 0, 2).reshape(sequence, -1)
    return dq * inverse, dk * inverse, dv * inverse


def native_layer_backward(
    layer,
    cache: NativeLayerCache,
    upstream: mx.array,
    config: ModelConfig,
    constants: NativeConstants,
    scales: NativeBackwardScales,
) -> tuple[mx.array, dict[str, mx.array]]:
    upstream = upstream.astype(mx.float32)
    dffn = upstream
    dsilu = _half_matmul(
        (dffn * scales.ffn_w2).astype(mx.float16),
        layer.mlp.down_proj.weight,
    ).astype(mx.float32) / scales.ffn_w2

    sigmoid = mx.sigmoid(cache.h1)
    silu = cache.h1 * sigmoid
    dh3 = dsilu * silu
    dh1 = (
        dsilu
        * cache.h3
        * sigmoid
        * (1.0 + cache.h1 * (1.0 - sigmoid))
    )
    dx_ffn = (
        _half_matmul(
            (dh1 * scales.ffn_w13).astype(mx.float16),
            layer.mlp.gate_proj.weight,
        )
        + _half_matmul(
            (dh3 * scales.ffn_w13).astype(mx.float16),
            layer.mlp.up_proj.weight,
        )
    ).astype(mx.float16).astype(mx.float32) / scales.ffn_w13

    dx2_norm, post_weight, post_bias = layer_norm_backward(
        dx_ffn,
        cache.x2,
        layer.post_attention_norm.weight,
        config.norm_eps,
    )
    dx2 = dx2_norm + upstream
    da = _half_matmul(
        (dx2 * scales.wo).astype(mx.float16),
        layer.attention.o_proj.weight,
    ).astype(mx.float32) / scales.wo

    dq, dk, dv = _sdpa_backward(
        cache.q,
        cache.k,
        cache.v,
        da,
        config,
        constants,
        scales.sdpa,
    )
    dq = _rope_backward(dq, config, constants)
    dk = _rope_backward(dk, config, constants)

    dx_q = _half_matmul(
        (dq * scales.qkv).astype(mx.float16),
        layer.attention.q_proj.weight,
    ).astype(mx.float32) / scales.qkv
    dx_kv = (
        _half_matmul(
            (dk * scales.qkv).astype(mx.float16),
            layer.attention.k_proj.weight,
        )
        + _half_matmul(
            (dv * scales.qkv).astype(mx.float16),
            layer.attention.v_proj.weight,
        )
    ).astype(mx.float16).astype(mx.float32) / scales.qkv
    dx_attention = dx_q + dx_kv
    dx_input_norm, input_weight, input_bias = layer_norm_backward(
        dx_attention,
        cache.layer_input,
        layer.input_norm.weight,
        config.norm_eps,
    )

    gradients = {
        "input_norm.weight": input_weight,
        "input_norm.bias": input_bias,
        "attention.q_proj.weight": mx.matmul(dq.T, cache.xnorm),
        "attention.k_proj.weight": mx.matmul(dk.T, cache.xnorm),
        "attention.v_proj.weight": mx.matmul(dv.T, cache.xnorm),
        "attention.o_proj.weight": mx.matmul(dx2.T, cache.attention_output),
        "post_attention_norm.weight": post_weight,
        "post_attention_norm.bias": post_bias,
        "mlp.gate_proj.weight": mx.matmul(dh1.T, cache.x2norm),
        "mlp.up_proj.weight": mx.matmul(dh3.T, cache.x2norm),
        "mlp.down_proj.weight": mx.matmul(dffn.T, cache.gate),
    }
    return dx_input_norm + dx2, gradients


def native_forward_residuals(
    model,
    input_ids: mx.array,
    config: ModelConfig,
    constants: NativeConstants,
) -> list[mx.array]:
    value = model.embed_tokens(input_ids).astype(mx.float32)
    mx.eval(value)
    residuals = [mx.stop_gradient(value)]
    for layer in model.layers:
        value, cache = native_layer_forward(layer, value, config, constants)
        mx.eval(value)
        value = mx.stop_gradient(value)
        residuals.append(value)
        del cache
    return residuals


def native_classifier_backward(
    model,
    final_residual: mx.array,
    targets: mx.array,
    config: ModelConfig,
    loss_scale: float,
) -> tuple[float, mx.array, mx.array, mx.array, mx.array]:
    final_value = layer_norm_forward(model.norm, final_residual, config.norm_eps)
    embedding = model.embed_tokens.weight.astype(mx.float32)
    logits = mx.matmul(final_value, embedding.T)
    probabilities = mx.softmax(logits, axis=-1).astype(mx.float32)
    target_indexes = targets[:, None]
    target_probabilities = mx.take_along_axis(
        probabilities, target_indexes, axis=-1
    )
    loss = -mx.mean(mx.log(target_probabilities + 1.0e-10))
    target_values = target_probabilities - 1.0
    dlogits = mx.put_along_axis(
        probabilities, target_indexes, target_values, axis=-1
    )
    dlogits = dlogits * (loss_scale / targets.size)
    classifier_embedding_gradient = mx.matmul(dlogits.T, final_value)
    final_upstream = mx.matmul(dlogits, embedding)
    residual_gradient, norm_weight, norm_bias = layer_norm_backward(
        final_upstream,
        final_residual,
        model.norm.weight,
        config.norm_eps,
    )
    mx.eval(
        loss,
        residual_gradient,
        classifier_embedding_gradient,
        norm_weight,
        norm_bias,
    )
    return (
        float(loss.item()),
        residual_gradient,
        classifier_embedding_gradient,
        norm_weight,
        norm_bias,
    )


def embedding_lookup_backward(
    input_ids: mx.array,
    upstream: mx.array,
    shape: tuple[int, int],
) -> mx.array:
    return mx.zeros(shape, dtype=mx.float32).at[input_ids].add(upstream)
