from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint

from .config import ModelConfig


def _make_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
    return nn.LayerNorm(
        config.hidden_size,
        eps=config.norm_eps,
        affine=True,
        bias=config.norm_bias,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = self.rope(q)
        k = self.rope(k)

        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        y = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask="causal",
        )
        y = y.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_norm = _make_norm(config)
        self.attention = CausalSelfAttention(config)
        self.post_attention_norm = _make_norm(config)
        self.mlp = SwiGLU(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.input_norm(x))
        return x + self.mlp(self.post_attention_norm(x))


class JishuiForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = _make_norm(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        if input_ids.shape[-1] > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {input_ids.shape[-1]} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = checkpoint(layer)(x) if self.config.gradient_checkpointing else layer(x)
        x = self.norm(x)
        if self.config.tie_word_embeddings:
            return self.embed_tokens.as_linear(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        from mlx.utils import tree_flatten

        return sum(value.size for _, value in tree_flatten(self.parameters()))


def causal_lm_loss(model: JishuiForCausalLM, inputs: mx.array, targets: mx.array) -> mx.array:
    logits = model(inputs)
    return nn.losses.cross_entropy(
        logits.astype(mx.float32), targets, reduction="mean"
    )
