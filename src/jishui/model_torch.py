"""PyTorch counterpart of model.py, with the same parameter names and RoPE."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


def make_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
    return nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)


def rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """MLX traditional=False: rotate first and second halves, not adjacent pairs."""
    size = x.shape[-1]
    inv = theta ** (-torch.arange(0, size, 2, device=x.device, dtype=torch.float32) / size)
    angles = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * inv
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * angles.cos() - b * angles.sin(),
                      a * angles.sin() + b * angles.cos()), dim=-1).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        kv_size = config.num_key_value_heads * config.head_dim
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.config
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, c.num_attention_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        q, k = rope(q, c.rope_theta), rope(k, c.rope_theta)
        if c.num_attention_heads != c.num_key_value_heads:
            repeats = c.num_attention_heads // c.num_key_value_heads
            k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.o_proj(y.transpose(1, 2).reshape(batch, length, c.hidden_size))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_norm = make_norm(config)
        self.attention = CausalSelfAttention(config)
        self.post_attention_norm = make_norm(config)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.input_norm(x))
        return x + self.mlp(self.post_attention_norm(x))


class JishuiForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = make_norm(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[-1] > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds configured context length")
        x = self.embed_tokens(input_ids)
        # Keep MLX's low-precision residual stream, while master weights and Adam
        # moments stay fp32 for reliable CUDA AMP updates.
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        for layer in self.layers:
            x = (checkpoint(layer, x, use_reentrant=False)
                 if self.training and self.config.gradient_checkpointing else layer(x))
        x = self.norm(x)
        return (F.linear(x, self.embed_tokens.weight) if self.config.tie_word_embeddings
                else self.lm_head(x))


def causal_lm_loss(model: JishuiForCausalLM, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(inputs)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
