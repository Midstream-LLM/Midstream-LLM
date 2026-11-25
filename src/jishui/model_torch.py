"""PyTorch counterpart of model.py, with the same parameter names and RoPE."""
from __future__ import annotations

import math
import os
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


ATTENTION_BACKENDS = ("auto", "dense", "flex")
DENSE_MASK_MAX_LEN = 8192
_compiled_flex = None


def attention_backend(length: int, device: torch.device, requested: str | None = None) -> str:
    """Pick the document-mask kernel. Dense SDPA masks are exact and portable
    (mem-efficient kernel on V100) but cost B*L*L bytes; FlexAttention is
    block-sparse and needed beyond DENSE_MASK_MAX_LEN."""
    choice = requested or os.environ.get("JISHUI_ATTENTION_BACKEND", "auto")
    if choice not in ATTENTION_BACKENDS:
        raise ValueError(f"JISHUI_ATTENTION_BACKEND must be one of {ATTENTION_BACKENDS}")
    if choice == "auto":
        choice = "dense" if length <= DENSE_MASK_MAX_LEN else "flex"
    if choice == "dense" and length > DENSE_MASK_MAX_LEN:
        raise ValueError(f"dense document mask is limited to {DENSE_MASK_MAX_LEN} tokens; use flex")
    return choice


class DocumentMask:
    """Causal attention restricted to equal segment ids, built once per batch.

    segment_ids[b, i] names the segment of INPUT position i. Callers pass the
    document of each position's target (document_ids[:, 1:L+1]), so an EOD token
    opens the next document like a BOS: its successor sees only that EOD.
    """

    def __init__(self, segment_ids: torch.Tensor, backend: str | None = None):
        if segment_ids.dim() != 2:
            raise ValueError("segment_ids must be [batch, length]")
        self.segment_ids = segment_ids
        batch, length = segment_ids.shape
        self.backend = attention_backend(length, segment_ids.device, backend)
        if self.backend == "dense":
            causal = torch.ones(length, length, dtype=torch.bool, device=segment_ids.device).tril_()
            same = segment_ids[:, :, None] == segment_ids[:, None, :]
            self.dense = (same & causal)[:, None]           # [B, 1, L, L]; True = attend
        else:
            from torch.nn.attention.flex_attention import create_block_mask
            seg = segment_ids

            def mask_mod(b, h, q, kv):
                return (q >= kv) & (seg[b, q] == seg[b, kv])
            self.block = create_block_mask(mask_mod, B=batch, H=None, Q_LEN=length, KV_LEN=length,
                                           device=segment_ids.device, _compile=segment_ids.is_cuda)

    def attend(self, q, k, v):
        if self.backend == "dense":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=self.dense, dropout_p=0.0)
        global _compiled_flex
        from torch.nn.attention.flex_attention import flex_attention
        if not q.is_cuda:
            return flex_attention(q, k, v, block_mask=self.block)
        if _compiled_flex is None:
            _compiled_flex = torch.compile(flex_attention, dynamic=False)
        return _compiled_flex(q, k, v, block_mask=self.block)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        kv_size = config.num_key_value_heads * config.head_dim
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

    def forward(self, x: torch.Tensor, mask: DocumentMask | None = None) -> torch.Tensor:
        c = self.config
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, c.num_attention_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        q, k = rope(q, c.rope_theta), rope(k, c.rope_theta)
        if c.num_attention_heads != c.num_key_value_heads:
            repeats = c.num_attention_heads // c.num_key_value_heads
            k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        y = (F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0) if mask is None
             else mask.attend(q, k, v))
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

    def forward(self, x: torch.Tensor, mask: DocumentMask | None = None) -> torch.Tensor:
        x = x + self.attention(self.input_norm(x), mask)
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

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        """segment_ids=None is plain causal attention (SFT, inference, old runs).
        With segment ids, packed documents cannot attend to each other."""
        if input_ids.shape[-1] > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds configured context length")
        if segment_ids is not None and segment_ids.shape != input_ids.shape:
            raise ValueError("segment_ids must match input_ids")
        mask = None if segment_ids is None else DocumentMask(segment_ids)
        x = self.embed_tokens(input_ids)
        # Keep MLX's low-precision residual stream, while master weights and Adam
        # moments stay fp32 for reliable CUDA AMP updates.
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        for layer in self.layers:
            x = (checkpoint(layer, x, mask, use_reentrant=False)
                 if self.training and self.config.gradient_checkpointing else layer(x, mask))
        x = self.norm(x)
        return (F.linear(x, self.embed_tokens.weight) if self.config.tie_word_embeddings
                else self.lm_head(x))

    def initialize_pretraining(self, std: float = 0.02) -> None:
        """Explicit initialization for new runs; loading old checkpoints is unchanged."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_std = std / math.sqrt(2 * self.config.num_hidden_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attention.o_proj.weight, std=residual_std)
            nn.init.normal_(layer.mlp.down_proj.weight, std=residual_std)


def causal_lm_loss(model: JishuiForCausalLM, inputs: torch.Tensor, targets: torch.Tensor,
                   segment_ids: torch.Tensor | None = None) -> torch.Tensor:
    logits = model(inputs) if segment_ids is None else model(inputs, segment_ids)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
