"""Muon for hidden weight matrices, AdamW for everything else, in one optimizer.

A single torch.optim.Optimizer keeps GradScaler, checkpoints and exact resume
unchanged: one state_dict, one step(). Param groups carry ``use_muon``.

Muon (Jordan et al., 2024) orthogonalizes the Nesterov momentum of each 2D
weight with a quintic Newton-Schulz iteration. Updates are rescaled by
0.2 * sqrt(max(rows, cols)) so their RMS matches AdamW's (Liu et al., 2025,
"Muon is Scalable for LLM Training"); the same learning rate and decoupled
weight decay as AdamW then apply.

Newton-Schulz runs in fp16 on CUDA (V100 tensor cores; no bf16 there) and fp32
on CPU. After normalization every entry of X, X X^T and its square is bounded by
1, so fp16 range is safe and its 10-bit mantissa beats bf16's. For the 200M
model that is ~3.2 TFLOP per update: ~0.25 s in V100 fp32, a few tens of ms in
fp16. Under DDP every rank holds identical all-reduced gradients and computes
the same update redundantly.
"""
from __future__ import annotations

import math

import torch
from torch import nn

OPTIMIZERS = ("adamw", "muon")
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7,
                  dtype: torch.dtype | None = None) -> torch.Tensor:
    """Approximately orthogonalize a matrix: U V^T of its SVD, singular values ~[0.7, 1.2]."""
    if g.dim() != 2:
        raise ValueError("Newton-Schulz expects a matrix")
    a, b, c = NS_COEFFICIENTS
    dtype = dtype or (torch.float16 if g.is_cuda else torch.float32)
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = (x / (x.norm() + eps)).to(dtype)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if transposed else x


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, groups, lr: float, weight_decay: float = 0.0, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, betas=(0.9, 0.95), eps: float = 1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, betas=betas, eps=eps, use_muon=False)
        super().__init__(groups, defaults)
        for group in self.param_groups:
            if group["use_muon"] and any(p.dim() != 2 for p in group["params"]):
                raise ValueError("Muon groups may only contain 2D matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if group["use_muon"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(grad)
                    update = grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    update = newton_schulz(update, group["ns_steps"]).to(p.dtype)
                    update.mul_(0.2 * math.sqrt(max(p.shape)))
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
                else:
                    if not state:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    beta1, beta2 = group["betas"]
                    state["step"] += 1
                    t = float(state["step"])
                    state["exp_avg"].lerp_(grad, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (state["exp_avg_sq"] / (1 - beta2 ** t)).sqrt_().add_(group["eps"])
                    p.mul_(1 - lr * wd)
                    p.addcdiv_(state["exp_avg"], denom, value=-lr / (1 - beta1 ** t))
        return loss


def muon_parameters(model: nn.Module):
    """Hidden 2D matrices inside transformer blocks use Muon; embeddings (tied
    with the output head), norms and biases use AdamW."""
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (muon if name.startswith("layers.") and p.dim() == 2 else adam).append(p)
    return muon, adam


def build_optimizer(model: nn.Module, config, settings: dict | None):
    """settings = pretraining.optimizer; None or {"name": "adamw"} keeps fused AdamW."""
    settings = dict(settings or {"name": "adamw"})
    name = settings.pop("name", "adamw")
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    if name == "adamw":
        if settings:
            raise ValueError("adamw takes its hyperparameters from the training section")
        device = next(model.parameters()).device
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
            betas=(config.beta1, config.beta2), eps=config.adam_eps,
            weight_decay=config.weight_decay, fused=device.type == "cuda")
    allowed = {"momentum", "nesterov", "ns_steps"}
    if set(settings) - allowed:
        raise ValueError(f"unknown muon settings: {sorted(set(settings) - allowed)}")
    momentum = float(settings.get("momentum", 0.95))
    ns_steps = int(settings.get("ns_steps", 5))
    if not 0 <= momentum < 1 or ns_steps < 1:
        raise ValueError("invalid muon momentum or ns_steps")
    muon, adam = muon_parameters(model)
    return MuonAdamW([{"params": muon, "use_muon": True}, {"params": adam, "use_muon": False}],
                     lr=config.learning_rate, weight_decay=config.weight_decay, momentum=momentum,
                     nesterov=bool(settings.get("nesterov", True)), ns_steps=ns_steps,
                     betas=(config.beta1, config.beta2), eps=config.adam_eps)
