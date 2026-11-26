"""Tests for Jishui-600M-Base (M2: decoupled GQA + fp32 RMSNorm + QK-Norm + Gated Attention)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from jishui.checkpoints_torch import check_model_config, load_checkpoint, save_checkpoint
from jishui.config import ModelConfig, TrainConfig, load_run_config
from jishui.model_torch import JishuiForCausalLM, RMSNorm, causal_lm_loss, rope
from jishui.optim_torch import build_optimizer, muon_parameters


REPO_ROOT = Path(__file__).resolve().parents[1]


def tiny_m2_config(max_len: int = 32, gradient_checkpointing: bool = False) -> ModelConfig:
    """Miniature M2 config with q_dim (4 * 8 = 32) > hidden_size (24)."""
    return ModelConfig(
        name="tiny-m2",
        vocab_size=48,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        norm_type="rmsnorm",
        norm_eps=1.0e-6,
        norm_bias=False,
        qk_norm=True,
        attn_output_gate=True,
        rope_theta=1_000_000.0,
        max_position_embeddings=max_len,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        gradient_checkpointing=gradient_checkpointing,
    )


class Jishui600MArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_600m_m2_config_and_exact_parameter_count(self):
        cfg_path = REPO_ROOT / "configs" / "jishui-600m-v3-docmask-8xv100.json"
        model_cfg, train_cfg = load_run_config(cfg_path)
        self.assertEqual(model_cfg.name, "Jishui-600M-Base")
        self.assertEqual(model_cfg.hidden_size, 1280)
        self.assertEqual(model_cfg.num_hidden_layers, 30)
        self.assertEqual(model_cfg.num_attention_heads, 16)
        self.assertEqual(model_cfg.num_key_value_heads, 4)
        self.assertEqual(model_cfg.head_dim, 128)
        self.assertEqual(model_cfg.intermediate_size, 3328)
        self.assertEqual(model_cfg.norm_type, "rmsnorm")
        self.assertTrue(model_cfg.qk_norm)
        self.assertTrue(model_cfg.attn_output_gate)
        self.assertEqual(model_cfg.rope_theta, 1_000_000.0)
        self.assertEqual(train_cfg.seq_len, 2048)

        with torch.device("meta"):
            model = JishuiForCausalLM(model_cfg)
        total = sum(p.numel() for p in model.parameters())
        emb = model.embed_tokens.weight.numel()
        self.assertEqual(emb, 32768 * 1280)
        self.assertEqual(total - emb, 658_722_560)
        self.assertEqual(total, 700_665_600)
        self.assertEqual(tuple(model.layers[0].attention.q_proj.weight.shape), (2048, 1280))
        self.assertEqual(tuple(model.layers[0].attention.gate_proj.weight.shape), (2048, 1280))
        self.assertEqual(tuple(model.layers[0].attention.k_proj.weight.shape), (512, 1280))
        self.assertEqual(tuple(model.layers[0].attention.o_proj.weight.shape), (1280, 2048))

    def test_rmsnorm_fp16_large_activation_does_not_overflow(self):
        norm = RMSNorm(16, eps=1e-6)
        # 1000^2 = 1e6 > 65504 (fp16 max); naive fp16 x.pow(2) overflows to inf -> 0.
        x = torch.full((2, 4, 16), 1000.0, dtype=torch.float16, requires_grad=True)
        y = norm(x)
        self.assertEqual(y.dtype, torch.float16)
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y.float(), torch.ones_like(y, dtype=torch.float32), rtol=1e-3, atol=1e-3)
        (y.float() * torch.arange(16, dtype=torch.float32)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(norm.weight.grad).all())

    def test_m2_attention_matches_explicit_reference_formula(self):
        cfg = tiny_m2_config()
        model = JishuiForCausalLM(cfg)
        model.initialize_pretraining()
        attn = model.layers[0].attention
        x = torch.randn(2, 6, cfg.hidden_size)

        q = attn.q_proj(x).view(2, 6, 4, 8).transpose(1, 2)
        k = attn.k_proj(x).view(2, 6, 2, 8).transpose(1, 2)
        v = attn.v_proj(x).view(2, 6, 2, 8).transpose(1, 2)
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + cfg.norm_eps) * attn.q_norm.weight
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + cfg.norm_eps) * attn.k_norm.weight
        q, k = rope(q, cfg.rope_theta), rope(k, cfg.rope_theta)
        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)
        scores = (q @ k.transpose(-1, -2)) / (8 ** 0.5)
        scores = scores.masked_fill(torch.ones(6, 6, dtype=torch.bool).triu(1), -float("inf"))
        sdpa_out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 6, 32)
        gated = sdpa_out * torch.sigmoid(attn.gate_proj(x))
        expected = attn.o_proj(gated)

        torch.testing.assert_close(attn(x), expected, rtol=1e-5, atol=1e-6)

    def test_document_mask_isolation_and_checkpointed_gradients(self):
        cfg = tiny_m2_config()
        model = JishuiForCausalLM(cfg)
        model.initialize_pretraining()
        checkpointed = JishuiForCausalLM(replace(cfg, gradient_checkpointing=True))
        checkpointed.load_state_dict(model.state_dict())

        # 1. Checkpointed vs non-checkpointed gradients match exactly
        ids = torch.randint(3, cfg.vocab_size, (2, 10))
        seg = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1],
                            [0, 0, 1, 1, 1, 2, 2, 2, 2, -1]])
        targets = ids.clone()
        targets[seg == -1] = -100
        for m in (model, checkpointed):
            m.train()
            causal_lm_loss(m, ids, targets, seg).backward()
        named_a = list(model.named_parameters())
        params_b = list(checkpointed.parameters())
        self.assertEqual(len(named_a), len(params_b))
        for (name, a), b in zip(named_a, params_b):
            self.assertIsNotNone(a.grad, name)
            torch.testing.assert_close(a.grad, b.grad, msg=name)

        # 2. Padding rows remain finite and do not affect valid prefix
        model.eval()
        with torch.no_grad():
            full_out = model(ids[:1], seg[:1])
            prefix_out = model(ids[:1, :8], seg[:1, :8])
        self.assertTrue(torch.isfinite(full_out).all())
        torch.testing.assert_close(full_out[:, :8], prefix_out, rtol=1e-5, atol=1e-5)

        # 3. Gradients from document 1 never leak into document 0
        model.zero_grad(set_to_none=True)
        model.train()
        emb = model.embed_tokens
        probe = emb(ids[:1, :8]).detach().requires_grad_(True)
        orig_forward = emb.forward
        emb.forward = lambda _: probe
        try:
            model(ids[:1, :8], seg[:1, :8])[0, 4:8].sum().backward()
        finally:
            emb.forward = orig_forward
        self.assertEqual(float(probe.grad[0, :4].abs().sum()), 0.0)
        self.assertGreater(float(probe.grad[0, 4:8].abs().sum()), 0.0)

    def test_muon_splits_gate_proj_independently_and_routes_norms_to_adamw(self):
        cfg = tiny_m2_config()
        model = JishuiForCausalLM(cfg)
        model.initialize_pretraining()
        muon, adam = muon_parameters(model)
        muon_ids = {id(p) for p in muon}
        adam_ids = {id(p) for p in adam}

        # 2 layers x 8 matrices (q, k, v, gate, o, mlp.gate, mlp.up, mlp.down) = 16
        self.assertEqual(len(muon), 16)
        for layer in model.layers:
            self.assertIn(id(layer.attention.q_proj.weight), muon_ids)
            self.assertIn(id(layer.attention.gate_proj.weight), muon_ids)
            self.assertNotEqual(id(layer.attention.q_proj.weight), id(layer.attention.gate_proj.weight))
            self.assertIn(id(layer.attention.q_norm.weight), adam_ids)
            self.assertIn(id(layer.attention.k_norm.weight), adam_ids)
            self.assertIn(id(layer.input_norm.weight), adam_ids)
            self.assertIn(id(layer.post_attention_norm.weight), adam_ids)
        self.assertIn(id(model.norm.weight), adam_ids)
        self.assertIn(id(model.embed_tokens.weight), adam_ids)

        opt = build_optimizer(model, TrainConfig(learning_rate=4e-4), {"name": "muon"})
        x = torch.randint(3, cfg.vocab_size, (2, 8))
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            causal_lm_loss(model, x[:, :-1], x[:, 1:]).backward()
            opt.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_legacy_200m_checkpoint_config_backward_compatibility(self):
        legacy_200m_cfg, _ = load_run_config(REPO_ROOT / "configs" / "jishui-200m-v3-docmask-8xv100.json")
        saved_without_new_keys = legacy_200m_cfg.to_dict()
        saved_without_new_keys.pop("qk_norm")
        saved_without_new_keys.pop("attn_output_gate")

        # Old 200M checkpoint dict must still match current 200M ModelConfig
        check_model_config(saved_without_new_keys, legacy_200m_cfg.to_dict())

        # Mismatched qk_norm or attn_output_gate must be rejected
        with self.assertRaisesRegex(ValueError, "architecture differs"):
            check_model_config(saved_without_new_keys, replace(legacy_200m_cfg, qk_norm=True).to_dict())
        with self.assertRaisesRegex(ValueError, "architecture differs"):
            check_model_config(saved_without_new_keys, replace(legacy_200m_cfg, attn_output_gate=True).to_dict())

    def test_m2_muon_checkpoint_save_and_exact_resume(self):
        cfg = tiny_m2_config()
        model = JishuiForCausalLM(cfg)
        model.initialize_pretraining()
        opt = build_optimizer(model, TrainConfig(learning_rate=4e-4, weight_decay=0.1), {"name": "muon"})
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        ids = torch.randint(3, cfg.vocab_size, (2, 9))

        def step_once(m, o):
            o.zero_grad(set_to_none=True)
            loss = causal_lm_loss(m, ids[:, :-1], ids[:, 1:])
            loss.backward()
            o.step()
            return loss.detach()

        step_once(model, opt)
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "model": cfg.to_dict(),
                "step": 1,
                "tokens_seen": 16,
                "sampler": {"batches_emitted": 1, "rng_state": np.random.default_rng(42).bit_generator.state},
            }
            ckpt = save_checkpoint(Path(tmp), state, model, opt, scaler)
            expected_loss = step_once(model, opt)

            restored = JishuiForCausalLM(cfg)
            restored_opt = build_optimizer(restored, TrainConfig(learning_rate=1e-2, weight_decay=0.1), {"name": "muon"})
            loaded_state = load_checkpoint(ckpt, restored, restored_opt, scaler)
            self.assertEqual(loaded_state, state)
            resumed_loss = step_once(restored, restored_opt)
            torch.testing.assert_close(resumed_loss, expected_loss, rtol=0, atol=0)
            for (name, p1), p2 in zip(model.named_parameters(), restored.parameters()):
                torch.testing.assert_close(p1, p2, rtol=0, atol=0, msg=name)


if __name__ == "__main__":
    unittest.main()
