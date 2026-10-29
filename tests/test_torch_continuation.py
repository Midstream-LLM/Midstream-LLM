import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from jishui.checkpoints_torch import import_mlx, load_checkpoint, save_checkpoint
from jishui.config import ModelConfig, TrainConfig
from jishui.model_torch import JishuiForCausalLM, causal_lm_loss, rope
from jishui.pretrain_torch import continuation_lr
from jishui.schedules_torch import make_continuation_schedule


def tiny_config():
    return ModelConfig(name="tiny", vocab_size=19, hidden_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=4, intermediate_size=12,
        max_position_embeddings=16, gradient_checkpointing=False)


class TorchContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)

    def test_rope_half_pairing(self):
        x = torch.randn(1, 2, 5, 4)
        expected = x.numpy().copy()
        for pos in range(5):
            for i in range(2):
                angle = pos / (10000 ** (i / 2))
                a, b = x[:, :, pos, i].numpy(), x[:, :, pos, i + 2].numpy()
                expected[:, :, pos, i] = a * np.cos(angle) - b * np.sin(angle)
                expected[:, :, pos, i + 2] = a * np.sin(angle) + b * np.cos(angle)
        np.testing.assert_allclose(rope(x, 10000).numpy(), expected, atol=2e-7, rtol=1e-6)

    def test_causality_and_checkpointed_gradients(self):
        model = JishuiForCausalLM(tiny_config())
        checkpointed = JishuiForCausalLM(replace(tiny_config(), gradient_checkpointing=True))
        checkpointed.load_state_dict(model.state_dict())
        ids = torch.randint(0, 19, (2, 7))
        changed = ids.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 19
        torch.testing.assert_close(model(ids)[:, :4], model(changed)[:, :4])
        for m in (model, checkpointed):
            causal_lm_loss(m, ids[:, :-1], ids[:, 1:]).backward()
        for a, b in zip(model.parameters(), checkpointed.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad)

    def test_attention_against_explicit_reference(self):
        attention = JishuiForCausalLM(tiny_config()).layers[0].attention
        x = torch.randn(2, 5, 8)
        q, k, v = [p(x).view(2, 5, 2, 4).transpose(1, 2)
                   for p in (attention.q_proj, attention.k_proj, attention.v_proj)]
        q, k = rope(q, 10000), rope(k, 10000)
        scores = q @ k.transpose(-1, -2) / 2
        scores = scores.masked_fill(torch.ones(5, 5, dtype=torch.bool).triu(1), -float("inf"))
        expected = attention.o_proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 5, 8))
        torch.testing.assert_close(attention(x), expected, rtol=1e-5, atol=1e-6)

    def test_mlx_import_preserves_all_weights_moments_and_step(self):
        model = JishuiForCausalLM(tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), eps=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            weights = {k: v.half() for k, v in model.state_dict().items()}
            save_file(weights, path / "model.safetensors")
            values = {"step": np.array(55200, dtype=np.uint64), "learning_rate": np.array(3.6e-5, dtype=np.float32)}
            for name, value in weights.items():
                values[name + ".m"] = np.full(value.shape, .012, dtype=np.float16)
                values[name + ".v"] = np.full(value.shape, .003, dtype=np.float16)
            np.savez(path / "optimizer.npz", **values)
            (path / "train_state.json").write_text(json.dumps({"model": tiny_config().to_dict(), "step": 55200}))
            state = import_mlx(path, model, optimizer)
            self.assertEqual(state["step"], 55200)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, weights[name].float(), rtol=0, atol=0)
                self.assertEqual(optimizer.state[parameter]["step"].item(), 55200)
                for old, new in (("m", "exp_avg"), ("v", "exp_avg_sq")):
                    torch.testing.assert_close(optimizer.state[parameter][new], torch.from_numpy(values[name + "." + old]).float(), rtol=0, atol=0)
            # Compare one imported AdamW update against the MLX documented fp32
            # formula. fp16 rounding from the old runtime is intentionally gone.
            p = next(model.parameters())
            before = p.detach().clone()
            m, v = optimizer.state[p]["exp_avg"].clone(), optimizer.state[p]["exp_avg_sq"].clone()
            p.grad = torch.full_like(p, .04)
            lr = optimizer.param_groups[0]["lr"]
            expected = before * (1 - lr * .01) - lr * (.9 * m + .1 * p.grad) / ((.95 * v + .05 * p.grad.square()).sqrt() + 1e-5)
            optimizer.step()
            torch.testing.assert_close(p, expected)
            del values[next(k for k in values if k.endswith('.m'))]
            np.savez(path / "optimizer.npz", **values)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                import_mlx(path, model, optimizer)

    def test_complete_checkpoint_resumes_identical_next_update(self):
        config = tiny_config()
        model = JishuiForCausalLM(config)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        ids = torch.randint(0, 19, (2, 7))
        def update(m, o):
            o.zero_grad()
            loss = causal_lm_loss(m, ids[:, :-1], ids[:, 1:])
            loss.backward()
            o.step()
            return loss.detach()
        update(model, opt)
        with tempfile.TemporaryDirectory() as directory:
            state = {"model": config.to_dict(), "step": 1, "tokens_seen": 12,
                     "sampler": {"batches_emitted": 1, "rng_state": np.random.default_rng(42).bit_generator.state}}
            path = save_checkpoint(Path(directory), state, model, opt, scaler)
            expected = update(model, opt)
            restored = JishuiForCausalLM(config)
            other_opt = torch.optim.AdamW(restored.parameters(), lr=.5)
            loaded = load_checkpoint(path, restored, other_opt, scaler)
            self.assertEqual(loaded, state)
            torch.testing.assert_close(update(restored, other_opt), expected, rtol=0, atol=0)
            for a, b in zip(model.parameters(), restored.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_continuation_schedule_no_lr_jump(self):
        schedule = dict(start_step=55200, end_step=244141, start_lr=3.6e-5, end_lr=3e-5)
        self.assertAlmostEqual(continuation_lr(schedule, 55200), 3.6e-5)
        self.assertAlmostEqual(continuation_lr(schedule, 244141), 3e-5)
        rates = [continuation_lr(schedule, i) for i in range(55200, 244142, 1000)]
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))

    def test_restarts_and_final_update_hit_exact_endpoints(self):
        config = TrainConfig(learning_rate=1e-4, min_learning_rate=1e-5, warmup_steps=1000)
        schedule = make_continuation_schedule({"schedule": "cosine_warm_restarts", "cycles": 3},
                                              config, 55200, 244141, 3.60865e-5)
        self.assertEqual(schedule["cycle_boundaries"], [55200, 118181, 181161, 244141])
        self.assertAlmostEqual(continuation_lr(schedule, 55200), 3.60865e-5)
        self.assertAlmostEqual(continuation_lr(schedule, 56199), 1e-4)
        for before_restart in (118180, 181160, 244140):
            self.assertAlmostEqual(continuation_lr(schedule, before_restart), 1e-5)
        for restart in (118181, 181161):
            self.assertAlmostEqual(continuation_lr(schedule, restart), 1e-4)
        for i, (start, end) in enumerate(zip(schedule["cycle_boundaries"], schedule["cycle_boundaries"][1:])):
            rates = [continuation_lr(schedule, s) for s in range(start + (1000 if i == 0 else 0), end)]
            self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        # Serializing the absolute-step schedule retains phase at warmup, decay,
        # restart boundaries and end, without an extra or missed optimizer step.
        restored = json.loads(json.dumps(schedule))
        for step in (55200, 55600, 80000, 118180, 118181, 181161, 244140, 244141):
            self.assertEqual(continuation_lr(restored, step), continuation_lr(schedule, step))

    def test_restart_schedule_matches_sgdr_formula(self):
        config = TrainConfig(learning_rate=1e-4, min_learning_rate=1e-5, warmup_steps=0)
        schedule = make_continuation_schedule({"schedule": "cosine_warm_restarts", "cycles": 3},
                                              config, 10, 310, 3.6e-5)
        p = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([p], lr=1e-4)
        reference = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-5)
        for cycle in range(3):
            for offset in (0, 1, 25, 50, 98):
                # Map discrete update endpoints to SGDR's continuous phase.
                reference.step(cycle * 100 + offset * 100 / 99)
                self.assertAlmostEqual(continuation_lr(schedule, 10 + cycle * 100 + offset), reference.get_last_lr()[0])

    def test_invalid_restart_schedule_fails_before_training(self):
        config = TrainConfig(learning_rate=1e-4, min_learning_rate=1e-5, warmup_steps=1000)
        for cycles in (0, -1, 1.5, 100):
            with self.assertRaises(ValueError):
                make_continuation_schedule({"schedule": "cosine_warm_restarts", "cycles": cycles}, config, 10, 3010, 3.6e-5)


if __name__ == "__main__":
    unittest.main()
