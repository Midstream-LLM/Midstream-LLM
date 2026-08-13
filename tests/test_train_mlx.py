from __future__ import annotations

import unittest
from dataclasses import replace

import mlx.core as mx
from mlx.utils import tree_flatten

from jishui.config import ModelConfig, TrainConfig
from jishui.model import JishuiForCausalLM, causal_lm_loss
from jishui.train_mlx import make_loss_and_grad, optimizer_gradients


class LossScalingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_config = ModelConfig(
            name="loss-scale-test",
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=32,
            max_position_embeddings=8,
            gradient_checkpointing=False,
        )
        self.train_config = TrainConfig(
            dtype="float16",
            loss_scale=128.0,
            seq_len=8,
            batch_size=1,
            grad_accum_steps=1,
            max_steps=1,
            compile_step=False,
        )
        mx.random.seed(7)
        self.model = JishuiForCausalLM(self.model_config)
        self.model.set_dtype(mx.float16)
        mx.eval(self.model.parameters())
        self.inputs = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        self.targets = mx.array([[2, 3, 4, 5, 6, 7, 8, 9]])

    def test_loss_is_unscaled_and_gradients_accumulate_in_float32(self) -> None:
        loss_and_grad = make_loss_and_grad(self.model, self.train_config)
        loss, gradients = loss_and_grad(self.inputs, self.targets)
        expected_loss = causal_lm_loss(self.model, self.inputs, self.targets)
        mx.eval(loss, expected_loss, gradients)

        self.assertAlmostEqual(
            float(loss.item()), float(expected_loss.item()), delta=1.0e-4
        )
        self.assertTrue(
            all(value.dtype == mx.float32 for _, value in tree_flatten(gradients))
        )

        optimizer_values = optimizer_gradients(self.model, gradients)
        for (_, gradient), (_, parameter) in zip(
            tree_flatten(optimizer_values),
            tree_flatten(self.model.trainable_parameters()),
            strict=True,
        ):
            self.assertEqual(gradient.dtype, parameter.dtype)

    def test_compiled_loss_scaling_matches_uncompiled(self) -> None:
        plain = make_loss_and_grad(self.model, self.train_config)
        compiled = make_loss_and_grad(
            self.model, replace(self.train_config, compile_step=True)
        )
        plain_loss, plain_gradients = plain(self.inputs, self.targets)
        compiled_loss, compiled_gradients = compiled(self.inputs, self.targets)
        mx.eval(plain_loss, plain_gradients, compiled_loss, compiled_gradients)

        self.assertAlmostEqual(
            float(plain_loss.item()), float(compiled_loss.item()), places=5
        )
        for (_, plain_value), (_, compiled_value) in zip(
            tree_flatten(plain_gradients),
            tree_flatten(compiled_gradients),
            strict=True,
        ):
            self.assertTrue(bool(mx.all(plain_value == compiled_value).item()))

    def test_loss_scale_must_be_positive_and_finite(self) -> None:
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.train_config, loss_scale=value)


if __name__ == "__main__":
    unittest.main()
