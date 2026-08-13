from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from jishui.ane_reference import (
    NativeBackwardScales,
    NativeConstants,
    _half_matmul,
    _rope_backward,
    _sdpa_backward,
    embedding_lookup_backward,
    layer_norm_backward,
    native_classifier_backward,
    native_layer_backward,
    native_layer_forward,
)
from jishui.config import ModelConfig
from jishui.model import JishuiForCausalLM


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        name="tiny-ane-reference",
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        max_position_embeddings=4,
        gradient_checkpointing=False,
    )


class AneReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self) -> None:
        mx.set_default_device(self.previous_device)

    def test_layer_norm_backward_matches_autodiff(self) -> None:
        rng = np.random.default_rng(1)
        value = mx.array(rng.normal(0.0, 0.3, (4, 8)).astype(np.float32))
        weight = mx.array(rng.normal(1.0, 0.1, (8,)).astype(np.float32))
        bias = mx.array(rng.normal(0.0, 0.1, (8,)).astype(np.float32))
        upstream = mx.array(rng.normal(0.0, 0.2, (4, 8)).astype(np.float32))
        eps = 1.0e-5

        def objective(x, gamma, beta):
            mean = mx.mean(x, axis=-1, keepdims=True)
            centered = x - mean
            normalized = centered * mx.rsqrt(
                mx.mean(centered * centered, axis=-1, keepdims=True) + eps
            )
            return mx.sum((normalized * gamma + beta) * upstream)

        _, expected = mx.value_and_grad(objective, argnums=(0, 1, 2))(
            value, weight, bias
        )
        actual = layer_norm_backward(upstream, value, weight, eps)
        mx.eval(expected, actual)
        for want, got in zip(expected, actual, strict=True):
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(want), rtol=2.0e-5, atol=2.0e-6
            )

    def test_rope_backward_is_the_transpose_rotation(self) -> None:
        config = _tiny_config()
        constants = NativeConstants.from_config(config)
        rng = np.random.default_rng(2)
        value = rng.normal(0.0, 0.2, (4, 2, 4)).astype(np.float32)
        half = config.head_dim // 2
        positions = np.arange(4, dtype=np.float32)[:, None]
        indexes = np.arange(half, dtype=np.float32)[None, :]
        frequencies = 1.0 / np.power(
            np.float32(config.rope_theta),
            np.float32(2.0) * indexes / np.float32(config.head_dim),
        )
        angles = positions * frequencies
        cosine = np.cos(angles)[:, None, :]
        sine = np.sin(angles)[:, None, :]
        first = value[..., :half]
        second = value[..., half:]
        rotated = np.concatenate(
            (first * cosine - second * sine, first * sine + second * cosine),
            axis=-1,
        ).reshape(4, 8)

        restored = _rope_backward(mx.array(rotated), config, constants)
        mx.eval(restored)
        np.testing.assert_allclose(
            np.asarray(restored), value.reshape(4, 8), rtol=2.0e-6, atol=2.0e-6
        )

    def test_sdpa_backward_matches_explicit_fp16_autodiff(self) -> None:
        config = _tiny_config()
        constants = NativeConstants.from_config(config)
        rng = np.random.default_rng(3)
        q = mx.array(rng.normal(0.0, 0.2, (4, 8)).astype(np.float16)).astype(
            mx.float32
        )
        k = mx.array(rng.normal(0.0, 0.2, (4, 8)).astype(np.float16)).astype(
            mx.float32
        )
        v = mx.array(rng.normal(0.0, 0.2, (4, 8)).astype(np.float16)).astype(
            mx.float32
        )
        upstream = mx.array(rng.normal(0.0, 0.2, (4, 8)).astype(np.float32))

        def objective(q_value, k_value, v_value):
            q_heads = q_value.reshape(4, 2, 4).transpose(1, 0, 2).astype(mx.float16)
            k_heads = k_value.reshape(4, 2, 4).transpose(1, 0, 2).astype(mx.float16)
            v_heads = v_value.reshape(4, 2, 4).transpose(1, 0, 2).astype(mx.float16)
            scores = _half_matmul(q_heads, k_heads.transpose(0, 2, 1))
            scores = (scores * mx.array(0.5, dtype=mx.float16)).astype(mx.float16)
            probabilities = mx.softmax(
                (scores[None, :, :, :] + constants.causal_mask).astype(mx.float16),
                axis=-1,
            ).astype(mx.float16)[0]
            output = _half_matmul(probabilities, v_heads)
            output = output.transpose(1, 0, 2).reshape(4, 8)
            return mx.sum(output.astype(mx.float32) * upstream)

        _, expected = mx.value_and_grad(objective, argnums=(0, 1, 2))(q, k, v)
        actual = _sdpa_backward(q, k, v, upstream, config, constants, scale=1.0)
        mx.eval(expected, actual)
        for want, got in zip(expected, actual, strict=True):
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(want), rtol=1.0e-3, atol=1.0e-5
            )

    def test_classifier_and_lookup_cover_both_tied_embedding_paths(self) -> None:
        config = _tiny_config()
        mx.random.seed(4)
        model = JishuiForCausalLM(config)
        model.set_dtype(mx.float16)
        mx.eval(model.parameters())
        rng = np.random.default_rng(4)
        residual = mx.array(rng.normal(0.0, 0.2, (4, 8)).astype(np.float32))
        targets = mx.array(np.asarray([1, 3, 5, 7], dtype=np.int32))
        loss_scale = 16.0
        actual = native_classifier_backward(
            model, residual, targets, config, loss_scale
        )
        embedding = model.embed_tokens.weight.astype(mx.float32)
        norm_weight = model.norm.weight.astype(mx.float32)
        norm_bias = model.norm.bias.astype(mx.float32)

        def objective(value, tied_embedding, gamma, beta):
            mean = mx.mean(value, axis=-1, keepdims=True)
            centered = value - mean
            normalized = centered * mx.rsqrt(
                mx.mean(centered * centered, axis=-1, keepdims=True)
                + config.norm_eps
            )
            logits = (normalized * gamma + beta) @ tied_embedding.T
            probabilities = mx.softmax(logits, axis=-1).astype(mx.float32)
            selected = mx.take_along_axis(probabilities, targets[:, None], axis=-1)
            return -mx.mean(mx.log(selected + 1.0e-10)) * loss_scale

        scaled_loss, expected = mx.value_and_grad(
            objective, argnums=(0, 1, 2, 3)
        )(residual, embedding, norm_weight, norm_bias)
        mx.eval(scaled_loss, expected, actual[1:])
        self.assertAlmostEqual(actual[0], float(scaled_loss.item()) / loss_scale, places=6)
        for want, got in zip(expected, actual[1:], strict=True):
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(want), rtol=2.0e-5, atol=5.0e-6
            )

        input_ids = mx.array(np.asarray([1, 2, 1, 3], dtype=np.int32))
        lookup_upstream = rng.normal(0.0, 0.2, (4, 8)).astype(np.float32)
        lookup = embedding_lookup_backward(
            input_ids, mx.array(lookup_upstream), tuple(embedding.shape)
        )
        expected_lookup = np.zeros(tuple(embedding.shape), dtype=np.float32)
        np.add.at(expected_lookup, np.asarray(input_ids), lookup_upstream)
        mx.eval(lookup)
        np.testing.assert_array_equal(np.asarray(lookup), expected_lookup)

    def test_layerwise_vjp_produces_all_parameter_gradients(self) -> None:
        config = _tiny_config()
        constants = NativeConstants.from_config(config)
        scales = NativeBackwardScales(
            loss=16.0,
            ffn_w2=8.0,
            ffn_w13=4.0,
            wo=8.0,
            sdpa=16.0,
            qkv=8.0,
        )
        mx.random.seed(5)
        model = JishuiForCausalLM(config)
        model.set_dtype(mx.float16)
        mx.eval(model.parameters())
        rng = np.random.default_rng(5)
        value = mx.array(rng.normal(0.0, 0.2, (4, 8)).astype(np.float32))
        upstream = mx.array(rng.normal(0.0, 0.2, (4, 8)).astype(np.float32))

        next_value, cache = native_layer_forward(
            model.layers[0], value, config, constants
        )
        next_upstream, gradients = native_layer_backward(
            model.layers[0], cache, upstream, config, constants, scales
        )
        mx.eval(next_value, next_upstream, gradients)
        parameters = dict(tree_flatten(model.layers[0].parameters()))
        self.assertEqual(set(gradients), set(parameters))
        self.assertEqual(tuple(next_value.shape), tuple(value.shape))
        self.assertEqual(tuple(next_upstream.shape), tuple(value.shape))
        self.assertTrue(np.isfinite(np.asarray(next_value)).all())
        self.assertTrue(np.isfinite(np.asarray(next_upstream)).all())
        for name, gradient in gradients.items():
            self.assertEqual(tuple(gradient.shape), tuple(parameters[name].shape))
            self.assertTrue(np.isfinite(np.asarray(gradient)).all(), name)


if __name__ == "__main__":
    unittest.main()
