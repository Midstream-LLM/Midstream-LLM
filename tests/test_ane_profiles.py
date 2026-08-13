from __future__ import annotations

import unittest

from jishui.ane_profiles import (
    IDENTITY_ANE_BACKWARD_SCALES,
    JISHUI_200M_2048_BACKWARD_SCALES,
    default_ane_backward_scales,
    resolve_ane_backward_scales,
)
from jishui.config import ModelConfig, load_run_config


class AneProfileTest(unittest.TestCase):
    def test_production_shape_uses_validated_profile(self) -> None:
        config, _ = load_run_config("configs/jishui-200m-mlx.json")
        self.assertEqual(
            default_ane_backward_scales(config),
            JISHUI_200M_2048_BACKWARD_SCALES,
        )

    def test_other_shapes_keep_identity_scales(self) -> None:
        config = ModelConfig(
            name="tiny",
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            intermediate_size=16,
            max_position_embeddings=8,
            gradient_checkpointing=False,
        )
        self.assertEqual(
            default_ane_backward_scales(config), IDENTITY_ANE_BACKWARD_SCALES
        )

    def test_explicit_overrides_replace_profile_and_validate(self) -> None:
        config, _ = load_run_config("configs/jishui-200m-mlx.json")
        resolved = resolve_ane_backward_scales(
            config, {"sdpa_bwd_scale": 64.0, "qkv_bwd_scale": 1.0}
        )
        self.assertEqual(resolved["sdpa_bwd_scale"], 64.0)
        self.assertEqual(resolved["qkv_bwd_scale"], 1.0)
        self.assertEqual(resolved["ffn_bwd_scale"], 128.0)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            resolve_ane_backward_scales(config, {"sdpa_bwd_scale": 0.0})


if __name__ == "__main__":
    unittest.main()
