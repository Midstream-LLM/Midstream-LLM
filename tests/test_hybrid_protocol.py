from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from jishui.config import ModelConfig
from jishui.hybrid_protocol import (
    HYBRID_CMD_RUN,
    HYBRID_CMD_STOP,
    HYBRID_KIND_WEIGHTS,
    HYBRID_STATUS_READY,
    control_state,
    initialize_control_file,
    initialize_tensor_file,
    parameter_count,
    publish_generation,
    publish_stop,
    read_parameter_tree,
    tensor_file_bytes,
    tensor_view,
)
from jishui.model import JishuiForCausalLM


class HybridProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
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

    def test_tensor_file_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.bin"
            with path.open("w+b") as handle:
                mapping = initialize_tensor_file(handle, self.config, HYBRID_KIND_WEIGHTS)
                try:
                    self.assertEqual(len(mapping), tensor_file_bytes(self.config))
                    self.assertEqual(tensor_view(mapping, self.config).size, parameter_count(self.config))
                finally:
                    mapping.close()

    def test_control_generation_and_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "control.bin"
            with path.open("w+b") as handle:
                mapping = initialize_control_file(handle)
                try:
                    initial = control_state(mapping)
                    self.assertEqual(initial.command, 0)
                    self.assertEqual(initial.status, 0)
                    publish_generation(mapping, generation=1, record_base=7)
                    running = control_state(mapping)
                    self.assertEqual(running.command, HYBRID_CMD_RUN)
                    self.assertEqual(running.status, HYBRID_STATUS_READY)
                    self.assertEqual(running.generation, 1)
                    self.assertEqual(running.microsteps, 1)
                    self.assertEqual(running.record_base, 7)
                    publish_stop(mapping)
                    self.assertEqual(control_state(mapping).command, HYBRID_CMD_STOP)
                finally:
                    mapping.close()

    def test_read_preserves_empty_rope_subtree(self):
        mx.set_default_device(mx.cpu)
        model = JishuiForCausalLM(self.config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gradients.bin"
            with path.open("w+b") as handle:
                mapping = initialize_tensor_file(handle, self.config, 2)
                try:
                    tensor_view(mapping, self.config)[:] = 0.25
                    rebuilt = read_parameter_tree(
                        mapping, model.parameters(), self.config
                    )
                    self.assertEqual(
                        rebuilt["layers"][0]["attention"]["rope"], {}
                    )
                    self.assertEqual(
                        [name for name, _ in tree_flatten(rebuilt)],
                        [name for name, _ in tree_flatten(model.parameters())],
                    )
                finally:
                    mapping.close()

    def test_generation_must_increase(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "control.bin"
            with path.open("w+b") as handle:
                mapping = initialize_control_file(handle)
                try:
                    publish_generation(mapping, generation=1, record_base=0)
                    with self.assertRaisesRegex(ValueError, "increase"):
                        publish_generation(mapping, generation=1, record_base=1)
                finally:
                    mapping.close()


if __name__ == "__main__":
    unittest.main()
