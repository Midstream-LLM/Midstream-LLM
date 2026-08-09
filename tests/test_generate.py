from __future__ import annotations

import unittest

import numpy as np

from jishui.generate_mlx import sample_next_token


class SamplingTest(unittest.TestCase):
    def test_greedy_respects_mask_and_repetition_penalty(self):
        logits = np.asarray([1.0, 3.0, 2.0])
        token = sample_next_token(
            logits,
            np.random.default_rng(1),
            temperature=0,
            top_p=1.0,
            top_k=0,
            history=[1],
            repetition_penalty=2.0,
            repetition_window=1,
            blocked_token_ids=[0],
        )
        self.assertEqual(token, 2)

    def test_top_k_one_is_deterministic(self):
        for seed in range(10):
            token = sample_next_token(
                np.asarray([0.0, 1.0, 4.0, 2.0]),
                np.random.default_rng(seed),
                temperature=1.0,
                top_p=1.0,
                top_k=1,
            )
            self.assertEqual(token, 2)

    def test_invalid_sampling_arguments_fail(self):
        with self.assertRaises(ValueError):
            sample_next_token(
                np.asarray([0.0, 1.0]),
                np.random.default_rng(1),
                temperature=1.0,
                top_p=0.0,
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()
