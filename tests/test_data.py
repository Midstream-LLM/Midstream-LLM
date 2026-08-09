from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jishui.data import (
    PackedBatchIterator,
    PretrainIndex,
    SequentialBatchIterator,
)


class DataIteratorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        (root / "tokens").mkdir()
        np.save(root / "tokens" / "shard_00000.npy", np.asarray([10, 11, 20, 21, 22], dtype=np.uint16))
        self.index = PretrainIndex(
            data_dir=root,
            shards=np.asarray([0, 0], dtype=np.uint8),
            offsets=np.asarray([0, 2], dtype=np.uint64),
            lengths=np.asarray([2, 3], dtype=np.uint32),
            categories=np.asarray([1, 1], dtype=np.uint8),
            splits=np.asarray([0, 0], dtype=np.uint8),
            sources=np.asarray([0, 0], dtype=np.uint8),
            source_names=("test",),
            metadata={},
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sequential_stream_inserts_eod(self):
        self.index.splits[:] = 1
        iterator = SequentialBatchIterator(self.index, "val", seq_len=7, batch_size=1)
        batch = next(iterator)
        np.testing.assert_array_equal(batch[0], [10, 11, 2, 20, 21, 22, 2, 10])

    def test_packed_batch_shape_and_resume(self):
        sampling = {
            "target_mix": {"1": 1.0},
            "weights_by_category": {"1": 1.0},
        }
        first = PackedBatchIterator(self.index, sampling, 7, 2, seed=7)
        next(first)
        state = first.state_dict()
        expected = next(first)
        second = PackedBatchIterator(self.index, sampling, 7, 2, seed=7)
        next(second)
        second.load_state_dict(state)
        np.testing.assert_array_equal(expected, next(second))
        self.assertEqual(expected.shape, (2, 8))
        self.assertTrue(np.all(expected >= 0))


if __name__ == "__main__":
    unittest.main()
