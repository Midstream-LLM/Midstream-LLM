import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jishui.block_data import DocumentStream, SampleSpec, ShuffledBlockIterator, FixedValidation
from jishui.data import PretrainIndex, ShuffledDocumentIterator


class BlockDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tokens").mkdir()

    def index(self, lengths=(9, 3, 17), splits=None, sources=None):
        lengths = np.asarray(lengths, dtype=np.int64)
        tokens = np.arange(10, 10+int(lengths.sum()), dtype=np.int32)
        np.save(self.root / "tokens/shard_00000.npy", tokens)
        n = len(lengths)
        return PretrainIndex(self.root, np.zeros(n, dtype=np.int32),
            np.cumsum(lengths)-lengths, lengths, np.ones(n, dtype=np.int32),
            np.zeros(n, dtype=np.int8) if splits is None else np.array(splits),
            np.zeros(n, dtype=np.int32) if sources is None else np.array(sources),
            ("a", "b", "c"), {"fixture": list(map(int, lengths))})

    def test_every_target_exactly_once_including_prefixes_eod_and_tail(self):
        index = self.index()
        sampler = ShuffledBlockIterator(index, SampleSpec(7), 1, 42)
        expected, _ = sampler.stream.read(0, sampler.stream.size)
        positions, observed, counts = [], {}, []
        for _ in range(len(sampler.order)):
            batch = next(sampler)
            target, mask = batch.targets()
            count = int(mask.sum()); counts.append(count)
            begin = int(batch.target_starts[0])
            for p, t in zip(range(begin, begin+count), target[0, mask[0]]):
                self.assertNotIn(p, observed)
                observed[p] = int(t)
            positions.extend(range(begin, begin+count))
            self.assertTrue(np.all(target[~mask] == -100))
        self.assertEqual(sorted(positions), list(range(len(expected))))
        np.testing.assert_array_equal([observed[p] for p in sorted(observed)], expected)
        self.assertIn(sampler.stream.size % 7, counts)
        next(sampler)
        self.assertEqual(sampler.epoch, 1)

    def test_long_documents_are_mixed_and_resume_is_exact_across_epochs(self):
        index = self.index((1000, 1000, 1000))
        a = ShuffledBlockIterator(index, SampleSpec(16), 3, 42, self.root / "cache")
        source_docs = np.concatenate([next(a).document_ids[:, 1] for _ in range(12)])
        self.assertEqual(len(set(source_docs.tolist())), 3)
        state = copy.deepcopy(a.state_dict())
        b = ShuffledBlockIterator(index, SampleSpec(16), 3, 42, self.root / "cache")
        b.load_state_dict(state)
        for _ in range(100):
            x, y = next(a), next(b)
            np.testing.assert_array_equal(x.tokens, y.tokens)
            np.testing.assert_array_equal(x.document_ids, y.document_ids)
            np.testing.assert_array_equal(x.target_counts, y.target_counts)
        self.assertEqual(a.state_dict(), b.state_dict())
        broken = copy.deepcopy(state)
        broken["spec"]["seq_len"] = 8
        with self.assertRaisesRegex(ValueError, "spec"):
            b.load_state_dict(broken)

    def test_migration_covers_only_unread_suffix(self):
        index = self.index((11, 5, 9))
        for emitted in (0, 1, 3, 6, 7, 8, 15):
            old = ShuffledDocumentIterator(index, 4, 1, 42)
            for _ in range(emitted): next(old)
            new = ShuffledBlockIterator(index, SampleSpec(4), 1, 42)
            new.migrate_document_stream(old.state_dict())
            saved = copy.deepcopy(new.state_dict())
            expected = new.stream.read(new.first_offset, new.stream.size-new.first_offset)[0]
            observed = {}
            for _ in range(len(new.order)):
                batch = next(new)
                targets, mask = batch.targets()
                start = int(batch.target_starts[0])
                for p, t in zip(range(start, start+int(mask.sum())), targets[0, mask[0]]):
                    self.assertNotIn(p, observed)
                    observed[p] = int(t)
            np.testing.assert_array_equal([observed[p] for p in sorted(observed)], expected)
            if emitted:
                original_next = next(old)[0, 1:]
                available = min(len(original_next), len(expected))
                np.testing.assert_array_equal(expected[:available], original_next[:available])
            restored = ShuffledBlockIterator(index, SampleSpec(4), 1, 42)
            restored.load_state_dict(saved)
            self.assertEqual(restored.state_dict(), saved)

    def test_modified_cache_is_rejected(self):
        index = self.index()
        cache = self.root / "cache"
        ShuffledBlockIterator(index, SampleSpec(4), 1, 42, cache)
        path = next(cache.glob("*.npy"))
        order = np.load(path)
        order[0] = order[1]
        np.save(path, order)
        with self.assertRaisesRegex(ValueError, "checksum"):
            ShuffledBlockIterator(index, SampleSpec(4), 1, 42, cache)

    def test_mtp_lookahead_comes_from_original_stream_and_masks_boundaries(self):
        index = self.index((8, 8, 8))
        sampler = ShuffledBlockIterator(index, SampleSpec(4, (1, 2, 4)), 1, 42)
        original, docs = sampler.stream.read(-1, sampler.stream.size+1)
        for _ in range(len(sampler.order)):
            batch = next(sampler)
            start = int(batch.target_starts[0])
            for k in (1, 2, 4):
                target, mask = batch.targets(k)
                for j in range(4):
                    p = start+j+k
                    valid = j < batch.target_counts[0] and p < len(original)
                    if k > 1:
                        valid = valid and docs[start+j] == docs[p]
                    self.assertEqual(bool(mask[0,j]), bool(valid))
                    if valid: self.assertEqual(target[0,j], original[p])
            self.assertEqual(batch.segment_offsets[0][0], 0)
            self.assertEqual(batch.segment_offsets[0][-1], batch.target_counts[0])

    def test_greater_than_128k_window_preserves_contiguous_text(self):
        index = self.index((300000,))
        length = 262144
        sampler = ShuffledBlockIterator(index, SampleSpec(length, (1, 4)), 1, 17)
        for _ in range(len(sampler.order)):
            batch = next(sampler)
            start = int(batch.target_starts[0]); count = int(batch.target_counts[0])
            self.assertEqual(batch.tokens.shape, (1, length+4))
            expected = sampler.stream.read(start, count)[0]
            np.testing.assert_array_equal(batch.targets(1)[0][0, :count], expected)
            self.assertEqual(batch.position_ids[0, -1], length-1)

    def test_validation_is_fixed_covers_all_sources_and_is_weighted(self):
        index = self.index((100, 1000, 30, 17), [1, 1, 1, 0], [0, 1, 0, 2])
        a = FixedValidation(index, SampleSpec(8), 20, 1, 72)
        b = FixedValidation(index, SampleSpec(8), 20, 1, 72)
        self.assertEqual(a.metadata, b.metadata)
        self.assertEqual(a.samples, b.samples)
        self.assertEqual({s for s, *_ in a.samples}, {0, 1})
        self.assertEqual(a.metadata["missing_training_sources"], ["c"])
        self.assertGreater(max(p for s,p,_ in a.samples if s == 1), 800)
        self.assertAlmostEqual(a.weighted_mean({0: 2., 1: 8.}), (132*2+1001*8)/1133)
        for source, batch in a:
            ids = batch.document_ids[batch.document_ids >= 0]
            self.assertTrue(np.all(index.splits[ids] == 1))
            self.assertTrue(np.all(index.sources[ids] == source))
        # A tiny source is fully evaluated once, never wrapped/repeated to fill a batch.
        small = FixedValidation(self.index((3, 500), [1, 1], [0, 1]), SampleSpec(16), 20, 1, 72)
        self.assertEqual(small.metadata["sources"]["a"]["evaluated_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
