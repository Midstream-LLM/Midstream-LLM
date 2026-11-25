"""Document-masked attention and multi-process (DDP) training equivalence."""
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp

from jishui.block_data import DocumentStream, SampleSpec, make_batch
from jishui.config import ModelConfig
from jishui.model_torch import DocumentMask, JishuiForCausalLM, attention_backend
from jishui.pretrain_torch import rows, segment_ids

EOD = 2


def tiny(max_len=64):
    torch.manual_seed(0)
    model = JishuiForCausalLM(ModelConfig(name="tiny", vocab_size=40, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, intermediate_size=24,
        max_position_embeddings=max_len, gradient_checkpointing=False))
    model.initialize_pretraining()
    return model.eval()


class Index:
    """Minimal stand-in for PretrainIndex as read by DocumentStream."""

    def __init__(self, docs):
        self.lengths = np.asarray([len(d) for d in docs], dtype=np.int64)
        self.offsets = np.cumsum(self.lengths) - self.lengths
        self.shards = np.zeros(len(docs), dtype=np.int64)
        self.token_shards = [np.concatenate([np.asarray(d, dtype=np.int64) for d in docs])]


class DocumentAttentionTests(unittest.TestCase):
    def packed(self, docs, length):
        index = Index(docs)
        stream = DocumentStream(index, np.arange(len(docs)), EOD)
        return make_batch([stream.sample(0, length, SampleSpec(length))], SampleSpec(length))

    def test_packed_documents_match_isolated_forward(self):
        model = tiny()
        docs = [[5, 6, 7, 8, 9], [10, 11, 12], [13, 14, 15, 16, 17, 18]]
        batch = self.packed(docs, 16)
        x = torch.as_tensor(batch.input_ids)
        seg = torch.as_tensor(np.ascontiguousarray(segment_ids(batch)))
        with torch.no_grad():
            packed = model(x, seg)[0]
            leaky = model(x)[0]
        # The stream is [EOD] A EOD B EOD C ...; each document's reference is
        # its own fresh forward from the preceding EOD, which acts as BOS.
        start, differs = 0, False
        for doc in docs:
            alone = torch.as_tensor([[EOD, *doc]])
            with torch.no_grad():
                reference = model(alone)[0]
            span = slice(start, start+len(doc)+1)
            n = min(len(doc)+1, x.shape[1]-start)
            torch.testing.assert_close(packed[span][:n], reference[:n], rtol=1e-5, atol=1e-5)
            differs |= not torch.allclose(leaky[span][:n], reference[:n], atol=1e-4)
            start += len(doc)+1
        self.assertTrue(differs, "fixture should show leakage without the document mask")

    def test_segments_turn_eod_into_next_document_bos(self):
        batch = self.packed([[5, 6], [7, 8, 9]], 6)
        np.testing.assert_array_equal(batch.input_ids[0], [EOD, 5, 6, EOD, 7, 8])
        np.testing.assert_array_equal(segment_ids(batch)[0], [0, 0, 0, 1, 1, 1])

    def test_single_document_equals_plain_causal(self):
        model = tiny()
        x = torch.randint(3, 40, (2, 12))
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(x, torch.zeros_like(x)), rtol=0, atol=1e-6)

    def test_padding_rows_stay_finite_and_unseen(self):
        model = tiny()
        x = torch.randint(3, 40, (1, 10))
        seg = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1, -1, -1]])
        with torch.no_grad():
            out = model(x, seg)
            ref = model(x[:, :6], seg[:, :6])
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out[:, :6], ref, rtol=1e-5, atol=1e-5)

    def test_gradients_do_not_cross_documents(self):
        model = tiny().train()
        emb = model.embed_tokens
        x = torch.randint(3, 40, (1, 8))
        seg = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        probe = emb(x).detach().requires_grad_(True)
        original = emb.forward
        emb.forward = lambda _: probe
        try:
            model(x, seg)[0, 4:].sum().backward()
        finally:
            emb.forward = original
        self.assertEqual(float(probe.grad[0, :4].abs().sum()), 0.0)
        self.assertGreater(float(probe.grad[0, 4:].abs().sum()), 0.0)

    def test_flex_matches_dense(self):
        q, k, v = (torch.randn(2, 2, 128, 8) for _ in range(3))
        seg = torch.as_tensor(np.repeat([[0, 1, 2, 3], [0, 0, 1, 1]], 32, axis=1))
        dense = DocumentMask(seg, "dense").attend(q, k, v)
        try:
            flex = DocumentMask(seg, "flex").attend(q, k, v)
        except Exception as error:  # CPU flex support depends on the torch build
            self.skipTest(f"flex attention unavailable here: {error}")
        torch.testing.assert_close(flex, dense, rtol=1e-4, atol=1e-4)

    def test_backend_selection(self):
        cpu = torch.device("cpu")
        self.assertEqual(attention_backend(2048, cpu, "auto"), "dense")
        self.assertEqual(attention_backend(32768, cpu, "auto"), "flex")
        with self.assertRaisesRegex(ValueError, "dense document mask"):
            attention_backend(32768, cpu, "dense")

    def test_rank_slices_partition_the_global_batch(self):
        batch = self.packed([[5, 6, 7], [8, 9]], 4)
        double = make_batch([(batch.tokens[0], batch.document_ids[0], 4, 0),
                             (batch.tokens[0]+1, batch.document_ids[0], 3, 4)], batch.spec)
        a, b = rows(double, 0, 2), rows(double, 1, 2)
        np.testing.assert_array_equal(np.concatenate([a.tokens, b.tokens]), double.tokens)
        self.assertEqual(int(a.targets(1)[1].sum() + b.targets(1)[1].sum()), int(double.targets(1)[1].sum()))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, size, port, kwargs):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(size),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    import torch.distributed as dist
    from jishui.pretrain_torch import init_distributed, run
    from jishui.tracking import ExperimentTracker
    init_distributed("cpu")
    try:
        args = SimpleNamespace(**kwargs)
        run(args, ExperimentTracker(args.run_dir, None, mode="disabled"))
    finally:
        dist.destroy_process_group()


class DistributedTrainingTests(unittest.TestCase):
    """Two gloo ranks x 1 row must equal one process x 2 rows, exactly in math."""

    def setUp(self):
        from tests.test_corpus_pretraining import CorpusPretrainingTests
        self.fixture = CorpusPretrainingTests("test_block_training_exact_resume_with_partial_blocks")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.data, _ = self.fixture.prepare()

    def config(self, batch_size, attention):
        spec = json.loads(self.fixture.config().read_text())
        # Validation budget is eval_batches * per-rank batch_size; keep it equal.
        spec["training"].update(batch_size=batch_size, grad_accum_steps=2, eval_batches=2 // batch_size)
        spec["pretraining"].update(sampling="shuffled_blocks", prediction_offsets=[1], attention=attention,
                                   validation={"mode": "source_stratified_blocks", "seed": 71},
                                   target_tokens=64)
        path = self.root / f"config-{batch_size}-{attention}.json"
        path.write_text(json.dumps(spec))
        return path

    def args(self, config, name, **extra):
        return dict(config=config, run_dir=self.root / name, data_dir=self.data, index_cache=None,
                    resume=None, init_from=None, device="cpu", steps=None, evaluate_first=True,
                    validate_only=False, expected_val_loss=None, val_tolerance=.03, migrate_from=None, **extra)

    def final(self, name):
        latest = json.loads((self.root / name / "latest.json").read_text())["checkpoint"]
        return torch.load(Path(latest) / "training.pt", weights_only=True)

    def run_pair(self, attention):
        from jishui.pretrain_torch import run
        from jishui.tracking import ExperimentTracker
        single = self.args(self.config(2, attention), f"single-{attention}")
        (self.root / f"single-{attention}").mkdir()
        run(SimpleNamespace(**single), ExperimentTracker(single["run_dir"], None, mode="disabled"))
        ddp = self.args(self.config(1, attention), f"ddp-{attention}")
        mp.spawn(_worker, args=(2, free_port(), ddp), nprocs=2, join=True)
        return self.final(f"single-{attention}"), self.final(f"ddp-{attention}")

    def test_two_rank_training_matches_single_process(self):
        for attention in ("causal", "document"):
            with self.subTest(attention=attention):
                single, ddp = self.run_pair(attention)
                for key in single["model"]:
                    torch.testing.assert_close(ddp["model"][key], single["model"][key], rtol=1e-5, atol=1e-6)
                for key in ("step", "tokens_seen", "sampler", "evaluation"):
                    self.assertEqual(ddp["state"][key], single["state"][key])
                self.assertEqual(ddp["state"]["distributed"]["world_size"], 2)
                self.assertEqual(single["state"]["pretraining"]["attention"], attention)

    def test_document_attention_changes_training(self):
        _, causal = self.run_pair("causal")
        _, document = self.run_pair("document")
        self.assertTrue(any(not torch.equal(causal["model"][k], document["model"][k]) for k in causal["model"]))


if __name__ == "__main__":
    unittest.main()
