"""DeepSeek-V3-style multi-token prediction: model, targets, training, checkpoints."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp

from jishui.block_data import DocumentStream, SampleSpec, make_batch
from jishui.checkpoints_torch import strip_mtp
from jishui.config import ModelConfig
from jishui.model_torch import JishuiForCausalLM, mtp_lm_loss
from jishui.pretrain_torch import mtp_tensors
from tests.test_document_attention import EOD, DistributedTrainingTests, Index, _worker, free_port


def tiny_mtp(depth=1):
    torch.manual_seed(0)
    model = JishuiForCausalLM(ModelConfig(name="tiny", vocab_size=40, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, intermediate_size=24,
        max_position_embeddings=64, gradient_checkpointing=False, mtp_depth=depth))
    model.initialize_pretraining()
    return model.eval()


class MTPModelTests(unittest.TestCase):
    def test_config_round_trip_keeps_old_layout(self):
        plain = tiny_mtp(0).config.to_dict()
        self.assertNotIn("mtp_depth", plain)
        self.assertEqual(tiny_mtp(2).config.to_dict()["mtp_depth"], 2)
        self.assertEqual(ModelConfig.from_dict(tiny_mtp(1).config.to_dict()).mtp_depth, 1)
        with self.assertRaises(ValueError):
            ModelConfig.from_dict(dict(plain, mtp_depth=-1))

    def test_main_logits_unchanged_and_inference_ignores_mtp(self):
        model = tiny_mtp(2)
        x = torch.randint(3, 40, (2, 10))
        future = [torch.randint(3, 40, (2, 10)) for _ in range(2)]
        with torch.no_grad():
            logits, extra = model(x, None, future)
            torch.testing.assert_close(logits, model(x), rtol=0, atol=0)
        self.assertEqual(len(extra), 2)
        self.assertEqual(extra[0].shape, logits.shape)
        self.assertEqual(sum(k.startswith("mtp.") for k in model.state_dict()) > 0, True)

    def test_mtp_is_causal(self):
        model = tiny_mtp(1)
        x = torch.randint(3, 40, (1, 12))
        future = torch.randint(3, 40, (1, 12))
        changed_x, changed_f = x.clone(), future.clone()
        changed_x[0, 7:] = 5
        changed_f[0, 7:] = 6
        with torch.no_grad():
            _, a = model(x, None, [future])
            _, b = model(changed_x, None, [changed_f])
        torch.testing.assert_close(a[0][:, :7], b[0][:, :7], rtol=1e-6, atol=1e-6)

    def test_future_inputs_and_targets_align_and_respect_documents(self):
        docs = [[5, 6, 7, 8], [10, 11, 12]]
        spec = SampleSpec(8, (1, 2))
        stream = DocumentStream(Index(docs), np.arange(2), EOD)
        batch = make_batch([stream.sample(0, 8, spec)], spec)
        # Stream: [EOD] 5 6 7 8 EOD 10 11 12 EOD
        (future,), (target,) = mtp_tensors(batch, "cpu", 1)
        np.testing.assert_array_equal(batch.input_ids[0], [EOD, 5, 6, 7, 8, EOD, 10, 11])
        np.testing.assert_array_equal(future[0].numpy(), [5, 6, 7, 8, EOD, 10, 11, 12])
        # Target i+2 must share the input position's document (block_data rule;
        # EOD belongs to the document it ends), so both boundary slots are masked.
        np.testing.assert_array_equal(target[0].numpy(), [6, 7, 8, EOD, -100, -100, 12, EOD])

    def test_mtp_loss_trains_main_model_and_handles_empty_depth(self):
        model = tiny_mtp(1).train()
        x = torch.randint(3, 40, (2, 8))
        y = torch.randint(3, 40, (2, 8))
        future = [torch.randint(3, 40, (2, 8))]
        empty = [torch.full((2, 8), -100)]
        main, sums = mtp_lm_loss(model, x, y, future, empty)
        self.assertEqual(float(sums[0].detach()), 0.0)
        main, sums = mtp_lm_loss(model, x, y, future, [y])
        self.assertGreater(float(sums[0].detach()), 0.0)
        (main + 0.3 * sums[0] / 16).backward()
        for name in ("mtp.0.proj.weight", "layers.0.attention.q_proj.weight", "embed_tokens.weight"):
            self.assertGreater(float(dict(model.named_parameters())[name].grad.abs().sum()), 0, name)

    def test_strip_mtp_gives_loadable_plain_model(self):
        model = tiny_mtp(1)
        weights, config = strip_mtp(model.state_dict(), model.config.to_dict())
        plain = JishuiForCausalLM(ModelConfig.from_dict(config))
        plain.load_state_dict(weights, strict=True)
        x = torch.randint(3, 40, (1, 9))
        with torch.no_grad():
            torch.testing.assert_close(plain(x), model(x), rtol=0, atol=0)


class MTPTrainingTests(DistributedTrainingTests):
    def config(self, batch_size, attention, depth=1):
        path = super().config(batch_size, attention)
        spec = json.loads(path.read_text())
        if depth:
            spec["model"]["mtp_depth"] = depth
            spec["pretraining"].update(prediction_offsets=list(range(1, depth+2)),
                                       mtp={"depth": depth, "weight": 0.3})
        path = path.with_name(path.stem + f"-mtp{depth}.json")
        path.write_text(json.dumps(spec))
        return path

    def train(self, name, config, **extra):
        from jishui.pretrain_torch import run
        from jishui.tracking import ExperimentTracker
        args = self.args(config, name)
        args.update(extra)
        args["run_dir"].mkdir(exist_ok=True)
        run(SimpleNamespace(**args), ExperimentTracker(args["run_dir"], None, mode="disabled"))
        latest = json.loads((args["run_dir"] / "latest.json").read_text())["checkpoint"]
        return Path(latest), torch.load(Path(latest) / "training.pt", weights_only=True)

    def test_two_rank_training_matches_single_process(self):
        single, ddp = self.run_pair("causal")
        self.assertTrue(any(k.startswith("mtp.") for k in single["model"]))
        for key in single["model"]:
            torch.testing.assert_close(ddp["model"][key], single["model"][key], rtol=1e-5, atol=1e-6)
        for key in ("step", "tokens_seen", "sampler"):
            self.assertEqual(ddp["state"][key], single["state"][key])

    def test_document_attention_changes_training(self):
        self.skipTest("covered by DistributedTrainingTests")

    def test_resume_validation_protocol_and_weight_transfer(self):
        mtp, plain = self.config(2, "causal"), self.config(2, "causal", depth=0)
        _, full = self.train("full", mtp)
        self.train("part", mtp, steps=1)
        _, resumed = self.train("part", mtp, resume="latest")
        for key in full["model"]:
            torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
        self.assertEqual(full["state"]["sampler"], resumed["state"]["sampler"])
        # Same NTP validation protocol as a run without MTP.
        plain_path, plain_bundle = self.train("plain", plain)
        self.assertEqual(full["state"]["evaluation"]["fingerprint"],
                         plain_bundle["state"]["evaluation"]["fingerprint"])
        # Weights-only starts: MTP -> plain drops mtp.*, plain -> MTP adds fresh modules.
        mtp_path = Path(json.loads((self.root / "full" / "latest.json").read_text())["checkpoint"])
        _, dropped = self.train("dropped", plain, steps=1, init_from=mtp_path)
        self.assertFalse(any(k.startswith("mtp.") for k in dropped["model"]))
        _, added = self.train("added", mtp, steps=1, init_from=plain_path)
        self.assertTrue(any(k.startswith("mtp.") for k in added["model"]))

    def test_inconsistent_mtp_settings_fail(self):
        spec = json.loads(self.config(2, "causal").read_text())
        broken = self.root / "broken.json"
        for change in ({"pretraining": {"prediction_offsets": [1]}},
                       {"pretraining": {"mtp": {"depth": 2, "weight": 0.3}}},
                       {"model": {"mtp_depth": 0}}):
            bad = json.loads(json.dumps(spec))
            for section, values in change.items():
                bad[section].update(values)
            broken.write_text(json.dumps(bad))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.train(f"broken-{len(list(self.root.iterdir()))}", broken)


if __name__ == "__main__":
    unittest.main()
