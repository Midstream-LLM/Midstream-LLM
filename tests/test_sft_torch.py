import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from jishui.checkpoints_torch import save_checkpoint
from jishui.config import ModelConfig
from jishui.model_torch import JishuiForCausalLM
from jishui import sft_torch

TINY = dict(name="Jishui-tiny", vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, head_dim=16, intermediate_size=64, norm_type="layernorm", norm_eps=1e-5,
            norm_bias=True, rope_theta=10000, max_position_embeddings=64, tie_word_embeddings=True,
            attention_bias=False, mlp_bias=False, gradient_checkpointing=False)


def write_compact(root: Path, n=60, seed=0):
    rng = np.random.default_rng(seed)
    ids, mask, offsets = [], [], [0]
    for _ in range(n):
        prompt = [3] + rng.integers(5, 64, rng.integers(3, 12)).tolist() + [4, 3]
        answer = rng.integers(5, 64, rng.integers(2, 10)).tolist() + [4]
        ids += prompt + answer
        mask += [0] * len(prompt) + [1] * len(answer)
        offsets.append(len(ids))
    root.mkdir(parents=True)
    np.save(root / "input_ids.npy", np.asarray(ids, dtype=np.uint16))
    np.save(root / "loss_mask.npy", np.asarray(mask, dtype=np.uint8))
    np.save(root / "sample_offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(root / "train_indices.npy", np.arange(0, n - 10, dtype=np.int32))
    np.save(root / "val_indices.npy", np.arange(n - 10, n - 5, dtype=np.int32))
    np.save(root / "test_indices.npy", np.arange(n - 5, n, dtype=np.int32))
    (root / "manifest.json").write_text("{}")


def write_parent(run: Path):
    torch.manual_seed(0)
    model = JishuiForCausalLM(ModelConfig.from_dict(TINY))
    model.initialize_pretraining()
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return save_checkpoint(run, {"model": TINY, "step": 7}, model, opt, scaler)


def args(tmp, run, *extra):
    return sft_torch.parse_args(["--data-dir", str(tmp / "data"), "--run-dir", str(run), "--device", "cpu",
                                 "--micro-tokens", "64", "--grad-accum", "2", "--epochs", "2",
                                 "--eval-interval", "3", "--save-interval", "3", "--lr", "1e-3", *extra])


def weights(run: Path):
    path = sorted((run / "checkpoints").iterdir())[-1] / "training.pt"
    return torch.load(path, weights_only=True)["model"]


class SFTTest(unittest.TestCase):
    def test_plan_covers_each_sample_once_within_budget(self):
        lengths = np.random.default_rng(1).integers(5, 40, 500)
        plan = sft_torch.batch_plan(lengths, np.arange(500), 128, seed=3, epoch=1, pool=64)
        flat = [i for b in plan for i in b]
        self.assertEqual(sorted(flat), list(range(500)))
        for b in plan:
            self.assertTrue(len(b) == 1 or max(lengths[b] - 1) * len(b) <= 128)
        self.assertEqual(plan, sft_torch.batch_plan(lengths, np.arange(500), 128, seed=3, epoch=1, pool=64))

    def test_only_masked_tokens_are_targets(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            write_compact(tmp / "data")
            data = sft_torch.CompactSFT(tmp / "data")
            x, y = sft_torch.collate(data, [0, 1], torch.device("cpu"))
            for r, i in enumerate([0, 1]):
                ids, mask = data.sample(i)
                kept = y[r][y[r] != -100].tolist()
                self.assertEqual(kept, [int(t) for t, m in zip(ids[1:], mask[1:]) if m])
                self.assertEqual(kept[-1], sft_torch.IM_END)

    def test_train_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            write_compact(tmp / "data")
            parent = write_parent(tmp / "parent")
            straight = sft_torch.run(args(tmp, tmp / "a", "--init-from", str(parent)))
            self.assertTrue(straight["sft"]["complete"])
            losses = [h["val_loss"]["all"] for h in straight["sft"]["history"] if "val_loss" in h]
            self.assertLess(losses[-1], losses[0])
            sft_torch.run(args(tmp, tmp / "b", "--init-from", str(parent), "--max-steps", "4"))
            sft_torch.run(args(tmp, tmp / "b", "--resume"))
            a, b = weights(tmp / "a"), weights(tmp / "b")
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]), k)
            final = json.loads((tmp / "b" / "final.json").read_text())
            self.assertEqual(final["step"], straight["step"])


def strict_config(tmp, parent, **expect):
    recipe = {"data_dir": str(tmp / "data"), "init_from": str(parent), "tokenizer": None, "epochs": 2,
              "micro_tokens": 64, "grad_accum": 2, "lr": 1e-3, "min_lr": 5e-6, "warmup_steps": 50,
              "weight_decay": 0.01, "grad_clip": 1.0, "eval_interval": 3, "save_interval": 3,
              "keep_checkpoints": 2, "gradient_checkpointing": False, "seed": 1234, "optimizer": "adamw"}
    path = tmp / "strict.json"
    path.write_text(json.dumps({"sft": recipe, "expect": expect}))
    return path


class StrictConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)
        write_compact(self.tmp / "data")
        self.parent = write_parent(self.tmp / "parent")

    def test_config_run_equals_cli_run_and_resumes(self):
        data = sft_torch.CompactSFT(self.tmp / "data")
        config = strict_config(self.tmp, self.parent, data_fingerprint=data.fingerprint(),
                               train_samples=50, val_samples=5, test_samples=5, parent_step=7,
                               parent_model=TINY)
        via_config = sft_torch.run(sft_torch.parse_args(["--config", str(config), "--run-dir",
                                                         str(self.tmp / "c"), "--device", "cpu"]))
        sft_torch.run(args(self.tmp, self.tmp / "a", "--init-from", str(self.parent)))
        a, c = weights(self.tmp / "a"), weights(self.tmp / "c")
        for k in a:
            self.assertTrue(torch.equal(a[k], c[k]), k)
        self.assertEqual(len(via_config["sft"]["hyperparameters"]["config_sha256"]), 64)
        run = ["--config", str(config), "--run-dir", str(self.tmp / "d"), "--device", "cpu"]
        sft_torch.run(sft_torch.parse_args(run + ["--max-steps", "4"]))
        sft_torch.run(sft_torch.parse_args(run + ["--resume"]))
        d = weights(self.tmp / "d")
        for k in a:
            self.assertTrue(torch.equal(a[k], d[k]), k)

    def test_muon_sft_trains_resumes_and_differs_from_adamw(self):
        config = strict_config(self.tmp, self.parent)
        raw = json.loads(config.read_text())
        raw["sft"]["optimizer"] = "muon"
        config.write_text(json.dumps(raw))
        run = ["--config", str(config), "--device", "cpu"]
        state = sft_torch.run(sft_torch.parse_args(run + ["--run-dir", str(self.tmp / "m")]))
        losses = [h["val_loss"]["all"] for h in state["sft"]["history"] if "val_loss" in h]
        self.assertLess(losses[-1], losses[0])
        sft_torch.run(sft_torch.parse_args(run + ["--run-dir", str(self.tmp / "n"), "--max-steps", "4"]))
        sft_torch.run(sft_torch.parse_args(run + ["--run-dir", str(self.tmp / "n"), "--resume"]))
        m, n = weights(self.tmp / "m"), weights(self.tmp / "n")
        for k in m:
            self.assertTrue(torch.equal(m[k], n[k]), k)
        sft_torch.run(args(self.tmp, self.tmp / "a", "--init-from", str(self.parent)))
        a = weights(self.tmp / "a")
        self.assertTrue(any(not torch.equal(a[k], m[k]) for k in a))
        bundle = torch.load(sorted((self.tmp / "m" / "checkpoints").iterdir())[-1] / "training.pt", weights_only=True)
        self.assertTrue(any("momentum_buffer" in v for v in bundle["optimizer"]["state"].values()))

    def test_cli_cannot_override_recipe(self):
        config = strict_config(self.tmp, self.parent)
        with self.assertRaisesRegex(ValueError, "--lr is fixed"):
            sft_torch.parse_args(["--config", str(config), "--run-dir", "x", "--lr", "1e-4"])

    def test_every_recipe_key_must_be_pinned(self):
        config = strict_config(self.tmp, self.parent)
        raw = json.loads(config.read_text())
        del raw["sft"]["seed"]
        config.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "missing \\['seed'\\]"):
            sft_torch.parse_args(["--config", str(config), "--run-dir", "x"])

    def test_expectation_mismatch_stops_before_training(self):
        for expect in ({"total_steps": 999}, {"parent_step": 8}, {"data_fingerprint": "0" * 64},
                       {"parent_model": dict(TINY, rope_theta=1e6)}):
            with self.subTest(expect=expect):
                config = strict_config(self.tmp, self.parent, **expect)
                run_dir = self.tmp / f"bad-{len(list(self.tmp.iterdir()))}"
                with self.assertRaisesRegex(ValueError, "expectation failed"):
                    sft_torch.run(sft_torch.parse_args(["--config", str(config), "--run-dir", str(run_dir),
                                                        "--device", "cpu"]))
                self.assertFalse((run_dir / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
