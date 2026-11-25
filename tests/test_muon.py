"""Muon + AdamW optimizer: math, parameter split, exact resume, DDP equivalence."""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from jishui.config import TrainConfig
from jishui.optim_torch import MuonAdamW, build_optimizer, muon_parameters, newton_schulz
from tests.test_document_attention import DistributedTrainingTests, _worker, free_port, tiny


class MuonMathTests(unittest.TestCase):
    def test_newton_schulz_orthogonalizes_both_orientations(self):
        torch.manual_seed(0)
        for shape in ((32, 96), (96, 32), (48, 48)):
            g = torch.randn(*shape)
            x = newton_schulz(g)
            self.assertEqual(x.shape, g.shape)
            s = torch.linalg.svdvals(x)
            # Five quintic steps push singular values into ~[0.7, 1.2]; the very
            # smallest ones of a random square matrix grow more slowly.
            self.assertGreater(float(s.median()), 0.6)
            self.assertLess(float(s.max()), 1.25)
            # Same singular vectors as the exact polar factor.
            u, _, vh = torch.linalg.svd(g, full_matrices=False)
            cosine = (x * (u @ vh)).sum() / (x.norm() * (u @ vh).norm())
            self.assertGreater(float(cosine), 0.95)

    def test_fp16_iteration_matches_fp32(self):
        torch.manual_seed(3)
        g = torch.randn(64, 160)
        full = newton_schulz(g, dtype=torch.float32)
        half = newton_schulz(g, dtype=torch.float16).float()
        self.assertTrue(torch.isfinite(half).all())
        self.assertLess(float((half - full).norm() / full.norm()), 2e-2)

    def test_update_rms_matches_adamw_scale(self):
        p = torch.nn.Parameter(torch.zeros(64, 256))
        opt = MuonAdamW([{"params": [p], "use_muon": True}], lr=1.0, momentum=0.0, nesterov=False)
        p.grad = torch.randn(64, 256)
        opt.step()
        rms = float(p.detach().pow(2).mean().sqrt())
        self.assertAlmostEqual(rms, 0.2, delta=0.05)

    def test_adam_branch_equals_torch_adamw(self):
        torch.manual_seed(1)
        a = torch.nn.Parameter(torch.randn(7, 5))
        b = torch.nn.Parameter(a.detach().clone())
        ours = MuonAdamW([{"params": [a], "use_muon": False}], lr=1e-2, weight_decay=0.1,
                         betas=(0.9, 0.95), eps=1e-5)
        ref = torch.optim.AdamW([b], lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-5)
        for _ in range(5):
            g = torch.randn(7, 5)
            a.grad, b.grad = g.clone(), g.clone()
            ours.step(); ref.step()
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    def test_muon_rejects_non_matrices(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            MuonAdamW([{"params": [torch.nn.Parameter(torch.zeros(3))], "use_muon": True}], lr=1.0)

    def test_parameter_split_covers_model_once(self):
        model = tiny()
        muon, adam = muon_parameters(model)
        ids = [id(p) for p in muon + adam]
        self.assertEqual(sorted(ids), sorted(id(p) for p in model.parameters()))
        self.assertEqual(len(set(ids)), len(ids))
        self.assertIn(id(model.embed_tokens.weight), {id(p) for p in adam})
        self.assertTrue(all(p.dim() == 2 for p in muon))
        # 2 layers x (q, k, v, o, gate, up, down)
        self.assertEqual(len(muon), 14)

    def test_build_optimizer_validates_settings(self):
        model, config = tiny(), TrainConfig()
        self.assertIsInstance(build_optimizer(model, config, None), torch.optim.AdamW)
        self.assertIsInstance(build_optimizer(model, config, {"name": "muon"}), MuonAdamW)
        with self.assertRaisesRegex(ValueError, "unknown muon"):
            build_optimizer(model, config, {"name": "muon", "lr": 0.02})
        with self.assertRaisesRegex(ValueError, "optimizer must be"):
            build_optimizer(model, config, {"name": "sgd"})

    def test_state_dict_round_trip_continues_identically(self):
        torch.manual_seed(2)
        model_a, model_b = tiny(), tiny()
        opt_a = build_optimizer(model_a, TrainConfig(), {"name": "muon"})
        x = torch.randint(3, 40, (2, 9))
        def step(model, opt):
            opt.zero_grad()
            model(x).float().logsumexp(-1).mean().backward()
            opt.step()
        for _ in range(3):
            step(model_a, opt_a)
        model_b.load_state_dict(model_a.state_dict())
        opt_b = build_optimizer(model_b, TrainConfig(), {"name": "muon"})
        # Deep copy: an uncast load shares tensors with the source optimizer.
        opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))
        step(model_a, opt_a); step(model_b, opt_b)
        for (k, a), b in zip(model_a.state_dict().items(), model_b.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=k)


class MuonTrainingTests(DistributedTrainingTests):
    def config(self, batch_size, attention, optimizer=("muon",)):
        path = super().config(batch_size, attention)
        spec = json.loads(path.read_text())
        spec["pretraining"]["optimizer"] = {"name": optimizer[0]}
        path = path.with_name(path.stem + f"-{optimizer[0]}.json")
        path.write_text(json.dumps(spec))
        return path

    def test_two_rank_training_matches_single_process(self):
        single, ddp = self.run_pair("document")
        self.assertEqual(single["state"]["pretraining"]["optimizer"], {"name": "muon"})
        for key in single["model"]:
            torch.testing.assert_close(ddp["model"][key], single["model"][key], rtol=1e-5, atol=1e-6)
        for key in ("step", "tokens_seen", "sampler"):
            self.assertEqual(ddp["state"][key], single["state"][key])
        self.assertTrue(any("momentum_buffer" in s for s in single["optimizer"]["state"].values()))

    def test_document_attention_changes_training(self):
        self.skipTest("covered by DistributedTrainingTests")

    def test_exact_resume_and_optimizer_change_rejected(self):
        from jishui.pretrain_torch import run
        from jishui.tracking import ExperimentTracker
        def train(name, config, **extra):
            args = self.args(config, name)
            args.update(extra)
            args["run_dir"].mkdir(exist_ok=True)
            run(SimpleNamespace(**args), ExperimentTracker(args["run_dir"], None, mode="disabled"))
            return self.final(name)
        muon = self.config(2, "document")
        full = train("full", muon)
        train("part", muon, steps=1)
        resumed = train("part", muon, resume="latest")
        for key in full["model"]:
            torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
        adamw = self.config(2, "document", optimizer=("adamw",))
        train("switch", adamw, steps=1)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            train("switch", muon, resume="latest")


if __name__ == "__main__":
    unittest.main()
