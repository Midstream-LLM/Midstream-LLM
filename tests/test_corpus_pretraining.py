import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from jishui.config import ModelConfig, TrainConfig
from jishui.data import build_pretrain_index, ShuffledDocumentIterator, PackedBatchIterator
from jishui.prepare_corpus import prepare, sha256, validate_training_data, date_reason
from jishui.pretrain_torch import run, natural_sampling
from jishui.schedules_torch import continuation_lr, make_pretraining_schedule
from jishui.tracking import ExperimentTracker


class CorpusPretrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.export = self.root / "export"
        self.export.mkdir()
        words = ["<unk>", "<pad>", "<|endoftext|>", "《史记》", "司马迁", "古文", "正文",
                 "诗", "经", "集", "未分类", "甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"]
        self.tokenizer = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        self.tokenizer_path = self.root / "tokenizer.json"
        self.tokenizer.save(str(self.tokenizer_path))
        self.records = [
            self.record(1, "train", "经", 100, 200, "《史記》 司馬遷 古文 正文"),
            self.record(2, "train", None, 1900, 1944, "甲 乙 丙 丁"),
            self.record(3, "val", "集", 1800, 1800, "诗 集 庚 辛"),
            self.record(4, "test", "集", 1945, 1945, "戊"),
            self.record(5, "train", "经", 1930, 1950, "己"),
            self.record(6, "val", None, None, None, "庚"),
            self.record(7, "train", "集", 2000, 2026, "辛"),
        ]
        self.write_export()

    def tearDown(self):
        self.temp.cleanup()

    def record(self, id, split, category, start, end, text):
        return {"doc_id": id, "version_id": id + 100, "split": split, "category": category,
                "source": "fixture", "composed_years": {"start_year": start, "end_year": end}
                if start is not None or end is not None else None, "text": text}

    def write_export(self):
        manifest = {"format": "jsonl", "unit": "document", "compression": None,
                    "release_id": 9, "export_id": 10, "content_digest": "fixture",
                    "recipe": {"name": "fixture"}, "template": {"format": "[{title} ]{text}", "ratio": 1, "seed": 0},
                    "fields": list(self.records[0]), "files": []}
        for split in ("train", "val", "test"):
            rows = [r for r in self.records if r["split"] == split]
            if not rows:
                continue
            path = self.export / f"{split}.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            manifest["files"].append({"file": path.name, "split": split, "rows": len(rows),
                                      "bytes": path.stat().st_size, "sha256": sha256(path)})
        (self.export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def prepare(self, name="data"):
        path = self.root / name
        report = prepare(self.export, path, self.tokenizer_path, shard_tokens=5)
        return path, report

    def test_templates_dates_splits_and_simplified_tokens(self):
        data, report = self.prepare()
        self.assertEqual(report["documents"], 3)
        self.assertEqual(report["excluded"], {"at_or_after_cutoff": 2, "crosses_cutoff": 1, "unknown_years": 1})
        records = [json.loads(line) for line in (data / "manifest/docs_tokens.jsonl").read_text().splitlines()]
        first = records[0]
        ids = np.load(data / "tokens" / f"shard_{first['shard']:05d}.npy")[first["offset"]:first["offset"] + first["tokens"]]
        self.assertEqual(ids.tolist(), [3, 4, 5, 6])  # title + author actually enter NPY, in simplified script
        self.assertEqual([r["split"] for r in records], ["train", "train", "val"])
        self.assertEqual(sum(report["by_split"][s]["tokens"] for s in report["by_split"]), report["tokens"])
        validate_training_data(data, before_year=1945, vocab_size=19, tokenizer_sha256=sha256(self.tokenizer_path))
        with self.assertRaisesRegex(ValueError, "cutoff"):
            validate_training_data(data, before_year=1946, vocab_size=19)
        self.assertIsNone(date_reason({"start_year": None, "end_year": 1944}, 1945))
        self.assertEqual(date_reason({"start_year": 1900, "end_year": None}, 1945), "unknown_upper_bound")
        for years in ({"start_year": 1950, "end_year": 1900}, {"start_year": True, "end_year": 1800}):
            with self.assertRaises(ValueError):
                date_reason(years, 1945)

    def test_incomplete_export_and_modified_files_fail(self):
        path = self.export / "train.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.prepare()
        self.write_export()
        self.records.append(self.record(1, "val", "经", 100, 200, "甲"))
        self.write_export()
        with self.assertRaisesRegex(ValueError, "duplicate document"):
            self.prepare()
        self.assertFalse((self.root / "data").exists())
        self.assertFalse(list(self.root.glob(".data-*")))

    def test_open_date_policy_and_parallel_encoding(self):
        self.records.extend([self.record(8, "test", None, 1950, None, "庚"),
                             self.record(9, "test", None, None, None, "辛")])
        for row in self.records:
            row["tags"] = []
        self.records[-1]["tags"] = [{"taxonomy": "era", "tag_id": "dangdai"}]
        self.records[4]["tags"] = [{"taxonomy": "era", "tag_id": "minguo"},
                                   {"taxonomy": "era", "tag_id": "dangdai"}]
        self.write_export()
        policy = "exclude_known_late"
        a, b = self.root / "serial", self.root / "parallel"
        serial = prepare(self.export, a, self.tokenizer_path, date_policy=policy)
        parallel = prepare(self.export, b, self.tokenizer_path, date_policy=policy, workers=2)
        self.assertEqual(serial, parallel)
        self.assertEqual(serial["documents"], 5)
        self.assertEqual(serial["excluded"], {"at_or_after_cutoff": 3, "contemporary_era_tag": 1})
        validate_training_data(b, before_year=1945, vocab_size=19, date_policy=policy)
        with self.assertRaisesRegex(ValueError, "date policy"):
            validate_training_data(b, before_year=1945, vocab_size=19)

    def test_dynamic_categories_and_unknown_are_sampled(self):
        data, _ = self.prepare()
        index = build_pretrain_index(data, category_mode="labels")
        self.assertEqual(index.metadata["category_labels"], {"1": "经", "2": None, "3": "集"})
        self.assertEqual(index.token_totals("train"), {1: 4, 2: 4, 3: 0})
        sampler = PackedBatchIterator(index, natural_sampling(index), 5, 2, 1)
        self.assertEqual(set(sampler.category_ids), {1, 2})
        self.assertEqual(next(sampler).shape, (2, 6))
        # More than 255 categories/shards/sources must not wrap around uint8.
        path = data / "manifest/docs_tokens.jsonl"
        path.write_text("".join(json.dumps({"doc_id": i, "source": str(i), "category": f"class{i}",
            "shard": i, "offset": 0, "tokens": 1, "split": "train"}) + "\n" for i in range(300)))
        large = build_pretrain_index(data, category_mode="labels")
        self.assertEqual(int(large.categories[-1]), 300)
        self.assertEqual(int(large.shards[-1]), 299)
        self.assertEqual(int(large.sources[-1]), 299)

    def test_parquet_matches_jsonl_and_missing_year_field_is_rejected(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        first, expected = self.prepare("jsonl-data")
        path = self.export / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["format"] = "parquet"
        for item in manifest["files"]:
            records = [r for r in self.records if r["split"] == item["split"]]
            target = self.export / f"{item['split']}.parquet"
            pq.write_table(pa.Table.from_pylist(records), target)
            item.update(file=target.name, bytes=target.stat().st_size, sha256=sha256(target))
        path.write_text(json.dumps(manifest))
        second, actual = self.prepare("parquet-data")
        self.assertEqual(actual["tokens"], expected["tokens"])
        self.assertEqual((first / "manifest/docs_tokens.jsonl").read_text(), (second / "manifest/docs_tokens.jsonl").read_text())
        for shard in (first / "tokens").glob("*.npy"):
            np.testing.assert_array_equal(np.load(shard), np.load(second / "tokens" / shard.name))
        manifest["fields"].remove("composed_years")
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            self.prepare("invalid-data")

    def test_document_stream_covers_prefixes_eod_and_resumes(self):
        data, _ = self.prepare()
        index = build_pretrain_index(data, category_mode="labels")
        stream = ShuffledDocumentIterator(index, 3, 1, 42)
        chunks = [next(stream)[0]]
        state = copy.deepcopy(stream.state_dict())
        expected = [next(stream) for _ in range(5)]
        other = ShuffledDocumentIterator(index, 3, 1, 42)
        other.load_state_dict(state)
        for batch in expected:
            np.testing.assert_array_equal(next(other), batch)
        chunks += [batch[0][1:] for batch in expected]
        full = np.concatenate(chunks)
        self.assertEqual(sorted(full[:10].tolist()), sorted([3, 4, 5, 6, 2, 11, 12, 13, 14, 2]))

    def config(self):
        model = ModelConfig(name="tiny", vocab_size=19, hidden_size=8, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=4, intermediate_size=12,
            max_position_embeddings=16, gradient_checkpointing=False)
        training = TrainConfig(dtype="float32", seq_len=4, batch_size=1, grad_accum_steps=2,
            warmup_steps=2, eval_batches=1, eval_interval=2, save_interval=2,
            max_checkpoints=5, optimizer_checkpoints=5, log_interval=1)
        config = {"model": model.to_dict(), "training": training.to_dict(),
                  "pretraining": {"target_tokens": 48, "schedule": "cosine", "sampling": "document_stream"},
                  "data": {"category_mode": "labels", "before_year": 1945, "tokenizer_sha256": sha256(self.tokenizer_path)}}
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        return path

    def train(self, data, name, *, steps=None, resume=None, init_from=None, block=False, migrate_from=None):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        config_path = self.config()
        if block:
            spec = json.loads(config_path.read_text())
            spec["pretraining"].update(sampling="shuffled_blocks", prediction_offsets=[1],
                                       validation={"mode": "source_stratified_blocks", "seed": 71})
            config_path.write_text(json.dumps(spec))
        args = SimpleNamespace(config=config_path, run_dir=directory, data_dir=data, index_cache=None,
            resume=resume, init_from=init_from, device="cpu", steps=steps, evaluate_first=True,
            validate_only=False, expected_val_loss=None, val_tolerance=.03, migrate_from=migrate_from)
        run(args, ExperimentTracker(directory))
        latest = json.loads((directory / "latest.json").read_text())["checkpoint"]
        return Path(latest), torch.load(Path(latest) / "training.pt", weights_only=True)

    def test_full_fresh_train_resume_and_weights_only_start(self):
        data, _ = self.prepare()
        _, continuous = self.train(data, "continuous")
        partial_path, _ = self.train(data, "resumed", steps=3)
        _, resumed = self.train(data, "resumed", resume="latest")
        for key in continuous["model"]:
            torch.testing.assert_close(continuous["model"][key], resumed["model"][key], rtol=0, atol=0)
        self.assertEqual(continuous["state"]["sampler"], resumed["state"]["sampler"])
        self.assertEqual(resumed["state"]["tokens_seen"], 48)
        _, initialized = self.train(data, "initialized", steps=1, init_from=partial_path)
        self.assertEqual(initialized["state"]["step"], 1)
        self.assertEqual(initialized["state"]["tokens_seen"], 8)
        self.assertEqual(initialized["state"]["origin"]["format"], "weights_only")
        for state in initialized["optimizer"]["state"].values():
            self.assertEqual(state["step"].item(), 1)
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.train(data, "initialized", steps=1)
        token_file = data / "tokens/shard_00000.npy"
        with token_file.open("ab") as handle:
            handle.write(b"x")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.train(data, "initialized", steps=1, resume="latest")

    def test_new_schedule_has_warmup_and_final_minimum(self):
        config = TrainConfig(warmup_steps=2)
        schedule = make_pretraining_schedule(config, 6)
        rates = [continuation_lr(schedule, step) for step in range(6)]
        self.assertAlmostEqual(rates[0], config.learning_rate / 2)
        self.assertAlmostEqual(rates[1], config.learning_rate)
        self.assertAlmostEqual(rates[-1], config.min_learning_rate)

    def test_block_training_exact_resume_with_partial_blocks(self):
        data, _ = self.prepare()
        _, full = self.train(data, "block-full", block=True)
        self.train(data, "block-resume", block=True, steps=3)
        _, resumed = self.train(data, "block-resume", block=True, resume="latest")
        for key in full["model"]:
            torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
        for key in ("sampler", "tokens_seen", "schedule", "evaluation", "log_window"):
            self.assertEqual(full["state"][key], resumed["state"][key])
        self.assertEqual(full["state"]["evaluation"]["version"], "source_stratified_blocks_v1")

    def test_full_state_migration_preserves_adam_schedule_and_rng(self):
        data, _ = self.prepare()
        parent_path, parent = self.train(data, "parent", steps=3)
        _, child = self.train(data, "blocks", block=True, migrate_from=parent_path, steps=1)
        migrated = torch.load(self.root / "blocks/checkpoints/step_00000003/training.pt", weights_only=True)
        for key in parent["model"]:
            torch.testing.assert_close(parent["model"][key], migrated["model"][key], rtol=0, atol=0)
        for pid, state in parent["optimizer"]["state"].items():
            for key in state:
                torch.testing.assert_close(state[key], migrated["optimizer"]["state"][pid][key], rtol=0, atol=0)
        self.assertEqual(parent["optimizer"]["param_groups"], migrated["optimizer"]["param_groups"])
        self.assertEqual(parent["scaler"], migrated["scaler"])
        torch.testing.assert_close(parent["rng"], migrated["rng"], rtol=0, atol=0)
        for key in ("step", "tokens_seen", "schedule"):
            self.assertEqual(parent["state"][key], migrated["state"][key])
        self.assertEqual(child["state"]["step"], 4)
        self.assertEqual(child["state"]["sampler"]["kind"], "shuffled_blocks")
        self.assertGreater(child["state"]["tokens_seen"], parent["state"]["tokens_seen"])
        for state in child["optimizer"]["state"].values():
            self.assertEqual(state["step"].item(), 4)
        # Exact resume cannot silently apply a sampler migration.
        with self.assertRaisesRegex(ValueError, "resume changed"):
            self.train(data, "parent", block=True, resume="latest", steps=1)


if __name__ == "__main__":
    unittest.main()
