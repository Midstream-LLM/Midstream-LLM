"""Convert a frozen Midstream text export into dated, templated NPY training data.

The export's text already contains its template. Encode that exact rendered text
after the recorded script conversion; never re-read untemplated database bodies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


FORMAT = "jishui-midstream-v1"
REQUIRED_FIELDS = {"doc_id", "version_id", "split", "text", "source", "category", "composed_years"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def date_reason(value, before_year):
    """Inclusive source bounds, exclusive training cutoff; unknown never passes."""
    if value is None:
        return "unknown_years"
    if not isinstance(value, dict) or set(value) - {"start_year", "end_year"}:
        raise ValueError("composed_years must use start_year/end_year inclusive bounds")
    start, end = value.get("start_year"), value.get("end_year")
    if any(v is not None and (type(v) is not int or v == 0) for v in (start, end)):
        raise ValueError("invalid historical year")
    if start is not None and end is not None and start > end:
        raise ValueError("reversed composed_years")
    if end is None:
        return "unknown_upper_bound"
    if end >= before_year:
        return "crosses_cutoff" if start is None or start < before_year else "at_or_after_cutoff"
    return None


def read_rows(path, format):
    if format == "jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)
    else:
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()


class TokenWriter:
    """Bounded-memory shards; a document is never split across NPY files."""

    def __init__(self, root, dtype, shard_tokens):
        self.root, self.dtype, self.limit = root, dtype, shard_tokens
        (root / "tokens").mkdir()
        self.shard, self.size, self.handle = 0, 0, None
        self.files = []

    def close_shard(self):
        if self.handle is None:
            return
        self.handle.close()
        raw = self.root / "tokens" / "current.bin"
        target = self.root / "tokens" / f"shard_{self.shard:05d}.npy"
        with target.open("wb") as out, raw.open("rb") as source:
            np.lib.format.write_array_header_1_0(out, {
                "descr": self.dtype.str, "fortran_order": False, "shape": (self.size,)})
            shutil.copyfileobj(source, out, 4 << 20)
        raw.unlink()
        self.files.append({"file": str(target.relative_to(self.root)), "tokens": self.size,
                           "bytes": target.stat().st_size, "sha256": sha256(target)})
        self.shard += 1
        self.size, self.handle = 0, None

    def write(self, ids):
        if self.size and self.size + len(ids) > self.limit:
            self.close_shard()
        if self.handle is None:
            self.handle = (self.root / "tokens" / "current.bin").open("wb")
        shard, offset = self.shard, self.size
        self.handle.write(np.asarray(ids, dtype=self.dtype).tobytes())
        self.size += len(ids)
        return shard, offset


def prepare(export_dir, output_dir, tokenizer_path, *, before_year=1945, shard_tokens=200_000_000):
    from opencc import OpenCC
    from tokenizers import Tokenizer

    export_dir, output_dir, tokenizer_path = map(Path, (export_dir, output_dir, tokenizer_path))
    if tokenizer_path.is_dir():
        tokenizer_path /= "tokenizer.json"
    if output_dir.exists():
        raise FileExistsError(f"use a new output directory: {output_dir}")
    if type(before_year) is not int or before_year <= 0 or shard_tokens < 1:
        raise ValueError("before_year and shard_tokens must be positive integers")
    manifest_path = export_dir / "manifest.json"
    source_digest = sha256(manifest_path)
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("format") not in {"jsonl", "parquet"} or source.get("compression"):
        raise ValueError("export uncompressed JSONL or Parquet (not the raw-body tokens export)")
    if source.get("unit") != "document":
        raise ValueError("export complete documents so template prefixes and splits remain intact")
    if missing := REQUIRED_FIELDS - set(source.get("fields", [])):
        raise ValueError(f"export is missing required fields: {sorted(missing)}")
    files = source.get("files", [])
    if not files:
        raise ValueError("empty export")
    # Validate the whole input before generating data. Only manifest-listed files count.
    paths = []
    for item in files:
        path = (export_dir / item["file"]).resolve()
        if not path.is_relative_to(export_dir.resolve()) or path in paths:
            raise ValueError("invalid or repeated export file")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"export checksum mismatch: {item['file']}")
        paths.append(path)
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer_spec = json.loads(tokenizer_bytes)
    if tokenizer_spec.get("model", {}).get("dropout"):
        raise ValueError("stochastic tokenizer dropout is unsupported")
    backend = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    backend.no_padding()
    backend.no_truncation()
    vocab_size = backend.get_vocab_size(with_added_tokens=True)
    if max(backend.get_vocab(with_added_tokens=True).values()) >= vocab_size:
        raise ValueError("token IDs must lie within vocab_size")
    dtype = np.dtype("<u2" if vocab_size <= 65536 else "<u4")
    converter = OpenCC("t2s")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    writer = TokenWriter(temporary, dtype, shard_tokens)
    counts, rejected = Counter(), Counter()
    by_source, by_split = {}, {s: {"documents": 0, "tokens": 0} for s in ("train", "val", "test")}
    seen = set()
    try:
        (temporary / "manifest").mkdir()
        (temporary / "reports").mkdir()
        docs_path = temporary / "manifest/docs_tokens.jsonl"
        with docs_path.open("w", encoding="utf-8") as docs, (temporary / "reports/excluded.jsonl").open("w", encoding="utf-8") as excluded:
            for item, path in zip(files, paths):
                rows = 0
                for record in read_rows(path, source["format"]):
                    rows += 1
                    if REQUIRED_FIELDS - record.keys():
                        raise ValueError("record is missing required export fields")
                    doc_id, split = record["doc_id"], record["split"]
                    if doc_id in seen:
                        raise ValueError(f"duplicate document in export: {doc_id}")
                    seen.add(doc_id)
                    if split not in by_split or split != item["split"]:
                        raise ValueError("record split differs from frozen export")
                    if not isinstance(record["text"], str) or not isinstance(record["source"], str) or not record["source"]:
                        raise ValueError("export text and source must be strings; source must be nonempty")
                    counts["input_documents"] += 1
                    source_counts = by_source.setdefault(record["source"], Counter())
                    source_counts["input_documents"] += 1
                    reason = date_reason(record["composed_years"], before_year)
                    text = converter.convert(record["text"]) if reason is None else ""
                    ids = backend.encode(text, add_special_tokens=False).ids if reason is None else []
                    if reason is None and not ids:
                        reason = "zero_tokens"
                    if reason:
                        rejected[reason] += 1
                        source_counts[reason] += 1
                        excluded.write(json.dumps({"doc_id": doc_id, "version_id": record["version_id"],
                            "split": split, "source": record["source"], "composed_years": record["composed_years"],
                            "reason": reason}, ensure_ascii=False) + "\n")
                        continue
                    shard, offset = writer.write(ids)
                    entry = {k: v for k, v in record.items() if k != "text"}
                    entry.update(tokens=len(ids), shard=shard, offset=offset,
                                 training_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
                    docs.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["documents"] += 1
                    counts["tokens"] += len(ids)
                    source_counts["documents"] += 1
                    source_counts["tokens"] += len(ids)
                    by_split[split]["documents"] += 1
                    by_split[split]["tokens"] += len(ids)
                if rows != item["rows"] or sha256(path) != item["sha256"]:
                    raise ValueError(f"export changed or row count mismatch: {item['file']}")
        writer.close_shard()
        if not by_split["train"]["documents"] or not by_split["val"]["documents"]:
            raise ValueError("date filtering must leave both training and validation documents")
        if sha256(manifest_path) != source_digest:
            raise ValueError("export manifest changed during preparation")
        output = {"format": FORMAT, "before_year": before_year, "unknown_years": "exclude",
            "category_mode": "labels", "vocab_size": vocab_size, "dtype": dtype.str,
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
            "add_special_tokens": False, "padding": False, "truncation": False,
            "script_conversion": {"config": "t2s", "package": "opencc-python-reimplemented",
                                  "version": importlib.metadata.version("opencc-python-reimplemented")},
            "template": source.get("template"), "template_applied_by": "midstream-text-export",
            "source_export": {"manifest_sha256": source_digest, "release_id": source["release_id"],
                              "export_id": source["export_id"], "content_digest": source["content_digest"]},
            "recipe": source["recipe"], **counts, "by_split": by_split,
            "excluded": dict(rejected), "by_source": by_source, "files": writer.files}
        output["files"].append({"file": "manifest/docs_tokens.jsonl", "sha256": sha256(docs_path),
                                "bytes": docs_path.stat().st_size})
        (temporary / "manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temporary / "source_export_manifest.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output_dir)
        return output
    except BaseException:
        if writer.handle is not None and not writer.handle.closed:
            writer.handle.close()
        shutil.rmtree(temporary)
        raise


def validate_training_data(data_dir, *, before_year, vocab_size, tokenizer_sha256=None):
    """Run before model allocation, including every resume of the new trainer."""
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("before_year") != before_year:
        raise ValueError("training data must be prepared with the configured exclusive year cutoff")
    if manifest.get("vocab_size") != vocab_size:
        raise ValueError("dataset vocabulary differs from model")
    if tokenizer_sha256 and manifest.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("dataset tokenizer fingerprint differs from configuration")
    if manifest.get("script_conversion", {}).get("config") != "t2s":
        raise ValueError("this run requires simplified training text")
    for item in manifest["files"]:
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"training data checksum mismatch: {item['file']}")
    with (root / "manifest/docs_tokens.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if date_reason(record.get("composed_years"), before_year):
                raise ValueError(f"document {record['doc_id']} violates the training year cutoff")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--before-year", type=int, default=1945)
    parser.add_argument("--shard-tokens", type=int, default=200_000_000)
    args = parser.parse_args()
    result = prepare(args.export_dir, args.output_dir, args.tokenizer,
                     before_year=args.before_year, shard_tokens=args.shard_tokens)
    print(json.dumps({k: result[k] for k in ("documents", "tokens", "by_split", "excluded")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
