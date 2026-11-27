"""Convert frozen Midstream text exports into dated, templated NPY training data.

The export's text already contains its template. Encode that exact rendered text
after the recorded script conversion (t2s, or none to keep each row's own script);
never re-read untemplated database bodies. Several exports (e.g. an original-script
release and its simplified copy) can be combined: rows are identified by
(version_id, segment), and every doc_id must keep one split across all exports.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import shutil
import tempfile
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np


FORMAT = "jishui-midstream-v1"
REQUIRED_FIELDS = {"doc_id", "version_id", "split", "text", "source", "category", "composed_years"}
DATE_POLICIES = ("strict_upper", "exclude_known_late")
SCRIPT_CONVERSIONS = ("t2s", "none")


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


def exclusion(record, before_year, policy):
    value = record["composed_years"]
    reason = date_reason(value, before_year)
    if policy == "strict_upper":
        return reason
    if policy != "exclude_known_late":
        raise ValueError(f"unknown date policy: {policy}")
    if value and value.get("start_year") is not None and value["start_year"] >= before_year:
        return "at_or_after_cutoff"
    eras = {t["tag_id"] for t in record["tags"] if t.get("taxonomy") == "era"}
    if value is None and eras == {"dangdai"}:
        return "contemporary_era_tag"
    return None


def _encoder(tokenizer_bytes, script_conversion):
    from tokenizers import Tokenizer
    backend = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    backend.no_padding()
    backend.no_truncation()
    if script_conversion == "none":
        return None, backend
    from opencc import OpenCC
    return OpenCC(script_conversion), backend


def _init_worker(tokenizer_bytes, script_conversion):
    global _worker_encoder
    _worker_encoder = _encoder(tokenizer_bytes, script_conversion)


def _encode_rows(rows, before_year, policy, dtype, encoder=None):
    converter, backend = encoder or _worker_encoder
    result = []
    for record in rows:
        reason = exclusion(record, before_year, policy)
        text = record.pop("text")
        if not isinstance(text, str):
            raise ValueError("export text must be a string")
        ids, digest = (), None
        if reason is None:
            if converter is not None:
                text = converter.convert(text)
            ids = np.asarray(backend.encode(text, add_special_tokens=False).ids, dtype=dtype)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if not len(ids):
                reason = "zero_tokens"
        result.append((record, reason, ids, digest))
    return result


def _batches(rows, fields):
    batch, chars = [], 0
    for row in rows:
        if fields - row.keys():
            raise ValueError("record is missing required export fields")
        batch.append(row)
        chars += len(row["text"])
        if chars >= 120_000 or len(batch) >= 256:
            yield batch
            batch, chars = [], 0
    if batch:
        yield batch


def _encoded(batches, pool, workers, args, encoder):
    if pool is None:
        for batch in batches:
            yield from _encode_rows(batch, *args, encoder=encoder)
        return
    pending = deque()
    for batch in batches:
        if any(len(row["text"]) > 1_000_000 for row in batch):
            # A single book can take gigabytes of temporary tokenizer memory.
            while pending:
                yield from pending.popleft().result()
            yield from pool.submit(_encode_rows, batch, *args).result()
            continue
        pending.append(pool.submit(_encode_rows, batch, *args))
        if len(pending) >= workers * 2:
            yield from pending.popleft().result()
    for future in pending:
        yield from future.result()


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


def _load_export(export_dir, fields):
    """Validate one export completely before any data is generated."""
    manifest_path = export_dir / "manifest.json"
    digest = sha256(manifest_path)
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("format") not in {"jsonl", "parquet"} or source.get("compression"):
        raise ValueError("export uncompressed JSONL or Parquet (not the raw-body tokens export)")
    if source.get("unit") != "document":
        raise ValueError("export complete documents so template prefixes and splits remain intact")
    if missing := fields - set(source.get("fields", [])):
        raise ValueError(f"export is missing required fields: {sorted(missing)}")
    files = source.get("files", [])
    if not files:
        raise ValueError("empty export")
    # Only manifest-listed files count.
    paths = []
    for item in files:
        path = (export_dir / item["file"]).resolve()
        if not path.is_relative_to(export_dir.resolve()) or path in paths:
            raise ValueError("invalid or repeated export file")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"export checksum mismatch: {item['file']}")
        paths.append(path)
    return {"manifest_path": manifest_path, "digest": digest, "source": source, "files": files, "paths": paths}


def prepare(export_dirs, output_dir, tokenizer_path, *, before_year=1945, shard_tokens=200_000_000,
            date_policy="strict_upper", workers=1, script_conversion="t2s"):
    if date_policy not in DATE_POLICIES or workers < 1:
        raise ValueError("invalid date policy or worker count")
    if script_conversion not in SCRIPT_CONVERSIONS:
        raise ValueError(f"script_conversion must be one of {SCRIPT_CONVERSIONS}")
    fields = REQUIRED_FIELDS | ({"tags"} if date_policy == "exclude_known_late" else set())

    if isinstance(export_dirs, (str, Path)):
        export_dirs = [export_dirs]
    export_dirs = [Path(d) for d in export_dirs]
    output_dir, tokenizer_path = Path(output_dir), Path(tokenizer_path)
    if tokenizer_path.is_dir():
        tokenizer_path /= "tokenizer.json"
    if output_dir.exists():
        raise FileExistsError(f"use a new output directory: {output_dir}")
    if type(before_year) is not int or before_year <= 0 or shard_tokens < 1:
        raise ValueError("before_year and shard_tokens must be positive integers")
    if not export_dirs or len({d.resolve() for d in export_dirs}) != len(export_dirs):
        raise ValueError("give each export directory once")
    exports = [_load_export(d, fields) for d in export_dirs]
    if len({json.dumps(e["source"].get("template"), sort_keys=True) for e in exports}) != 1:
        raise ValueError("combined exports must share one template")
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer_spec = json.loads(tokenizer_bytes)
    if tokenizer_spec.get("model", {}).get("dropout"):
        raise ValueError("stochastic tokenizer dropout is unsupported")
    encoder = _encoder(tokenizer_bytes, script_conversion)
    backend = encoder[1]
    vocab_size = backend.get_vocab_size(with_added_tokens=True)
    if max(backend.get_vocab(with_added_tokens=True).values()) >= vocab_size:
        raise ValueError("token IDs must lie within vocab_size")
    dtype = np.dtype("<u2" if vocab_size <= 65536 else "<u4")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    writer = TokenWriter(temporary, dtype, shard_tokens)
    counts, rejected = Counter(), Counter()
    by_source, by_split = {}, {s: {"documents": 0, "tokens": 0} for s in ("train", "val", "test")}
    seen, doc_splits = set(), {}
    try:
        (temporary / "manifest").mkdir()
        (temporary / "reports").mkdir()
        docs_path = temporary / "manifest/docs_tokens.jsonl"
        with ExitStack() as stack:
            docs = stack.enter_context(docs_path.open("w", encoding="utf-8"))
            excluded = stack.enter_context((temporary / "reports/excluded.jsonl").open("w", encoding="utf-8"))
            pool = stack.enter_context(ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker,
                       initargs=(tokenizer_bytes, script_conversion))) if workers > 1 else None
            for export, item, path in ((e, i, p) for e in exports for i, p in zip(e["files"], e["paths"])):
                rows = 0
                batches = _batches(read_rows(path, export["source"]["format"]), fields)
                for record, reason, ids, text_digest in _encoded(batches, pool, workers, (before_year, date_policy, dtype), encoder):
                    rows += 1
                    doc_id, split = record["doc_id"], record["split"]
                    # Span-cut exports emit several rows per version; a simplified copy is
                    # its own version of the same doc_id.
                    key = (record["version_id"], record.get("segment"))
                    if key in seen:
                        raise ValueError(f"duplicate document row in exports: version {key[0]} segment {key[1]}")
                    seen.add(key)
                    if split not in by_split or split != item["split"]:
                        raise ValueError("record split differs from frozen export")
                    if doc_splits.setdefault(doc_id, split) != split:
                        raise ValueError(f"document {doc_id} has rows in both {doc_splits[doc_id]} and {split}")
                    if not isinstance(record["source"], str) or not record["source"]:
                        raise ValueError("export source must be a nonempty string")
                    counts["input_documents"] += 1
                    source_counts = by_source.setdefault(record["source"], Counter())
                    source_counts["input_documents"] += 1
                    if reason:
                        rejected[reason] += 1
                        source_counts[reason] += 1
                        excluded.write(json.dumps({"doc_id": doc_id, "version_id": record["version_id"],
                            "segment": record.get("segment"), "split": split, "source": record["source"], "composed_years": record["composed_years"],
                            "reason": reason}, ensure_ascii=False) + "\n")
                        continue
                    shard, offset = writer.write(ids)
                    entry = record
                    entry.update(tokens=len(ids), shard=shard, offset=offset,
                                 training_text_sha256=text_digest)
                    docs.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["documents"] += 1
                    counts["tokens"] += len(ids)
                    source_counts["documents"] += 1
                    source_counts["tokens"] += len(ids)
                    by_split[split]["documents"] += 1
                    by_split[split]["tokens"] += len(ids)
                if rows != item["rows"] or sha256(path) != item["sha256"]:
                    raise ValueError(f"export changed or row count mismatch: {item['file']}")
                if workers > 1:
                    print(f"Encoded {item['file']}: {rows} rows; {counts['tokens']:,} total tokens", flush=True)
        writer.close_shard()
        if not by_split["train"]["documents"] or not by_split["val"]["documents"]:
            raise ValueError("date filtering must leave both training and validation documents")
        if any(sha256(e["manifest_path"]) != e["digest"] for e in exports):
            raise ValueError("export manifest changed during preparation")
        if script_conversion == "none":
            conversion = {"config": "none"}
        else:
            conversion = {"config": script_conversion, "package": "opencc-python-reimplemented",
                          "version": importlib.metadata.version("opencc-python-reimplemented")}
        output = {"format": FORMAT, "before_year": before_year, "date_policy": date_policy,
            "unknown_years": "exclude" if date_policy == "strict_upper" else "include",
            "category_mode": "labels", "vocab_size": vocab_size, "dtype": dtype.str,
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
            "add_special_tokens": False, "padding": False, "truncation": False,
            "script_conversion": conversion, "row_key": ["version_id", "segment"],
            "template": exports[0]["source"].get("template"), "template_applied_by": "midstream-text-export",
            "source_exports": [{"manifest_sha256": e["digest"], "release_id": e["source"]["release_id"],
                                "export_id": e["source"]["export_id"], "content_digest": e["source"]["content_digest"],
                                "recipe": e["source"]["recipe"]} for e in exports], **counts, "by_split": by_split,
            "excluded": dict(rejected), "by_source": by_source, "files": writer.files}
        output["files"].append({"file": "manifest/docs_tokens.jsonl", "sha256": sha256(docs_path),
                                "bytes": docs_path.stat().st_size})
        (temporary / "manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temporary / "source_export_manifests.json").write_text(
            json.dumps([e["source"] for e in exports], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output_dir)
        return output
    except BaseException:
        if writer.handle is not None and not writer.handle.closed:
            writer.handle.close()
        shutil.rmtree(temporary)
        raise


def validate_training_data(data_dir, *, before_year, vocab_size, tokenizer_sha256=None, date_policy="strict_upper",
                           script_conversion="t2s"):
    """Run before model allocation, including every resume of the new trainer."""
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if date_policy not in DATE_POLICIES or manifest.get("date_policy", "strict_upper") != date_policy:
        raise ValueError("training data date policy differs from configuration")
    if manifest.get("format") != FORMAT or manifest.get("before_year") != before_year:
        raise ValueError("training data must be prepared with the configured exclusive year cutoff")
    if manifest.get("vocab_size") != vocab_size:
        raise ValueError("dataset vocabulary differs from model")
    if tokenizer_sha256 and manifest.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("dataset tokenizer fingerprint differs from configuration")
    if manifest.get("script_conversion", {}).get("config") != script_conversion:
        raise ValueError(f"training data script conversion differs from configuration ({script_conversion})")
    for item in manifest["files"]:
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"training data checksum mismatch: {item['file']}")
    with (root / "manifest/docs_tokens.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if exclusion(record, before_year, date_policy):
                raise ValueError(f"document {record['doc_id']} violates the training year cutoff")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, action="append", required=True,
                        help="repeat to combine exports, e.g. an original-script release and its t2s copy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--before-year", type=int, default=1945)
    parser.add_argument("--date-policy", choices=DATE_POLICIES, default="strict_upper")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--script-conversion", choices=SCRIPT_CONVERSIONS, default="t2s")
    parser.add_argument("--shard-tokens", type=int, default=200_000_000)
    args = parser.parse_args()
    result = prepare(args.export_dir, args.output_dir, args.tokenizer,
                     before_year=args.before_year, shard_tokens=args.shard_tokens,
                     date_policy=args.date_policy, workers=args.workers,
                     script_conversion=args.script_conversion)
    print(json.dumps({k: result[k] for k in ("documents", "tokens", "by_split", "excluded")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
