#!/usr/bin/env python3
"""Export deterministic ANE training records from the frozen Jishui sampler.

The ANE reference trainer consumes a flat little-endian ``uint16`` stream,
while Jishui's source data is a set of NPY memmaps with document metadata.  A
record contains ``seq_len + 1`` ids: the first ``seq_len`` are inputs and the
shifted ``seq_len`` are labels.  EOD ids are therefore preserved exactly as
the MLX sampler emits them.

This exporter is intended for smoke tests and reproducible benchmark slices;
it deliberately requires an explicit record count instead of silently
materialising the multi-billion-token corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jishui.data import (  # noqa: E402
    PackedBatchIterator,
    load_or_build_index,
    load_sampling_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/processed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-cache", type=Path)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-mode", choices=("target", "weights"), default="target")
    parser.add_argument("--eod-token-id", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.records < 1:
        raise SystemExit("--records must be positive")
    if args.seq_len < 1 or args.batch_size < 1:
        raise SystemExit("--seq-len and --batch-size must be positive")
    if args.eod_token_id < 0 or args.eod_token_id >= 32768:
        raise SystemExit("EOD id must fit the uint16 token format")

    cache = args.index_cache or args.output.parent / "pretrain_index.npz"
    index, rebuilt = load_or_build_index(args.data_dir, cache)
    sampling = load_sampling_config(args.data_dir)
    sampler = PackedBatchIterator(
        index,
        sampling,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        mode=args.sampling_mode,
        eod_token_id=args.eod_token_id,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0
    tokens_written = 0
    with args.output.open("wb") as handle:
        while records_written < args.records:
            batch = next(sampler)
            # The ANE reference path is batch=1.  Keeping this check explicit
            # avoids producing a file whose record layout is ambiguous.
            if batch.shape[0] != 1:
                raise SystemExit("ANE record export currently requires --batch-size 1")
            ids = np.asarray(batch[0], dtype=np.uint16)
            ids.tofile(handle)
            records_written += 1
            tokens_written += args.seq_len

    metadata = {
        "format": "jishui-ane-records-v1",
        "dtype": "uint16-le",
        "records": records_written,
        "seq_len": args.seq_len,
        "record_width": args.seq_len + 1,
        "batch_size": 1,
        "training_tokens": tokens_written,
        "seed": args.seed,
        "sampling_mode": args.sampling_mode,
        "eod_token_id": args.eod_token_id,
        "data_dir": str(args.data_dir.resolve()),
        "index_cache": str(cache.resolve()),
        "index_rebuilt": rebuilt,
        "index_metadata": index.metadata,
        "category_probabilities": {
            str(int(category)): float(probability)
            for category, probability in zip(
                sampler.category_ids, sampler.category_probabilities, strict=True
            )
        },
        "sha256": sha256_file(args.output),
    }
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
