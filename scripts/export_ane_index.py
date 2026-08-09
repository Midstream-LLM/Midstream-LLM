#!/usr/bin/env python3
"""Export the usable train document index for the native ANE loader.

Unlike ``export_ane_batches.py`` this does not copy token payloads. The C
loader mmaps the original NPY shards and uses these records to reproduce the
same document-boundary packing on demand. Version 2 also stores the target
category mixture and EOD id so a native run cannot silently use stale sampler
settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jishui.data import load_or_build_index, load_sampling_config  # noqa: E402

MAGIC = b"JSHANEI1"
VERSION = 2
HEADER = struct.Struct("<8sIIIIQ6dII")
RECORD = struct.Struct("<IQIB3x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/processed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-cache", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    cache = args.index_cache or args.output.with_suffix(".npz")
    index, rebuilt = load_or_build_index(args.data_dir, cache)
    ids = index.indices_for("train", known_categories_only=True)
    if len(ids) == 0:
        raise SystemExit("no usable train documents")

    sampling = load_sampling_config(args.data_dir)
    target_mix = tuple(float(sampling["target_mix"][str(category)]) for category in range(1, 7))
    eod_token_id = 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(
            HEADER.pack(
                MAGIC,
                VERSION,
                32768,
                6,
                0,
                len(ids),
                *target_mix,
                eod_token_id,
                0,
            )
        )
        for doc_index in ids:
            handle.write(
                RECORD.pack(
                    int(index.shards[doc_index]),
                    int(index.offsets[doc_index]),
                    int(index.lengths[doc_index]),
                    int(index.categories[doc_index]),
                )
            )

    totals = {
        str(category): int(index.lengths[ids[index.categories[ids] == category]].sum())
        for category in range(1, 7)
    }
    metadata = {
        "format": f"jishui-ane-index-v{VERSION}",
        "magic": MAGIC.decode("ascii"),
        "version": VERSION,
        "records": len(ids),
        "record_bytes": RECORD.size,
        "vocab_size": 32768,
        "eod_token_id": eod_token_id,
        "data_dir": str(args.data_dir.resolve()),
        "index_cache": str(cache.resolve()),
        "index_rebuilt": rebuilt,
        "train_tokens_by_category": totals,
        "target_mix": sampling["target_mix"],
        "sampling_mode": "target",
        "sha256": sha256_file(args.output),
        "index_metadata": index.metadata,
    }
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
