#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen Jishui token data")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--sample-tokens", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.data_dir
    errors: list[str] = []
    warnings: list[str] = []
    with (root / "tokens" / "shard_info.jsonl").open(encoding="utf-8") as handle:
        shard_info = [json.loads(line) for line in handle]

    shard_lengths: dict[int, int] = {}
    shard_max_token = 0
    for expected_id, info in enumerate(shard_info):
        shard_id = int(info["shard"])
        path = root / "tokens" / f"shard_{shard_id:05d}.npy"
        if shard_id != expected_id:
            errors.append(f"non-contiguous shard id {shard_id}, expected {expected_id}")
        values = np.load(path, mmap_mode="r")
        shard_lengths[shard_id] = len(values)
        if values.dtype != np.uint16:
            errors.append(f"{path}: dtype is {values.dtype}, expected uint16")
        if len(values) != int(info["tokens"]):
            errors.append(f"{path}: shape disagrees with shard_info")
        if len(values):
            stride = max(1, len(values) // max(args.sample_tokens, 1))
            shard_max_token = max(shard_max_token, int(values[::stride].max()))

    docs = Counter()
    tokens = Counter()
    category_docs = Counter()
    category_tokens = Counter()
    expected_offset = Counter()
    manifest_path = root / "manifest" / "docs_tokens.jsonl"
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            shard = int(record["shard"])
            offset = int(record["offset"])
            length = int(record["tokens"])
            split = record["split"]
            category = record.get("category") or "unknown"
            if shard not in shard_lengths:
                errors.append(f"line {line_number}: unknown shard {shard}")
                continue
            if offset != expected_offset[shard]:
                errors.append(
                    f"line {line_number}: shard {shard} offset {offset}, "
                    f"expected {expected_offset[shard]}"
                )
                if len(errors) >= 20:
                    break
            if offset + length > shard_lengths[shard]:
                errors.append(f"line {line_number}: document exceeds shard {shard}")
            expected_offset[shard] = offset + length
            docs[split] += 1
            tokens[split] += length
            category_docs[category] += 1
            category_tokens[category] += length

    for shard, length in shard_lengths.items():
        if expected_offset[shard] != length:
            errors.append(
                f"shard {shard}: manifest covers {expected_offset[shard]} of {length} tokens"
            )
    total_tokens = sum(shard_lengths.values())
    if shard_max_token >= 32768:
        errors.append(f"sampled token id {shard_max_token} exceeds vocabulary")
    unknown_tokens = category_tokens["unknown"]
    if unknown_tokens:
        warnings.append(
            f"category missing for {category_docs['unknown']:,} docs / "
            f"{unknown_tokens:,} tokens ({unknown_tokens / total_tokens:.2%})"
        )

    report = {
        "status": "error" if errors else "ok_with_warnings" if warnings else "ok",
        "shards": len(shard_info),
        "total_tokens": total_tokens,
        "tokenized_documents": sum(docs.values()),
        "documents_by_split": dict(docs),
        "tokens_by_split": dict(tokens),
        "documents_by_category": dict(category_docs),
        "tokens_by_category": dict(category_tokens),
        "sampled_max_token_id": shard_max_token,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
