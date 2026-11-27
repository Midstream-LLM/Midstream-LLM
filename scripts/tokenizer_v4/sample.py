#!/usr/bin/env python3
"""Stratified, deterministic sample of v4 delivery rows for tokenizer training/eval.

Strata are (archive label, source). Allocation: share_s ∝ chars_s ** alpha, capped at
the stratum size, redistributed until the target is met. A row is kept when a hash of
(label, version_id, segment) falls under the stratum rate, so reruns are identical.

Train rows (split == "train") go to train.txt.jsonl; val/test rows are kept separately
in full as the evaluation set (eval.jsonl), together with an independent train-row
hash slice (eval_train.jsonl) for in-distribution compression numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stats import iter_rows


def allocate(sizes: dict[str, int], target: int, alpha: float) -> dict[str, float]:
    rates: dict[str, float] = {}
    free = dict(sizes)
    remaining = target
    while free:
        weights = {k: v ** alpha for k, v in free.items()}
        total_w = sum(weights.values())
        capped = {k for k, v in free.items() if remaining * weights[k] / total_w >= v}
        if not capped:
            for k, v in free.items():
                rates[k] = remaining * weights[k] / total_w / v
            break
        for k in capped:
            rates[k] = 1.0
            remaining -= free.pop(k)
    return rates


def unit(label: str, row: dict, salt: str) -> float:
    h = hashlib.blake2b(f"{salt}|{label}|{row['version_id']}|{row['segment']}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") / 2**64


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    p.add_argument("--totals", type=Path, required=True, help="source_totals.json from stats.py")
    p.add_argument("--target-chars", type=int, default=600_000_000)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--eval-train-chars", type=int, default=20_000_000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    by_source = json.loads(args.totals.read_text())["by_source"]
    sizes = {k: v["chars"] for k, v in by_source.items()}
    rates = allocate(sizes, args.target_chars, args.alpha)
    all_chars = sum(sizes.values())
    eval_rate = args.eval_train_chars / all_chars

    kept: dict[str, dict[str, int]] = {k: {"rows": 0, "chars": 0} for k in sizes}
    with (args.out / "train.jsonl").open("w") as ftrain, \
         (args.out / "eval.jsonl").open("w") as feval, \
         (args.out / "eval_train.jsonl").open("w") as fevtr:
        for label, path in args.archive:
            for _, r in iter_rows(Path(path)):
                key = f"{label}\t{r['source']}"
                rec = {"label": label, "source": r["source"], "split": r["split"], "text": r["text"]}
                if r["split"] != "train":
                    feval.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
                if unit(label, r, "eval") < eval_rate:
                    fevtr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
                if unit(label, r, "train") < rates[key]:
                    ftrain.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
                    kept[key]["rows"] += 1
                    kept[key]["chars"] += len(r["text"])

    report = {
        "target_chars": args.target_chars,
        "alpha": args.alpha,
        "strata": {
            k: {"chars": sizes[k], "rate": rates[k], **kept[k]}
            for k in sorted(sizes, key=lambda k: -sizes[k])
        },
        "train_chars": sum(v["chars"] for v in kept.values()),
    }
    (args.out / "sample_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({"train_chars": report["train_chars"]}))


if __name__ == "__main__":
    main()
