#!/usr/bin/env python3
"""Full pass over v4 delivery archives: per-source char totals and char frequencies.

Streams the tar.gz archives directly (no extraction). Output: JSON with
per-(archive, source) row/char totals and a pickle of char Counters per archive.
"""
from __future__ import annotations

import argparse
import json
import pickle
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path


def iter_rows(archive: Path):
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".jsonl"):
                continue
            f = tar.extractfile(member)
            for line in f:
                yield member.name, json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    totals = defaultdict(lambda: {"rows": 0, "chars": 0})
    split_totals = defaultdict(lambda: {"rows": 0, "chars": 0})
    counters = {}
    for label, path in args.archive:
        c = Counter()
        t0 = time.time()
        n = 0
        for _, r in iter_rows(Path(path)):
            text = r["text"]
            key = f"{label}\t{r['source']}"
            totals[key]["rows"] += 1
            totals[key]["chars"] += len(text)
            s = split_totals[f"{label}\t{r['split']}"]
            s["rows"] += 1
            s["chars"] += len(text)
            c.update(text)
            n += 1
            if n % 500000 == 0:
                print(label, n, f"{time.time() - t0:.0f}s", flush=True)
        counters[label] = c
        print(label, "done", n, f"{time.time() - t0:.0f}s", flush=True)

    (args.out / "source_totals.json").write_text(
        json.dumps({"by_source": totals, "by_split": split_totals}, ensure_ascii=False, indent=1)
    )
    with (args.out / "char_counts.pkl").open("wb") as f:
        pickle.dump(counters, f)


if __name__ == "__main__":
    main()
