#!/usr/bin/env python3
"""Stage 6: work-level train/val/test split (jsonl format).

- work-level sha256 hash assignment (seed 42): test < 0.5%, val < 1.5%
- author disjointness (authors in train never appear in val/test)
- forced category coverage for val/test (all six categories)
- L5 cross-split dedup (sampled 24-char windows; val/test docs with >=0.5
  sampled overlap vs train are moved to train with a record)
Outputs: processed/split/{train,val,test}.jsonl + split_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (FINAL_MANIFEST, INTERIM, PROCESSED, PRIORITY_ALL,
                    read_jsonl, setup_logging, write_jsonl)

log = setup_logging("06_split")

SEED = 42
SPLIT_DIR = PROCESSED / "split"
FINAL_DIR = INTERIM / "final"

WIN_LEN = 24
WIN_STEP = 8
WIN_SAMPLE = 16


def work_of(doc: dict) -> str:
    return f"{doc['source']}::{doc['work']}"


def hassign(s: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{s}".encode()).hexdigest()[:8], 16)


def text_iter(src: str):
    p = FINAL_DIR / f"{src}.jsonl"
    if not p.exists():
        return
    with p.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main() -> None:
    docs = list(read_jsonl(FINAL_MANIFEST))
    log.info("final docs: %d", len(docs))
    doc_by_id = {d["doc_id"]: d for d in docs}

    split_of: dict[str, str] = {}
    for d in docs:
        wk = work_of(d)
        if wk not in split_of:
            r = hassign(wk) % 1000
            split_of[wk] = "test" if r < 5 else "val" if r < 15 else "train"
        d["split"] = split_of[wk]

    train_authors = set()
    for d in docs:
        if d["split"] == "train" and d.get("author"):
            for a in d["author"].split("、"):
                a = a.strip()
                if a:
                    train_authors.add(a)
    moved = 0
    for d in docs:
        if d["split"] in ("val", "test") and d.get("author"):
            if any(a.strip() in train_authors for a in d["author"].split("、") if a.strip()):
                d["split"] = "train"
                d["split_note"] = "author_seen_in_train"
                moved += 1
    log.info("moved %d docs to train (author overlap)", moved)

    cat_docs: dict[str, list] = {}
    for d in docs:
        if d["category"]:
            cat_docs.setdefault(d["category"], []).append(d)
    for cat in "123456":
        for split in ("val", "test"):
            if cat not in cat_docs:
                break
            if any(d["split"] == split for d in cat_docs[cat]):
                continue
            cand = max((d for d in cat_docs[cat] if d["split"] == "train"),
                       key=lambda d: d.get("chars_final", 0), default=None)
            if cand:
                cand["split"] = split
                cand["split_note"] = f"forced_{split}_category_coverage"
                log.info("forced cat %s -> %s: %s", cat, split, cand["doc_id"])

    # L5: cross-split dedup (stream final texts)
    train_wins: set[int] = set()
    for src in PRIORITY_ALL:
        for r in text_iter(src):
            d = doc_by_id.get(r["doc_id"])
            if d is None or d["split"] != "train":
                continue
            text = r["text"]
            if len(text) < WIN_LEN:
                continue
            for i in range(0, len(text) - WIN_LEN + 1, WIN_STEP):
                h = int(hashlib.sha1(text[i:i + WIN_LEN].encode()).hexdigest()[:16], 16)
                if h % WIN_SAMPLE == 0:
                    train_wins.add(h)
    log.info("L5 train window index: %d sampled windows", len(train_wins))

    l5_moved = 0
    for src in PRIORITY_ALL:
        for r in text_iter(src):
            d = doc_by_id.get(r["doc_id"])
            if d is None or d["split"] not in ("val", "test"):
                continue
            text = r["text"]
            if len(text) < WIN_LEN:
                continue
            hits = total = 0
            for i in range(0, len(text) - WIN_LEN + 1, WIN_STEP):
                h = int(hashlib.sha1(text[i:i + WIN_LEN].encode()).hexdigest()[:16], 16)
                if h % WIN_SAMPLE == 0:
                    total += 1
                    if h in train_wins:
                        hits += 1
            if total and hits / total >= 0.5:
                d["split"] = "train"
                d["split_note"] = "l5_cross_split_overlap"
                l5_moved += 1
    log.info("L5: moved %d val/test docs to train", l5_moved)

    counts = {}
    for split in ("train", "val", "test"):
        sub = [d for d in docs if d["split"] == split]
        counts[split] = len(sub)
        write_jsonl(SPLIT_DIR / f"{split}.jsonl", sub)
        log.info("%s: %d docs, %s chars", split, len(sub), f"{sum(d.get('chars_final',0) for d in sub):,}")
    # write split assignment back so tokenize/report can join on it
    write_jsonl(FINAL_MANIFEST, docs)
    log.info("split assignment written back to final manifest")
    write_jsonl(SPLIT_DIR / "split_manifest.json",
                [{"seed": SEED, "method": "work-level sha256; author disjoint; category coverage; L5 cross-dedup",
                  "counts": counts}])


if __name__ == "__main__":
    main()
