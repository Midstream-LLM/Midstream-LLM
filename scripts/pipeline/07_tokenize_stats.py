#!/usr/bin/env python3
"""Stage 7: tokenize final docs with frozen ccc-bbpe-32k and write shards.

Phase A (parallel): workers encode per-source docs -> tmp/{worker}.npz
Phase B: assemble uint16 shards (200M tokens) + per-doc shard/offset records
Phase C: token statistics (15 required items) -> reports/pipeline/token_stats.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (FINAL_MANIFEST, INTERIM, PRIORITY_ALL, PROCESSED, REPORTS,
                    read_jsonl, setup_logging, write_jsonl)

log = setup_logging("07_tokenize")

FINAL_DIR = INTERIM / "final"
TMP = INTERIM / "token_tmp"
TOK_DIR = PROCESSED / "tokens"
SHARD_TOKENS = 200_000_000
N_WORKERS = 8


def worker(args) -> dict:
    wid, srcs, out = args
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR := Path(__file__).parents[2] / "tokenizer/ccc-bbpe-32k"), use_fast=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_tok = 0
    with out.open("wb") as bin_f, out.with_suffix(".jsonl").open("w", encoding="utf-8") as jl_f:
        for src in srcs:
            p = FINAL_DIR / f"{src}.jsonl"
            if not p.exists():
                continue
            with p.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    ids = tok.encode(r["text"], add_special_tokens=False)
                    bin_f.write(np.asarray(ids, dtype=np.uint16).tobytes())
                    jl_f.write(json.dumps({"doc_id": r["doc_id"], "n": len(ids)}) + "\n")
                    n_docs += 1
                    n_tok += len(ids)
    log.info("worker %d: docs=%d tokens=%d", wid, n_docs, n_tok)
    return {"worker": wid, "docs": n_docs, "tokens": n_tok}


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("all", "a"):
        import shutil
        shutil.rmtree(TMP, ignore_errors=True)
        TMP.mkdir(parents=True, exist_ok=True)
        per = [PRIORITY_ALL[i::N_WORKERS] for i in range(N_WORKERS)]
        tasks = [(i, srcs, TMP / f"w{i}.bin") for i, srcs in enumerate(per)]
        with mp.Pool(N_WORKERS) as pool:
            results = pool.map(worker, tasks)
        log.info("phase A done: %s", results)

    if phase in ("all", "b"):
        docs = list(read_jsonl(FINAL_MANIFEST))
        doc_by_id = {d["doc_id"]: d for d in docs}
        TOK_DIR.mkdir(parents=True, exist_ok=True)
        for f in TOK_DIR.glob("shard_*.npy"):
            f.unlink()
        shard_info = []
        doc_records = []
        buf = np.zeros(SHARD_TOKENS, dtype=np.uint16)
        cur_shard = -1
        cur_off = 0
        n_tok_total = 0

        def flush_shard():
            nonlocal buf, cur_shard, cur_off, n_tok_total
            if cur_shard < 0 or cur_off == 0:
                return
            path = TOK_DIR / f"shard_{cur_shard:05d}.npy"
            np.save(path, buf[:cur_off])
            shard_info.append({"shard": cur_shard, "path": str(path), "tokens": int(cur_off)})
            log.info("shard %d: %d tokens", cur_shard, cur_off)

        for w in range(N_WORKERS):
            jl = TMP / f"w{w}.jsonl"
            binp = TMP / f"w{w}.bin"
            if not jl.exists() or not binp.exists():
                continue
            mm = np.memmap(binp, dtype=np.uint16, mode="r")
            start = 0
            with jl.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    doc_id, n = r["doc_id"], int(r["n"])
                    d = doc_by_id.get(doc_id)
                    if n == 0:
                        continue
                    if cur_off + n > SHARD_TOKENS:
                        flush_shard()
                        cur_shard += 1
                        cur_off = 0
                    if cur_shard < 0:
                        cur_shard = 0
                    buf[cur_off:cur_off + n] = mm[start:start + n]
                    start += n
                    cur_off += n
                    n_tok_total += n
                    doc_records.append({
                        "doc_id": doc_id, "source": d["source"] if d else "",
                        "category": d["category"] if d else "", "era": d.get("era", "") if d else "",
                        "tokens": n, "shard": cur_shard, "offset": cur_off - n,
                        "split": d.get("split", "") if d else "",
                    })
            del mm
        flush_shard()
        write_jsonl(TOK_DIR / "shard_info.jsonl", shard_info)
        write_jsonl(PROCESSED / "manifest" / "docs_tokens.jsonl", doc_records)
        log.info("phase B done: %d docs, %d tokens, %d shards",
                 len(doc_records), n_tok_total, len(shard_info))
        import shutil
        shutil.rmtree(TMP, ignore_errors=True)
        log.info("token tmp cleaned")


if __name__ == "__main__":
    main()
