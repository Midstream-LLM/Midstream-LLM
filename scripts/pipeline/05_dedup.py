#!/usr/bin/env python3
"""Stage 5 v4: five-level dedup, memory-safe, audit every drop.

L1 file-level exact (raw bytes sha1)          -> l1_file_exact.jsonl
L2 doc-level exact (clean text sha1)          -> l2_doc_exact.jsonl
L3 paragraph-level exact (t2s-normalized sha) -> sqlite para + l3 jsonl (streamed)
L4 doc-level near-dup (sampled window index)  -> l4_doc_near.jsonl
L3n paragraph near-dup inside flagged pairs   -> l3n_para_near.jsonl
Final rewrite pass applies L4/L3n drops to interim/final/{source}.jsonl.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (DOC_MANIFEST, DEDUP_DIR, INTERIM, PRIORITY_ALL, read_jsonl,
                    setup_logging, sha1_text, write_jsonl)

log = setup_logging("05_dedup")

CLEAN_DIR = INTERIM / "clean"
FINAL_DIR = INTERIM / "final"
CLEAN_MANIFEST = INTERIM / "manifest" / "docs_clean_norm.jsonl"
FINAL_MANIFEST = INTERIM / "manifest" / "docs_final.jsonl"
DB = Path(os.environ.get("JISHUI_PARA_DB", str(DEDUP_DIR / "para.sqlite")))
WIN_DB = Path(os.environ.get("JISHUI_WIN_DB", str(DEDUP_DIR / "win.sqlite")))

WIN_LEN = 24
WIN_STEP = 8
WIN_SAMPLE = 16
L3_FLUSH = 100_000
L4_TOPK = 12000
BATCH_PARAS = 200_000
N_WORKERS = max(2, min(os.cpu_count() or 4, 6))


_CC = None


def _cc_init() -> None:
    """Worker initializer: one OpenCC t2s converter per process."""
    global _CC
    from opencc import OpenCC
    _CC = OpenCC("t2s")


def _t2s_sha(item):
    """(doc_id, '\\n'.join(paras)) -> (doc_id, [15-hex sha per para]).

    One OpenCC call per doc (per-call overhead dominates for short paras);
    joined-with-newline conversion is identical to per-paragraph conversion
    (mmseg treats '\\n' as a segment boundary; verified on 50k paras).
    """
    doc_id, text = item
    conv = _CC.convert(text)
    return doc_id, [int(sha1_text(p)[:15], 16) for p in conv.split("\n")]


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n+", text) if p.strip()]


def windows_24(text: str, step: int = WIN_STEP):
    if len(text) < WIN_LEN:
        return
    for i in range(0, len(text) - WIN_LEN + 1, step):
        yield text[i:i + WIN_LEN]


def src_prio(src: str) -> int:
    try:
        return PRIORITY_ALL.index(src)
    except ValueError:
        return 99


def stream_source(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main() -> None:
    clean_docs = [d for d in read_jsonl(DOC_MANIFEST)]
    kept_ids: dict[str, set] = {}
    for p in sorted(CLEAN_DIR.glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            kept_ids[p.stem] = {json.loads(l)["doc_id"] for l in f}
    clean_docs = [d for d in clean_docs if d["doc_id"] in kept_ids.get(d["source"], set())]
    doc_by_id = {d["doc_id"]: d for d in clean_docs}
    write_jsonl(CLEAN_MANIFEST, clean_docs)
    log.info("clean docs: %d", len(clean_docs))
    del kept_ids

    l1p, l2p = DEDUP_DIR / "l1_file_exact.jsonl", DEDUP_DIR / "l2_doc_exact.jsonl"
    if not (l1p.exists() and l2p.exists()):
        file_sha: dict[str, str] = {}
        l1 = []
        for d in sorted(clean_docs, key=lambda x: (src_prio(x["source"]), x["doc_id"])):
            if d["source"] not in ("daizhige", "poetry", "gutenberg"):
                continue
            try:
                h = sha1_text(Path(d["path"]).read_bytes().hex())
            except Exception as e:
                l1.append({"doc_id": d["doc_id"], "source": d["source"], "reason": f"read error: {e}",
                           "dup_source": None, "similarity": 1.0})
                continue
            prev = file_sha.get(h)
            if prev is not None:
                l1.append({"doc_id": d["doc_id"], "source": d["source"], "dup_source": prev,
                           "reason": "文件级精确重复(原始字节sha1)", "similarity": 1.0,
                           "chars": d.get("chars_raw", 0)})
                d["l1"] = "dup"
            else:
                file_sha[h] = d["doc_id"]
        write_jsonl(l1p, l1)
        log.info("L1 dropped: %d", sum(1 for r in l1 if r.get("dup_source")))

        doc_sha: dict[str, str] = {}
        l2 = []
        for src in PRIORITY_ALL:
            p = CLEAN_DIR / f"{src}.jsonl"
            if not p.exists():
                continue
            for r in stream_source(p):
                d = doc_by_id.get(r["doc_id"])
                if d is None or d.get("l1") == "dup":
                    continue
                h = sha1_text(r["text"])
                d["sha1_clean"] = h
                prev = doc_sha.get(h)
                if prev is not None:
                    l2.append({"doc_id": d["doc_id"], "source": src, "dup_source": prev,
                               "reason": "文档级精确重复(清洗后文本sha1)", "similarity": 1.0,
                               "chars": len(r["text"])})
                    d["l2"] = "dup"
                else:
                    doc_sha[h] = d["doc_id"]
        write_jsonl(l2p, l2)
        log.info("L2 dropped: %d", sum(1 for r in l2 if r.get("dup_source")))
    else:
        log.info("L1/L2 done, skipping")

    # ---- L3 (resumable per source, parallel t2s+sha1) ----
    l3p = DEDUP_DIR / "l3_para_exact.jsonl"
    done_path = DEDUP_DIR / "l3_done.json"
    done_srcs: set[str] = set(read_jsonl(done_path)) if done_path.exists() else set()
    DEDUP_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-300000")
    db.executescript("CREATE TABLE IF NOT EXISTS para(sha INTEGER PRIMARY KEY, doc TEXT)")
    if not done_srcs:
        # fresh run: clear any leftover state from an interrupted first attempt
        db.execute("DELETE FROM para")
        db.commit()
        l3p.unlink(missing_ok=True)
        for f in FINAL_DIR.glob("*.jsonl"):
            f.unlink()
    dropped_ids: set[str] = set()
    if l3p.exists():
        n_rec = 0
        for r in read_jsonl(l3p):
            dropped_ids.add(r["doc_id"])
            n_rec += 1
        log.info("L3 existing records: %d, docs with drops: %d", n_rec, len(dropped_ids))
    if done_srcs:
        log.info("L3 resume: skipping %s", sorted(done_srcs))

    progress_path = DEDUP_DIR / "l3_progress.json"
    progress: dict[str, dict] = {}
    if progress_path.exists():
        for r in read_jsonl(progress_path):
            progress[r["src"]] = {"last_doc": r["last_doc"], "done": r.get("done", False)}
        log.info("L3 progress: %s", progress)

    pool = mp.Pool(N_WORKERS, initializer=_cc_init)
    l3_buf: list[dict] = []
    dropped_ids: set[str] = set()
    n_para = n_drop = 0
    final_writers: dict[str, object] = {}
    sql = "INSERT OR IGNORE INTO para(sha, doc) VALUES (?, ?)"

    def flush_audit() -> None:
        nonlocal l3_buf
        if l3_buf:
            write_jsonl(l3p, l3_buf, append=True)
            l3_buf = []

    def flush_buf(buf) -> None:
        """Parallel t2s+sha for a batch of docs, then serial sqlite decisions."""
        nonlocal n_para, n_drop
        flat = [(did, "\n".join(paras)) for did, paras in buf]
        res = pool.map(_t2s_sha, flat, chunksize=64)
        # batch-fetch existing owners (PK lookups, one round trip per 500 shas)
        owners: dict[int, str] = {}
        all_hashes = [h for _, hs in res for h in hs]
        for i in range(0, len(all_hashes), 500):
            chunk = all_hashes[i:i + 500]
            qmarks = ",".join("?" * len(chunk))
            for sha, doc in db.execute(f"SELECT sha, doc FROM para WHERE sha IN ({qmarks})", chunk):
                owners[sha] = doc
        # batch-insert new rows (INSERT OR IGNORE keeps first owner)
        pending: list[tuple[int, str]] = []
        for did, hs in res:
            for h in hs:
                if h not in owners:
                    pending.append((h, did))
        for i in range(0, len(pending), 500):
            db.executemany(sql, pending[i:i + 500])
        pos = 0
        local_seen: dict[int, str] = {}
        for did, paras in buf:
            hashes = res[pos][1]
            pos += 1
            if len(hashes) != len(paras):
                raise RuntimeError(f"hash count mismatch for {did}")
            d = doc_by_id.get(did)
            kept = []
            for para, h in zip(paras, hashes):
                n_para += 1
                owner = owners.get(h, local_seen.get(h))
                if owner is not None and owner != did:
                    l3_buf.append({"doc_id": did, "source": d["source"], "dup_source": owner,
                                   "reason": "段落级精确重复(t2s归一)", "similarity": 1.0,
                                   "chars": len(para), "para_preview": para[:40]})
                    dropped_ids.add(did)
                    n_drop += 1
                    continue
                if h not in local_seen:
                    local_seen[h] = did
                kept.append(para)
            if len(l3_buf) >= L3_FLUSH:
                flush_audit()
                db.commit()
            src = d["source"]
            if src not in final_writers:
                out = FINAL_DIR / f"{src}.jsonl"
                out.parent.mkdir(parents=True, exist_ok=True)
                final_writers[src] = out.open(final_mode, encoding="utf-8")
            final_writers[src].write(json.dumps({"doc_id": did, "text": "\n".join(kept)}, ensure_ascii=False) + "\n")

    for src in PRIORITY_ALL:
        if src in done_srcs:
            continue
        p = CLEAN_DIR / f"{src}.jsonl"
        if not p.exists():
            continue
        prog = progress.get(src)
        skip_until = prog["last_doc"] if (prog and not prog.get("done")) else None
        if skip_until:
            # truncate final file to the checkpoint doc (inclusive); docs after
            # it are reprocessed so buffered l3 audit records get regenerated.
            fp = FINAL_DIR / f"{src}.jsonl"
            if fp.exists():
                tmp = fp.with_suffix(".tmp")
                with fp.open(encoding="utf-8") as fr, tmp.open("w", encoding="utf-8") as fw:
                    for line in fr:
                        rr = json.loads(line)
                        fw.write(line)
                        if rr["doc_id"] == skip_until:
                            break
                tmp.replace(fp)
                log.info("resume %s: truncated final to checkpoint %s", src, skip_until)
            final_mode = "a"
        else:
            # fresh source: remove any partial rows from an interrupted attempt
            db.execute("DELETE FROM para WHERE doc LIKE ?", (src + "|%",))
            db.commit()
            final_mode = "w"
        buf = []
        n_buf_paras = 0
        waiting = skip_until is not None
        last_doc = None
        n_docs = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if waiting:
                    # skip everything up to and including the checkpoint doc;
                    # only then start processing (file order is authoritative)
                    if r["doc_id"] == skip_until:
                        waiting = False
                    continue
                d = doc_by_id.get(r["doc_id"])
                if d is None or d.get("l1") == "dup" or d.get("l2") == "dup":
                    continue
                paras = split_paragraphs(r["text"])
                if not paras:
                    continue
                buf.append((r["doc_id"], paras))
                n_buf_paras += len(paras)
                last_doc = r["doc_id"]
                n_docs += 1
                if n_docs % 5000 == 0:
                    write_jsonl(progress_path, [{"src": src, "last_doc": last_doc, "done": False}])
                if n_buf_paras >= BATCH_PARAS:
                    flush_buf(buf)
                    buf = []
                    n_buf_paras = 0
        if buf:
            flush_buf(buf)
        flush_audit()
        db.commit()
        if last_doc:
            write_jsonl(progress_path, [{"src": src, "last_doc": last_doc, "done": True}])
        done_srcs.add(src)
        write_jsonl(done_path, sorted(done_srcs))
        log.info("L3 done %s (paras=%d, dropped=%d)", src, n_para, n_drop)
    pool.close()
    pool.join()
    for w in final_writers.values():
        w.close()
    db.close()
    for suffix in ("", "-wal", "-shm"):
        f = Path(str(DB) + suffix)
        f.unlink(missing_ok=True)
    log.info("L3 done; para sqlite removed")
    log.info("L3 total: %d paras, %d dropped paragraphs across %d docs", n_para, n_drop, len(dropped_ids))

    # dedupe l3 audit records (interrupted/buggy reruns may have appended duplicates)
    seen: set[tuple] = set()
    tmp_l3 = l3p.with_suffix(".tmp2")
    n_in = n_out = 0
    with l3p.open(encoding="utf-8") as fr, tmp_l3.open("w", encoding="utf-8") as fw:
        for line in fr:
            n_in += 1
            r = json.loads(line)
            key = (r["doc_id"], r.get("dup_source", ""), r.get("chars"), r.get("para_preview", ""))
            if key in seen:
                continue
            seen.add(key)
            fw.write(line)
            n_out += 1
    tmp_l3.replace(l3p)
    log.info("L3 audit dedup: %d -> %d records", n_in, n_out)

    # ---- L4 index (sqlite-backed window index, memory-safe) ----
    if (DEDUP_DIR / "l4_doc_near.jsonl").exists() and FINAL_MANIFEST.exists():
        log.info("L4 already done (l4 records + final manifest present), skipping")
        return
    win_db = sqlite3.connect(WIN_DB)
    win_db.execute("PRAGMA journal_mode=WAL")
    win_db.execute("PRAGMA synchronous=OFF")
    win_db.execute("PRAGMA cache_size=-200000")
    win_db.execute("CREATE TABLE IF NOT EXISTS win(hash INTEGER NOT NULL, doc TEXT NOT NULL, PRIMARY KEY(hash, doc))")
    win_db.execute("DELETE FROM win")
    n_win = 0
    batch = []
    for src in PRIORITY_ALL:
        p = FINAL_DIR / f"{src}.jsonl"
        if not p.exists():
            continue
        for r in stream_source(p):
            for w in windows_24(r["text"], WIN_STEP):
                h = int(sha1_text(w)[:15], 16)
                if h % WIN_SAMPLE == 0:
                    batch.append((h, r["doc_id"]))
                    n_win += 1
                    if len(batch) >= 20000:
                        win_db.executemany("INSERT OR IGNORE INTO win(hash, doc) VALUES (?, ?)", batch)
                        batch = []
        if batch:
            win_db.executemany("INSERT OR IGNORE INTO win(hash, doc) VALUES (?, ?)", batch)
            batch = []
    win_db.commit()
    log.info("L4 index: %d sampled windows", n_win)
    q = """SELECT a.doc AS da, b.doc AS db, COUNT(*) AS c
           FROM win a JOIN win b ON a.hash = b.hash AND a.doc < b.doc
           GROUP BY a.doc, b.doc HAVING c >= 8
           ORDER BY c DESC LIMIT ?"""
    pairs = [(da, db, c) for da, db, c in win_db.execute(q, (L4_TOPK,))]
    win_db.close()
    for f in (Path(str(WIN_DB) + s) for s in ("", "-wal", "-shm")):
        f.unlink(missing_ok=True)
    log.info("L4 candidate pairs: %d", len(pairs))

    # ---- L4/L3n on top pairs ----
    l4p, l3np = DEDUP_DIR / "l4_doc_near.jsonl", DEDUP_DIR / "l3n_para_near.jsonl"
    l4p.unlink(missing_ok=True)
    l3np.unlink(missing_ok=True)
    l4, l3n = [], []
    l4_ids: set[str] = set()
    l3n_paras: dict[str, list[dict]] = {}
    # preload texts of pair-involved docs per source
    want: dict[str, set[str]] = {}
    for a, b, _ in pairs:
        want.setdefault(a.split("|", 1)[0], set()).add(a)
        want.setdefault(b.split("|", 1)[0], set()).add(b)
    texts: dict[str, str] = {}
    for src, ids in want.items():
        p = FINAL_DIR / f"{src}.jsonl"
        for r in stream_source(p):
            if r["doc_id"] in ids:
                texts[r["doc_id"]] = r["text"]
    log.info("L4 pair texts loaded: %d docs", len(texts))

    for a, b, shared in pairs:
        if a in l4_ids or b in l4_ids:
            continue
        ta, tb = texts.get(a, ""), texts.get(b, "")
        wa = set(w for w in windows_24(ta, 4))
        wb = set(w for w in windows_24(tb, 4))
        inter, union = len(wa & wb), len(wa | wb)
        if union == 0:
            continue
        sim = inter / union
        if sim < 0.3:
            continue
        da, db = doc_by_id[a], doc_by_id[b]
        ai, bi = src_prio(da["source"]), src_prio(db["source"])
        if ai == bi:
            low, high = sorted([da, db], key=lambda x: x["doc_id"])
            same = True
        elif ai < bi:
            low, high = db, da
            same = False
        else:
            low, high = da, db
            same = False
        t_low = texts[low["doc_id"]]
        if sim >= (0.99 if same else 0.8):
            l4.append({"doc_id": low["doc_id"], "source": low["source"], "dup_source": high["doc_id"],
                       "reason": "文档级近重复(24字窗口Jaccard)", "similarity": round(sim, 4),
                       "chars": len(t_low)})
            l4_ids.add(low["doc_id"])
            continue
        hw = set(w for w in windows_24(texts[high["doc_id"]], 4))
        out = []
        for para in split_paragraphs(t_low):
            pw = list(windows_24(para, 4))
            if not pw:
                out.append(para)
                continue
            hit = sum(1 for w in pw if w in hw)
            if hit / len(pw) >= 0.9:
                l3n.append({"doc_id": low["doc_id"], "source": low["source"], "dup_source": high["doc_id"],
                            "reason": "段落级近重复(窗口覆盖>=0.9)", "similarity": round(hit / len(pw), 4),
                            "chars": len(para), "para_preview": para[:40]})
            else:
                out.append(para)
        new_text = "\n".join(out)
        if new_text != t_low:
            l3n_paras[low["doc_id"]] = new_text
    write_jsonl(l4p, l4)
    write_jsonl(l3np, l3n)
    log.info("L4 dropped docs: %d; L3n rewrites: %d docs, %d paras dropped",
             len(l4), len(l3n_paras), len(l3n))

    # ---- rewrite final jsonl per affected source (single pass each) ----
    affected: dict[str, set[str]] = {}
    for did in set(l3n_paras) | l4_ids:
        affected.setdefault(did.split("|", 1)[0], set()).add(did)
    for src, ids in affected.items():
        p = FINAL_DIR / f"{src}.jsonl"
        tmp = p.with_suffix(".tmp")
        with p.open(encoding="utf-8") as fr, tmp.open("w", encoding="utf-8") as fw:
            for line in fr:
                rr = json.loads(line)
                if rr["doc_id"] in l4_ids:
                    continue
                if rr["doc_id"] in l3n_paras:
                    rr["text"] = l3n_paras[rr["doc_id"]]
                fw.write(json.dumps(rr, ensure_ascii=False) + "\n")
        tmp.replace(p)
        log.info("rewrote final %s", src)

    # ---- final manifest (chars from audit arithmetic) ----
    l3_by_doc: dict[str, int] = {}
    for r in stream_source(l3p):
        l3_by_doc[r["doc_id"]] = l3_by_doc.get(r["doc_id"], 0) + int(r["chars"])
    l3n_by_doc: dict[str, int] = {}
    for r in stream_source(l3np):
        l3n_by_doc[r["doc_id"]] = l3n_by_doc.get(r["doc_id"], 0) + int(r["chars"])
    final_docs = []
    for d in clean_docs:
        if d.get("l1") == "dup" or d.get("l2") == "dup" or d["doc_id"] in l4_ids:
            continue
        d["chars_final"] = max(int(d.get("chars_raw", 0)) - l3_by_doc.get(d["doc_id"], 0)
                               - l3n_by_doc.get(d["doc_id"], 0), 0)
        d["final_status"] = "kept"
        final_docs.append(d)
    write_jsonl(FINAL_MANIFEST, final_docs)
    log.info("FINAL: %d docs, chars=%s", len(final_docs), f"{sum(d['chars_final'] for d in final_docs):,}")


if __name__ == "__main__":
    main()
