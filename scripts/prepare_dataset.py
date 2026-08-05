#!/usr/bin/env python3
"""Jishui (集水) dataset preparation tooling.

Subcommands:
  manifest   build dataset/manifests/daizhigev20.jsonl + chinese-classical-corpus.jsonl
  normalize  build dataset/normalized/ccc_{corpus,translate,punctuate}.jsonl
  reports    build dataset/reports/ccc_{schema,script,missing_char,overlap_daizhige}_report.json

Reads only dataset/raw/. Never writes into raw/.
Daizhige text: UTF-8 (often BOM-prefixed), traditional script, U+3000 fullwidth
spaces used as indentation/segment separators -- never stripped on the raw side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
RAW = DATASET / "raw"
DAIZHIGE = RAW / "daizhigev20"
CCC = RAW / "chinese-classical-corpus"
MANIFESTS = DATASET / "manifests"
NORMALIZED = DATASET / "normalized"
REPORTS = DATASET / "reports"

CCC_FILES = ("corpus.jsonl", "translate.jsonl", "punctuate.jsonl")
NORMALIZED_NAMES = {
    "corpus.jsonl": "ccc_corpus.jsonl",
    "translate.jsonl": "ccc_translate.jsonl",
    "punctuate.jsonl": "ccc_punctuate.jsonl",
}
TEXT_FIELDS = ("content", "input", "output", "instruction")
CHAR_FIELDS = ("content", "input", "output")


def _punct_class() -> str:
    """CJK + ASCII punctuation set (unicode category P) from relevant blocks."""
    blocks = (
        range(0x0021, 0x007F),
        range(0x2000, 0x2070),
        range(0x2E80, 0x9FFF),
        range(0xFE10, 0xFE20),
        range(0xFE30, 0xFE50),
        range(0xFF00, 0xFFF0),
    )
    chars = set()
    for blk in blocks:
        for cp in blk:
            ch = chr(cp)
            if unicodedata.category(ch).startswith("P"):
                chars.add(ch)
    return "".join(sorted(chars))


_PUNCT = _punct_class()
_PUNCT_WS_RE = re.compile(f"[{re.escape(_PUNCT)}\\s]+")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")


def md5_int(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


def iter_records(fname: str):
    with (CCC / fname).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                yield i, {"__json_error__": line[:200]}


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    out, prev_blank = [], False
    for ln in s.split("\n"):
        ln = ln.rstrip(" \t\u3000")
        blank = ln == ""
        if blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = blank
    return "\n".join(out).strip(" \t\u3000\n")


# ---------------------------------------------------------------- manifest
TEXT_KINDS = {"txt", "src", "tgt", "jsonl", "json", "md", "log", "srt"}
EXTRA_DATASETS = [
    "chinese-poetry",
    "ming-qing-wenji-corpus",
    "erya-dataset",
    "greathangpt-classical-chinese",
]


def cmd_extra(args: argparse.Namespace) -> None:
    """Inventory the extra raw datasets: file-level manifest + lightweight
    report (sizes, char counts, unique chars, json/jsonl/parquet schemas)."""
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in EXTRA_DATASETS:
        root = RAW / name
        if not root.is_dir():
            log(f"skip {name}: not found")
            continue
        entries = []
        total_size = total_chars = 0
        uniq = set()
        kinds = Counter()
        for p in sorted(root.rglob("*")):
            if not p.is_file() or ".git" in p.parts:
                continue
            rel = p.relative_to(root)
            kind = p.suffix.lower().lstrip(".") or "none"
            entry = {
                "path": rel.as_posix(),
                "kind": kind,
                "size_bytes": p.stat().st_size,
                "line_count": None,
                "char_count": None,
            }
            kinds[kind] += 1
            total_size += entry["size_bytes"]
            if kind in TEXT_KINDS:
                text = read_text(p)
                entry["line_count"] = text.count("\n") + (0 if text.endswith("\n") else 1) if text else 0
                entry["char_count"] = len(text)
                total_chars += len(text)
                uniq.update(set(text))
            entries.append(entry)
        with (MANIFESTS / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for e in entries:
                out.write(json.dumps(e, ensure_ascii=False) + "\n")
        ds = {
            "files": len(entries),
            "total_size_bytes": total_size,
            "total_chars": total_chars,
            "unique_chars": len(uniq),
            "kinds": dict(kinds),
        }
        schema = detect_schemas(root)
        if schema:
            ds["schemas"] = schema
        pq_chars = sum_parquet_chars(root)
        if pq_chars:
            ds["parquet_chars"] = pq_chars
        report[name] = ds
        log(f"{name}: {len(entries)} files, {total_size/2**30:.2f} GB, "
            f"{total_chars:,} chars, {len(uniq):,} unique chars")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "extra_datasets_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_schemas(root: Path) -> dict:
    """Sample-based schema detection for jsonl / json-array / parquet files."""
    out = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        kind = p.suffix.lower().lstrip(".")
        if kind == "jsonl":
            keys, count = set(), 0
            with p.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    keys.update(rec.keys())
                    count += 1
                    if count >= 5000:
                        break
            out.setdefault("jsonl", []).append(
                {"file": p.name, "sampled_records": count, "fields": sorted(keys)})
        elif kind == "json":
            try:
                data = json.loads(read_text(p))
            except json.JSONDecodeError:
                continue
            if isinstance(data, list) and data and isinstance(data[0], dict):
                out.setdefault("json_arrays", []).append(
                    {"file": p.name, "records_in_file": len(data), "fields": sorted(data[0].keys())})
        elif kind == "parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                log("pyarrow not installed; skipping parquet schemas")
                break
            md = pq.read_metadata(p)
            out.setdefault("parquet", []).append(
                {"file": p.name, "rows": md.num_rows,
                 "columns": [md.schema.column(i).name for i in range(md.num_columns)]})
    return out


def sum_parquet_chars(root: Path) -> int:
    total = 0
    for p in sorted(root.rglob("*.parquet")):
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(p, columns=["char_count"])
            total += sum(t.column("char_count").to_pylist())
        except Exception:
            continue
    return total


def cmd_manifest(args: argparse.Namespace) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    log("scanning daizhigev20 ...")
    n_txt = n_lines = n_chars = 0
    with (MANIFESTS / "daizhigev20.jsonl").open("w", encoding="utf-8") as out:
        for p in sorted(DAIZHIGE.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(DAIZHIGE)
            entry = {
                "path": rel.as_posix(),
                "category": rel.parts[0],
                "size_bytes": p.stat().st_size,
                "line_count": None,
                "char_count": None,
            }
            if p.suffix.lower() == ".txt":
                text = read_text(p)
                entry["line_count"] = text.count("\n") + (0 if text.endswith("\n") else 1)
                entry["char_count"] = len(text)
                n_txt += 1
                n_lines += entry["line_count"]
                n_chars += entry["char_count"]
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log(f"daizhigev20: {n_txt} txt files, {n_lines:,} lines, {n_chars:,} chars")

    log("scanning chinese-classical-corpus ...")
    n_records = 0
    with (MANIFESTS / "chinese-classical-corpus.jsonl").open("w", encoding="utf-8") as out:
        for fname in CCC_FILES:
            for idx, rec in iter_records(fname):
                if "__json_error__" in rec:
                    continue
                body = "".join(str(rec.get(k) or "") for k in CHAR_FIELDS)
                entry = {
                    "file": fname,
                    "idx": idx,
                    "id": rec.get("id"),
                    "source": rec.get("source"),
                    "sha1": hashlib.sha1(body.encode("utf-8")).hexdigest(),
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_records += 1
    log(f"chinese-classical-corpus: {n_records:,} records")


# ---------------------------------------------------------------- normalize
def cmd_normalize(args: argparse.Namespace) -> None:
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    for fname in CCC_FILES:
        outname = NORMALIZED_NAMES[fname]
        seen, dropped, kept = set(), 0, 0
        with (NORMALIZED / outname).open("w", encoding="utf-8") as out:
            for _, rec in iter_records(fname):
                if "__json_error__" in rec:
                    continue
                rid = rec.get("id")
                if rid is not None:
                    if rid in seen:
                        dropped += 1
                        continue
                    seen.add(rid)
                for k in TEXT_FIELDS:
                    if isinstance(rec.get(k), str):
                        rec[k] = clean_text(rec[k])
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
        log(f"{outname}: kept {kept:,}, dropped {dropped:,} duplicate ids")


# ---------------------------------------------------------------- reports
WINDOW_LEN = 24
WINDOW_STEP = 4
SAMPLE_K = 64


def daizhige_window_keys(text: str, t2s) -> set:
    """Sampled 24-char window keys (md5 of t2s-converted, punct/ws-stripped
    window). Both corpora are script-heterogeneous, so windows are converted to
    simplified before hashing. Step-1 window positions on this side, so any
    substring window present in the text is guaranteed to be visited regardless
    of its alignment; sampling (builtin hash % SAMPLE_K) bounds set size."""
    stripped = _PUNCT_WS_RE.sub("", text)
    keys = set()
    n = len(stripped) - WINDOW_LEN + 1
    for i in range(n):
        w = stripped[i : i + WINDOW_LEN]
        if hash(w) % SAMPLE_K == 0:
            keys.add(md5_int(t2s.convert(w)))
    return keys


def scan_daizhige(t2s):
    """One pass over daizhigev20: char frequencies (per-chunk-unique approx),
    sampled window keys for overlap detection, line/char totals."""
    counts = Counter()
    window_keys = set()
    files = lines = chars = 0
    for p in sorted(DAIZHIGE.rglob("*.txt")):
        text = read_text(p)
        files += 1
        lines += text.count("\n") + (0 if text.endswith("\n") else 1)
        chars += len(text)
        counts.update(set(text))
        window_keys |= daizhige_window_keys(text, t2s)
        if files % 5000 == 0:
            log(f"  daizhigev20 scanned {files:,} files")
    log(f"daizhigev20: {files:,} files, {lines:,} lines, {chars:,} chars, "
        f"{len(window_keys):,} sampled window keys")
    return counts, window_keys, files, lines, chars


def ccc_window_stats(text: str, t2s, keys: set):
    """Per-record overlap stats: (windows_sampled_eligible, windows_matched,
    first_match). Only sampled-eligible windows cost a lookup."""
    stripped = _PUNCT_WS_RE.sub("", text)
    n = len(stripped) - WINDOW_LEN + 1
    eligible = matched = 0
    first = None
    for i in range(0, n, WINDOW_STEP):
        w = stripped[i : i + WINDOW_LEN]
        if hash(w) % SAMPLE_K != 0:
            continue
        eligible += 1
        if md5_int(t2s.convert(w)) in keys:
            matched += 1
            if first is None:
                first = w
    return eligible, matched, first


def daizhige_has_book(source: str, basenames: set) -> bool:
    book = source.split("/")[0].split("·")[0].strip()
    if not book or len(book) < 2:
        return None
    for b in basenames:
        if book in b or b in book:
            return True
    return False


def schema_stats(rec, stats, fidx):
    fields = stats.setdefault("_field_order", [])
    for k, v in rec.items():
        if k not in stats["fields"]:
            stats["fields"][k] = {"present": 0, "empty": 0, "types": Counter(), "total_chars": 0}
            fields.append(k)
        f = stats["fields"][k]
        f["present"] += 1
        f["types"][type(v).__name__] += 1
        if isinstance(v, str):
            if not v:
                f["empty"] += 1
            else:
                f["total_chars"] += len(v)
                f.setdefault("lengths", []).append(len(v))
    rid = rec.get("id")
    if rid is not None:
        ids = stats.setdefault("_ids", set())
        dups = stats.setdefault("_dups", [])
        if rid in ids:
            stats["_dup_count"] += 1
            if len(dups) < 20:
                dups.append(rid)
        else:
            ids.add(rid)


def quantiles(lengths, qs=(0.5, 0.9, 0.99)):
    if not lengths:
        return {"p50": 0, "p90": 0, "p99": 0}
    s = sorted(lengths)
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = s[min(len(s) - 1, int(q * (len(s) - 1)))]
    return out


def classify_script(text: str, s2t, t2s) -> str:
    changed_s2t = s2t.convert(text) != text
    changed_t2s = t2s.convert(text) != text
    if changed_s2t and changed_t2s:
        return "mixed"
    if changed_s2t:
        return "simplified"
    if changed_t2s:
        return "traditional"
    return "identical"


def cmd_reports(args: argparse.Namespace) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    from opencc import OpenCC

    s2t, t2s = OpenCC("s2t"), OpenCC("t2s")

    log("pass 1/2: scanning daizhigev20 (chars + window keys) ...")
    dz_counts, dz_keys, dz_files, dz_lines, dz_chars = scan_daizhige(t2s)
    dz_basenames = {p.stem for p in DAIZHIGE.rglob("*.txt")}

    schema = {"total_chars_ccc": 0, "files": {}}
    script = {"overall": Counter(), "files": {}}
    ccc_counts = Counter()
    overlap = {
        "method": {
            "window_len": WINDOW_LEN,
            "daizhige_window_step": 1,
            "ccc_window_step": WINDOW_STEP,
            "sample_k": SAMPLE_K,
            "normalization": "strip whitespace+CJK punct, t2s-convert both sides, md5-int window keys",
            "note": "overlap fraction is estimated as matched/(eligible sampled windows); "
                    "daizhige char frequencies are per-chunk-unique approximations",
        },
        "daizhige": {"txt_files": dz_files, "lines": dz_lines, "chars": dz_chars,
                     "sampled_window_keys": len(dz_keys)},
        "files": {},
    }

    log("pass 2/2: scanning chinese-classical-corpus ...")
    for fname in CCC_FILES:
        n_records = n_json_errors = n_dup_ids = 0
        stats = {
            "fields": {},
            "_field_order": [], "_ids": set(), "_dups": [], "_dup_count": 0,
        }
        script_counts, script_examples = Counter(), {}
        char_counts = Counter()
        win_eligible = win_matched = records_with_match = too_few = 0
        per_source = {}
        match_examples = []

        for idx, rec in iter_records(fname):
            n_records += 1
            if "__json_error__" in rec:
                n_json_errors += 1
                continue
            schema_stats(rec, stats, idx)
            text = " ".join(str(rec.get(k) or "") for k in CHAR_FIELDS)
            if text:
                char_counts.update(text)
                cls = classify_script(text, s2t, t2s)
                script_counts[cls] += 1
                if cls not in script_examples:
                    script_examples[cls] = rec.get("id")
                eligible, matched, first = ccc_window_stats(text, t2s, dz_keys)
                win_eligible += eligible
                win_matched += matched
                if eligible == 0:
                    too_few += 1
                elif matched:
                    records_with_match += 1
                    if len(match_examples) < 10 and first:
                        match_examples.append(first)
                    src = rec.get("source") or ""
                    ps = per_source.setdefault(src, {"records": 0, "eligible": 0, "matched": 0})
                    ps["records"] += 1
                    ps["eligible"] += eligible
                    ps["matched"] += matched

        ccc_counts.update(char_counts)
        schema["total_chars_ccc"] += sum(char_counts.values())

        field_summary = {}
        for k in stats["_field_order"]:
            f = stats["fields"][k]
            summary = {
                "present": f["present"],
                "empty": f["empty"],
                "types": dict(f["types"]),
                "total_chars": f["total_chars"],
            }
            if "lengths" in f:
                lens = f["lengths"]
                summary.update({
                    "min_chars": min(lens), "max_chars": max(lens),
                    "mean_chars": round(sum(lens) / len(lens), 1),
                    **quantiles(lens),
                })
            field_summary[k] = summary

        schema["files"][fname] = {
            "record_count": n_records,
            "json_errors": n_json_errors,
            "unique_ids": len(stats["_ids"]),
            "duplicate_ids": stats["_dup_count"],
            "duplicate_id_examples": stats["_dups"][:10],
            "fields": field_summary,
        }

        script["files"][fname] = {
            "counts": dict(script_counts),
            "example_ids": script_examples,
        }
        script["overall"].update(script_counts)

        per_source_out = [
            {
                "source": src,
                "records": ps["records"],
                "eligible_windows": ps["eligible"],
                "matched_windows": ps["matched"],
                "est_overlap_fraction": round(ps["matched"] / ps["eligible"], 6)
                if ps["eligible"] else 0.0,
                "daizhige_book_file": daizhige_has_book(src, dz_basenames),
            }
            for src, ps in sorted(per_source.items(), key=lambda x: -x[1]["matched"])
        ]
        overlap["files"][fname] = {
            "records": n_records,
            "eligible_windows": win_eligible,
            "matched_windows": win_matched,
            "est_overlap_fraction": round(win_matched / win_eligible, 6) if win_eligible else 0.0,
            "records_with_any_match": records_with_match,
            "records_without_windows": too_few,
            "per_source": per_source_out,
            "example_matches": match_examples,
        }
        log(f"  {fname}: {n_records:,} records, est overlap "
            f"{win_matched / win_eligible:.1%}" if win_eligible else f"  {fname}: no windows")

    script["overall"] = dict(script["overall"])

    schema_report = {
        "source_files": list(CCC_FILES),
        "total_chars_ccc": schema["total_chars_ccc"],
        "files": schema["files"],
    }
    (REPORTS / "ccc_schema_report.json").write_text(
        json.dumps(schema_report, ensure_ascii=False, indent=2), encoding="utf-8")

    (REPORTS / "ccc_script_report.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

    dz_missing_ccc = {ch: n for ch, n in dz_counts.items() if ch not in ccc_counts}
    ccc_missing_dz = {ch: n for ch, n in ccc_counts.items() if ch not in dz_counts}
    missing_report = {
        "ccc": {"total_chars": sum(ccc_counts.values()), "unique_chars": len(ccc_counts)},
        "daizhige": {"total_chars": dz_chars, "unique_chars": len(dz_counts)},
        "chars_in_daizhige_missing_from_ccc": {
            "count": len(dz_missing_ccc),
            "top_200": [{"char": ch, "freq": n} for ch, n in
                        sorted(dz_missing_ccc.items(), key=lambda x: -x[1])[:200]],
        },
        "chars_in_ccc_missing_from_daizhige": {
            "count": len(ccc_missing_dz),
            "top_200": [{"char": ch, "freq": n} for ch, n in
                        sorted(ccc_missing_dz.items(), key=lambda x: -x[1])[:200]],
        },
        "union_unique_chars": len(dz_counts.keys() | ccc_counts.keys()),
        "note": "daizhige char frequencies are approximations (counted once per chunk of unique chars); CCC frequencies are exact.",
    }
    (REPORTS / "ccc_missing_char_report.json").write_text(
        json.dumps(missing_report, ensure_ascii=False, indent=2), encoding="utf-8")

    (REPORTS / "ccc_overlap_daizhige_report.json").write_text(
        json.dumps(overlap, ensure_ascii=False, indent=2), encoding="utf-8")

    log("reports written to dataset/reports/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest", help="build manifests")
    sub.add_parser("extra", help="inventory extra raw datasets (manifest + report)")
    sub.add_parser("normalize", help="build normalized ccc files")
    sub.add_parser("reports", help="build schema/script/missing-char/overlap reports")
    args = ap.parse_args()
    {"manifest": cmd_manifest, "extra": cmd_extra, "normalize": cmd_normalize,
     "reports": cmd_reports}[args.cmd](args)


if __name__ == "__main__":
    main()
