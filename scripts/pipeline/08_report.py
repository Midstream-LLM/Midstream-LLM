#!/usr/bin/env python3
"""Stage 8: final statistics, sampling weights, gap analysis and markdown report.

Streams all big inputs (memory-safe on 16GB machines).
Reads inventory/clean/dedup/split/tokenize outputs -> reports/pipeline/*
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (CAT_NAMES, CLEAN_MANIFEST, DEDUP_DIR, DOC_MANIFEST,
                    FINAL_MANIFEST, PROCESSED, QUARANTINE, REPORTS, read_jsonl,
                    setup_logging, write_jsonl)

log = setup_logging("08_report")

TARGET_MIX = {"1": 0.35, "2": 0.15, "3": 0.08, "4": 0.12, "5": 0.15, "6": 0.15}
TARGET_TOTAL_TOKENS = 6_000_000_000
SOURCES = ["ccc_corpus", "ccc_punctuate", "wanli", "greathangpt", "mingqing",
           "poetry", "erya", "daizhige", "wikisource", "gutenberg"]


def stream(path: Path):
    yield from read_jsonl(path)


def count_lines(path: Path) -> int:
    n = 0
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for _ in f:
                n += 1
    return n


def main() -> None:
    stats: dict = {}

    # 1/2/3/13/14/15: single pass over inventory
    n_raw: Counter = Counter()
    n_ok: Counter = Counter()
    chars_raw_n: defaultdict = defaultdict(int)
    tg: Counter = Counter()
    lic: Counter = Counter()
    n_inv = 0
    parse_err = 0
    cjk_min = 1.0
    for d in stream(DOC_MANIFEST):
        n_inv += 1
        s = d["source"]
        n_raw[s] += 1
        if d.get("parse_status") == "ok":
            n_ok[s] += 1
        else:
            parse_err += 1
        chars_raw_n[s] += int(d.get("chars_raw", 0) or 0)
        tg[(s, d.get("time_status", "?"))] += 1
        lic[d.get("license", "unknown")] += 1
        cjk_min = min(cjk_min, float(d.get("cjk_ratio", 1) or 1))
    stats["1_raw_file_counts"] = dict(n_raw)
    stats["2_parsed_counts"] = dict(n_ok)
    stats["4_chars_before_clean"] = {k: int(v) for k, v in chars_raw_n.items()}
    stats["13_time_gate_by_source"] = {f"{s}/{st}": int(n) for (s, st), n in sorted(tg.items())}
    stats["14_license_counts"] = dict(lic)

    # 3: excluded by reason (stream quarantine)
    drop_reasons: Counter = Counter()
    fffd = 0
    n_quar = 0
    for q in stream(QUARANTINE / "rejected.jsonl"):
        n_quar += 1
        reason = q.get("reason", "unknown")
        drop_reasons[str(reason)[:40]] += 1
        if "FFFD" in reason:
            fffd += 1
    stats["3_excluded_by_reason"] = dict(drop_reasons.most_common())

    # 5: chars after clean (stream clean manifest for doc counts)
    n_clean = 0
    for _ in stream(CLEAN_MANIFEST):
        n_clean += 1

    # 6/7/11/12: final + audit records (stream)
    chars_final: defaultdict = defaultdict(int)
    final_cat: dict[str, str] = {}
    n_final = 0
    for d in stream(FINAL_MANIFEST):
        chars_final[d["source"]] += int(d.get("chars_final", 0) or 0)
        final_cat[d["doc_id"]] = d.get("category", "?") or "?"
        n_final += 1

    l3_chars: defaultdict = defaultdict(int)
    l3_matrix: defaultdict = defaultdict(int)
    l3_cat_matrix: defaultdict = defaultdict(int)
    n_l3 = 0
    for r in stream(DEDUP_DIR / "l3_para_exact.jsonl"):
        n_l3 += 1
        c = int(r.get("chars", 0) or 0)
        l3_chars[r["source"]] += c
        dsrc = (r.get("dup_source") or "").split("|", 1)[0]
        l3_matrix[(r["source"], dsrc)] += c
        ca = final_cat.get(r["doc_id"], "?")
        cb = final_cat.get(r.get("dup_source", ""), "?")
        l3_cat_matrix[(ca, cb)] += c
    l4 = list(stream(DEDUP_DIR / "l4_doc_near.jsonl"))
    l3n = list(stream(DEDUP_DIR / "l3n_para_near.jsonl"))
    l3n_chars: defaultdict = defaultdict(int)
    l3n_matrix: defaultdict = defaultdict(int)
    l3n_cat_matrix: defaultdict = defaultdict(int)
    for r in l3n:
        c = int(r.get("chars", 0) or 0)
        l3n_chars[r["source"]] += c
        dsrc = (r.get("dup_source") or "").split("|", 1)[0]
        l3n_matrix[(r["source"], dsrc)] += c
        l3n_cat_matrix[(final_cat.get(r["doc_id"], "?"), final_cat.get(r.get("dup_source", ""), "?"))] += c

    chars_exact = defaultdict(int, chars_final)
    for r in l3n + l4:
        chars_exact[r["source"]] += int(r.get("chars", 0) or 0)
    for s, c in l3_chars.items():
        chars_exact[s] += c
    # clean chars = after-exact + l1/l2 exact-duplicate removals
    l1 = list(stream(DEDUP_DIR / "l1_file_exact.jsonl"))
    l2 = list(stream(DEDUP_DIR / "l2_doc_exact.jsonl"))
    chars_clean = defaultdict(int, chars_exact)
    for r in l1 + l2:
        if r.get("dup_source"):
            chars_clean[r["source"]] += int(r.get("chars", 0) or 0)
    stats["5_chars_after_clean"] = {k: int(v) for k, v in chars_clean.items()}
    stats["6_chars_after_exact_dedup"] = {k: int(v) for k, v in chars_exact.items()}
    stats["7_chars_after_near_dedup"] = {k: int(v) for k, v in chars_final.items()}

    matrix = defaultdict(int)
    for (a, b), c in l3_matrix.items():
        matrix[(a, b)] += c
    for (a, b), c in l3n_matrix.items():
        matrix[(a, b)] += c
    stats["11_source_overlap_matrix"] = {f"{a}->{b}": int(v) for (a, b), v in sorted(matrix.items())}

    cat_matrix = defaultdict(int)
    for (a, b), c in l3_cat_matrix.items():
        cat_matrix[(a, b)] += c
    for (a, b), c in l3n_cat_matrix.items():
        cat_matrix[(a, b)] += c
    stats["12_category_overlap"] = {f"{a}->{b}": int(v) for (a, b), v in sorted(cat_matrix.items())}

    # 8/9/10: token stats (stream docs_tokens)
    tok_by_source: defaultdict = defaultdict(int)
    tok_by_cat: defaultdict = defaultdict(int)
    split_tokens: defaultdict = defaultdict(int)
    n_tok_docs = 0
    for r in stream(PROCESSED / "manifest" / "docs_tokens.jsonl"):
        n_tok_docs += 1
        n = int(r.get("tokens", 0) or 0)
        tok_by_source[r.get("source", "?")] += n
        tok_by_cat[r.get("category") or "?"] += n
        split_tokens[r.get("split", "?")] += n
    total_tok = int(sum(tok_by_source.values()))
    stats["8_total_tokens_bbpe32k"] = total_tok
    stats["9_tokens_by_source"] = {k: int(v) for k, v in tok_by_source.items()}
    stats["10_tokens_by_category"] = {k: int(v) for k, v in tok_by_cat.items()}

    # 15: quality
    stats["15_quality"] = {
        "cjk_ratio_min": round(cjk_min, 4),
        "docs_with_fffd": fffd,
        "avg_doc_chars_final": round(sum(chars_final.values()) / max(n_final, 1)),
        "dropped_parse_errors": parse_err,
    }

    stats["dedup_counts"] = {
        "l1_file_exact": count_lines(DEDUP_DIR / "l1_file_exact.jsonl"),
        "l2_doc_exact": count_lines(DEDUP_DIR / "l2_doc_exact.jsonl"),
        "l3_para_exact": n_l3,
        "l3n_para_near": len(l3n),
        "l4_doc_near": len(l4),
    }
    stats["docs"] = {"inventory": n_inv, "kept_clean": n_clean,
                     "kept_final": n_final, "tokenized": n_tok_docs,
                     "quarantine": n_quar}

    # split stats
    split_counts = {}
    for s in ("train", "val", "test"):
        split_counts[s] = count_lines(PROCESSED / "split" / f"{s}.jsonl")
    stats["split_counts"] = split_counts
    stats["split_tokens"] = dict(split_tokens)

    # sampling weights + gap
    actual_frac = {c: tok_by_cat.get(c, 0) / max(total_tok, 1) for c in TARGET_MIX}
    weights = {}
    for c, tgt in TARGET_MIX.items():
        act = actual_frac.get(c, 0)
        w = tgt / act if act > 0 else 0.0
        weights[c] = round(min(max(w, 0.01), 100.0), 4)
    stats["sampling_weights_by_category"] = weights
    stats["gap_tokens_by_category"] = {
        c: int(tok_by_cat.get(c, 0) - TARGET_MIX[c] * TARGET_TOTAL_TOKENS) for c in TARGET_MIX}
    stats["gap_total_tokens_vs_6B"] = int(total_tok - TARGET_TOTAL_TOKENS)

    write_jsonl(REPORTS / "token_stats.json", [stats])

    # markdown report
    lines = ["# Jishui 数据生产报告", "",
             f"生成时间: 2026-08-07 | 阶段: inventory→clean→dedup→split→tokenize", "",
             "## 总览",
             f"- 原始文档/文件数: {n_inv:,}（各源 {dict(n_raw)}）",
             f"- 解析成功: {n_inv - parse_err:,} / {n_inv:,} | 排除: {n_quar:,}",
             f"- 清洗后保留: {n_clean:,} | 去重后保留: {n_final:,} | 已 tokenize: {n_tok_docs:,}",
             f"- 最终字符数: {sum(chars_final.values()):,} | BBPE32k tokens: {total_tok:,}",
             "",
             "## 1–7 字符流水统计（按源）",
             "| 源 | 原始 | 解析 | 清洗前字符 | 清洗后字符 | 精确去重后 | 近重去重后 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in SOURCES:
        lines.append(f"| {s} | {n_raw.get(s,0):,} | {n_ok.get(s,0):,} | {chars_raw_n.get(s,0):,} | "
                     f"{chars_clean.get(s,0):,} | {chars_exact.get(s,0):,} | {chars_final.get(s,0):,} |")
    lines += ["", "## 8–10 Token 统计（ccc-bbpe-32k 实测编码）",
              "| 源 | tokens |", "|---|---:|"]
    for s in SOURCES:
        lines.append(f"| {s} | {tok_by_source.get(s,0):,} |")
    lines.append("")
    lines.append("按类别：")
    lines.append("| 类 | 名称 | tokens | 占比 | 目标占比 | 缺口 vs 6B | 权重 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in "123456":
        act = tok_by_cat.get(c, 0)
        lines.append(f"| {c} | {CAT_NAMES[int(c)]} | {act:,} | {act/max(total_tok,1)*100:.1f}% | "
                     f"{TARGET_MIX[c]*100:.0f}% | {stats['gap_tokens_by_category'][c]:+,} | "
                     f"{weights[c]} |")
    lines += ["", "## 11–12 重复矩阵（被删字符，按 源→重复源）", "```"]
    for k, v in stats["11_source_overlap_matrix"].items():
        lines.append(f"{k}: {v:,}")
    lines += ["```", "## 13 1945 时间门控（按 源/状态）", "```"]
    for k, v in stats["13_time_gate_by_source"].items():
        lines.append(f"{k}: {v:,}")
    lines += ["```", "## 14 许可证", "```"]
    for k, v in sorted(stats["14_license_counts"].items()):
        lines.append(f"{k}: {v}")
    lines += ["```", "## 15 质量", "```"]
    for k, v in stats["15_quality"].items():
        lines.append(f"{k}: {v}")
    lines += ["```", "## 去重明细", "```"]
    for k, v in stats["dedup_counts"].items():
        lines.append(f"{k}: {v}")
    lines += ["```", "## 划分", "```"]
    for k, v in split_counts.items():
        lines.append(f"{k}: {v} docs / {split_tokens.get(k,0):,} tokens")
    lines += ["```", "## 已知问题与人工审核事项", "",
              "1. wikisource 年代门控基于标题与头尾年份标记启发式，`待审核`与误判需人工抽查。",
              "2. ccc_corpus/punctuate 为现代点校本（现代标点），按规则标记为整理本保留正文。",
              "3. erya monolingual 按 100 行切块，块边界不保证篇章完整。",
              "4. 大汉书 未分类 语料全部进入待审核（时间与类别未知）。",
              "5. L4 只检查了采样窗口命中>=8 的前 12000 对文档，覆盖面为采样近似。",
              "6. 类别 5/6 数据源有限（万历邸钞 262 个月档、wikisource 少量），缺口大。",
              "7. Gutenberg 无保留文档（时间门控/来源问题，待核查）。",
              "8. 现代标点版 CCC punctuate 的 output 字段含跨句粘连（首字符为句号）等小瑕疵。"]
    report = "\n".join(lines) + "\n"
    (REPORTS / "report.md").write_text(report, encoding="utf-8")
    log.info("report written; total tokens=%d; weights=%s", total_tok, weights)


if __name__ == "__main__":
    main()
