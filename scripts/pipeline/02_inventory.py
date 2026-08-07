#!/usr/bin/env python3
"""Stage 2: inventory all raw sources -> unified doc manifest (metadata only).

Each iterator yields {"doc": {...meta...}, "text": "..."} streaming the source
ONCE. parse_doc does hashing/metrics only (no I/O). Records go to
dataset/interim/manifest/docs.jsonl (atomic rewrite = idempotent).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (RAW, DOC_MANIFEST, cjk_ratio, fix_utf8, normalize_newlines,
                    sha1_text, setup_logging, write_jsonl)

log = setup_logging("02_inventory")

CAT1, CAT2, CAT3, CAT4, CAT5, CAT6 = "1", "2", "3", "4", "5", "6"
NORM = RAW.parent / "normalized"

# ---------- classification helpers ----------

_FO_BUDDHA = re.compile(r"(佛說|佛經|金剛|般若|華嚴|法華|涅槃|楞嚴|維摩|阿彌陀|藥師|地藏|無量壽|觀無量壽|大藏|禪|瑜伽師地|四分律|十誦律|比丘|菩薩|如來|阿含|圓覺|楞伽|勝鬘|仁王|盂蘭|普門|心經|法句)")
_FO_DAO = re.compile(r"(道德經|道藏|太上|洞玄|洞真|太清|黃庭|南華|沖虛|文始|抱朴|參同契|雲笈|靈寶|上清)")
_CAT3_POET = re.compile(r"(詩|詞|曲|賦|樂府|楚辭|詩話|詞話|詩集|詞集|曲譜|詩評|詩品|文心雕龍)")
_CAT2_FIC = re.compile(r"(小說|筆記|演義|話本|笑話|寶卷|謎語|傳奇|雜劇|南戲|彈詞|鼓詞|說部|世說|聊齋|紅樓|三國|水滸|西遊|金瓶梅|儒林外史|鏡花緣|老殘|官場現形|二十年目睹|孽海花|海上花|醒世姻緣|封神|東周列國|隋唐演義|楊家將|西廂|牡丹亭|長生殿|桃花扇|琵琶記|竇娥|漢宮秋)")
_CAT5_MINGUO = re.compile(r"(魯迅|呐喊|彷徨|朝花夕拾|野草|故事新編|墳|熱風|華蓋|而已集|三閒|二心|南腔北調|偽自由書|准風月談|花邊文學|且介亭|集外集|兩地書|梁啟超|飲冰室|章太炎|嚴復|天演論|林紓|李伯元|吳趼人|劉鶚|曾樸|蘇曼殊|郁達夫|朱自清|背影|荷塘月色|徐志摩|冰心|老舍|沈從文|巴金|茅盾|張恨水|孽海花|老殘遊記|官場現形記|二十年目睹之怪現狀|海上花列傳|茶館|駱駝祥子|阿Q|故鄉|社戲|狂人日記|藥|祝福|傷逝|雷雨|日出|邊城)")
_CAT6_KNOW = re.compile(r"(教材|講義|教科書|辭典|字典|年鑑|年鑒|條約|章程|公報|法令|法律|法規|憲法|報|新聞|社論|格致|算學|幾何|代數|聲學|光學|化學|物理|地學|博物|農|桑|棉|蠶|鐵路|輪船|電報|郵政|銀行|通商|稅|度量衡|地圖|地理|歷史|國文|修身|啟蒙|蒙學|三字經|百家姓|千字文|弟子規|增廣賢文|幼學瓊林|龍文鞭影|聲律啟蒙|笠翁對韻|日知錄|通典|通志|文獻通考|會典|實錄|邸鈔|邸報|奏議|詔令|國策|戰國策)")
_WIKI_JUNK = re.compile(r"(作者:|分類:|Author:|Category:|Portal:|Template:|模板:|Help:|說明:|討論:|Talk:|Wikisource|维基文库|來源請求|待審|請求|存檔|索引:|圍棋|編輯|首頁|首页)")


def read_text_bytes(p: Path) -> str:
    return normalize_newlines(fix_utf8(p.read_bytes()))


def iter_daizhige():
    root = RAW / "daizhigev20"
    for p in sorted(root.rglob("*.txt")):
        if p.name.startswith("._"):
            continue
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        zang = parts[0]
        cat = {"史藏": CAT1, "儒藏": CAT1, "易藏": CAT1, "艺藏": CAT1}.get(zang)
        if cat is None:
            sub = parts[1] if len(parts) > 1 else ""
            if zang == "诗藏":
                cat = CAT2 if sub == "剧曲" else CAT3
            elif zang == "子藏":
                cat = CAT2 if sub == "笔记" else CAT1
            elif zang == "集藏":
                cat = CAT2 if sub in ("小说", "演义", "话本", "笑话", "宝卷", "谜语") else CAT1
            elif zang == "佛藏":
                cat = CAT4
            elif zang == "道藏":
                cat = CAT4
            elif zang == "医藏":
                cat = CAT4
            else:
                cat = CAT1
        try:
            text = read_text_bytes(p)
        except Exception as e:
            yield {"doc": {"doc_id": f"daizhige|{rel}", "source": "daizhige", "error": f"read: {e}"}}
            continue
        yield {
            "doc": {"doc_id": f"daizhige|{rel}", "source": "daizhige", "work": rel,
                    "title": p.stem, "author": "", "category": cat, "secondary": "",
                    "era": "unknown", "path": str(p), "license": "pd-research",
                    "time_evidence": "古代典籍", "time_status": "pass", "time_reason": "古代典籍"},
            "text": text,
        }


def _ccc_loader():
    cache: dict[str, dict] = {}
    for name, key in (("ccc_corpus.jsonl", "content"), ("ccc_punctuate.jsonl", "output"),
                      ("ccc_translate.jsonl", None)):
        src = "ccc_corpus" if name.startswith("ccc_corpus") else "ccc_punctuate" if name.startswith("ccc_punctuate") else "ccc_translate"
        d = {}
        for line in open(NORM / name, encoding="utf-8"):
            r = json.loads(line)
            if src == "ccc_translate":
                text = (r.get("input") or "") + "\n" + (r.get("output") or "")
            else:
                text = r.get(key) or ""
            r["_text"] = text
            d[r["id"]] = r
        cache[src] = d
    return cache


def iter_ccc(cache):
    for src, d in cache.items():
        if src == "ccc_translate":
            continue
        for rid, r in d.items():
            yield {
                "doc": {"doc_id": f"{src}|{rid}", "source": src, "work": r.get("source") or rid,
                        "title": f"{r.get('source','')} {r.get('chapter','')}".strip(),
                        "author": r.get("author", ""), "category": CAT1, "secondary": "",
                        "era": "unknown", "path": str(NORM / ("ccc_corpus.jsonl" if src == "ccc_corpus" else "ccc_punctuate.jsonl")),
                        "license": "pd-research", "time_evidence": "古代典籍(点校本,现代标点)",
                        "time_status": "pass", "time_reason": "古代典籍"},
                "text": r["_text"],
            }


def iter_ccc_translate(cache):
    for rid, r in cache["ccc_translate"].items():
        yield {
            "doc": {"doc_id": f"ccc_translate|{rid}", "source": "ccc_translate",
                    "work": r.get("source") or rid, "title": f"{r.get('source','')} {r.get('task','')}",
                    "author": "", "category": "", "secondary": "", "era": "unknown",
                    "path": str(NORM / "ccc_translate.jsonl"), "license": "pd-research",
                    "time_evidence": "现代白话译文(后加翻译)", "time_status": "reject",
                    "time_reason": "现代白话译文(后加翻译)"},
            "text": r["_text"],
        }


def iter_poetry():
    root = RAW / "chinese-poetry"
    for p in sorted(root.rglob("*.json")):
        if p.name.startswith("._") or ".git" in p.parts or p.name in ("README",):
            continue
        name = p.relative_to(root).as_posix()
        cat = CAT1 if any(s in name for s in ("蒙学", "四书五经")) else CAT3
        try:
            text = read_text_bytes(p)
        except Exception as e:
            yield {"doc": {"doc_id": f"poetry|{name}", "source": "poetry", "error": f"read: {e}"}}
            continue
        yield {
            "doc": {"doc_id": f"poetry|{name}", "source": "poetry", "work": name,
                    "title": name, "author": "", "category": cat, "secondary": "",
                    "era": "unknown", "path": str(p), "license": "pd-research",
                    "time_evidence": "古代诗词总集", "time_status": "pass", "time_reason": "古代诗词总集"},
            "text": text,
        }


def iter_mingqing():
    p = RAW / "ming-qing-wenji-corpus/data/ming_qing_wenji_corpus.jsonl"
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        yield {
            "doc": {"doc_id": f"mingqing|{r['id']}", "source": "mingqing",
                    "work": r["collection"], "title": r["collection"],
                    "author": r.get("author", ""), "category": CAT1, "secondary": "",
                    "era": "ming_qing" if r.get("dynasty") == "明" else "late_qing",
                    "path": str(p), "license": "pd-research",
                    "time_evidence": f"{r.get('dynasty')}文集(四库本)", "time_status": "pass",
                    "time_reason": "明清文集"},
            "text": r.get("text") or "",
        }


def iter_erya():
    CHUNK = 100
    for name in ("train.src", "valid.src"):
        p = RAW / "erya-dataset/extracted/monolingual" / name
        chunk = 0
        buf = []
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if len(buf) >= CHUNK:
                    chunk += 1
                    yield {
                        "doc": {"doc_id": f"erya|{name}|chunk{chunk}", "source": "erya",
                                "work": f"erya-{name}", "title": f"erya-monolingual-{name} chunk{chunk}",
                                "author": "", "category": CAT1, "secondary": "", "era": "unknown",
                                "path": str(p), "license": "pd-research",
                                "time_evidence": "上古至宋史料(编年体)", "time_status": "pass",
                                "time_reason": "古代史料"},
                        "text": "\n".join(buf),
                    }
                    buf = []
                buf.append(line)
        if buf:
            chunk += 1
            yield {
                "doc": {"doc_id": f"erya|{name}|chunk{chunk}", "source": "erya",
                        "work": f"erya-{name}", "title": f"erya-monolingual-{name} chunk{chunk}",
                        "author": "", "category": CAT1, "secondary": "", "era": "unknown",
                        "path": str(p), "license": "pd-research",
                        "time_evidence": "上古至宋史料(编年体)", "time_status": "pass",
                        "time_reason": "古代史料"},
                "text": "\n".join(buf),
            }


def iter_greathangpt():
    import pyarrow.parquet as pq
    root = RAW / "greathangpt-classical-chinese"
    era_map = {"先秦": "ancient", "汉魏": "ancient", "隋唐": "medieval",
               "两宋": "early_modern", "明清": "ming_qing", "佛经": "unknown", "未分类": "unknown"}
    for p in sorted(root.glob("*.parquet")):
        if p.name in ("preview.parquet", "all.parquet"):
            continue
        t = pq.read_table(p, columns=["id", "title", "author", "dynasty", "era", "content_category", "text"])
        name = p.stem
        for row in t.to_pylist():
            cat = CAT4 if row.get("content_category") == "佛" else CAT1
            era = era_map.get(name, "unknown")
            if name == "明清" and row.get("dynasty") in ("清末民国初", "清末近现代初"):
                cat, era = CAT5, "late_qing"
            if name == "未分类":
                cat, era = "", "unknown"
            yield {
                "doc": {"doc_id": f"greathangpt|{name}|{row['id']}", "source": "greathangpt",
                        "work": row.get("title") or row["id"], "title": row.get("title") or "",
                        "author": row.get("author", "") or "", "category": cat, "secondary": "",
                        "era": era, "path": str(p), "license": "pd-research",
                        "time_evidence": f"大汉书({name})",
                        "time_status": "quarantine" if cat == "" else "pass",
                        "time_reason": "未分类/无年代证据" if cat == "" else "大汉书时代分片"},
                "text": row.get("text") or "",
            }


def iter_wanli():
    p = RAW / "wanli-dibao-corpus/wanli_shilu_corpus.jsonl"
    groups: dict[str, dict] = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        key = f"{r.get('year')}-{int(r.get('month') or 0):02d}"
        g = groups.setdefault(key, {"texts": [], "title": ""})
        g["texts"].append((r.get("raw_text") or "").strip())
        m = re.search(r"卷(\d+)", r.get("title", ""))
        g["title"] = f"万历邸钞{'-卷' + m.group(1) if m else ''}"
    for key, g in sorted(groups.items()):
        yield {
            "doc": {"doc_id": f"wanli|{key}", "source": "wanli", "work": g["title"],
                    "title": f"{g['title']} {key}", "author": "", "category": CAT6,
                    "secondary": CAT1, "era": "ming_qing", "path": str(p),
                    "license": "cc-by-4.0", "time_evidence": f"万历{key.split('-')[0]}年",
                    "time_status": "pass", "time_reason": "万历年间(1573-1620)"},
            "text": "\n".join(g["texts"]),
        }


def iter_wikisource():
    import pyarrow.parquet as pq
    root = RAW / "wikisource-zh/20231201.zh"
    for p in sorted(root.glob("*.parquet")):
        t = pq.read_table(p, columns=["id", "title", "text"])
        for row in t.to_pylist():
            title = (row["title"] or "").strip()
            if not title or _WIKI_JUNK.search(title) or len(title) > 120:
                continue
            yield {
                "doc": {"doc_id": f"wikisource|{row['id']}", "source": "wikisource",
                        "work": title.split("/")[0], "title": title, "author": "",
                        "category": "", "secondary": "", "era": "unknown", "path": str(p),
                        "license": "pd-or-ccbysa", "time_evidence": "待文本审核",
                        "time_status": "pending_text", "time_reason": "需文本审核(年代标记检测)"},
                "text": row.get("text") or "",
            }


def iter_gutenberg():
    p = RAW / "gutenberg-zh/books.jsonl"
    if not p.exists():
        return
    seen = set()
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") != "pass":
            continue
        bid = r["id"]
        if bid in seen:
            continue
        seen.add(bid)
        tp = RAW / f"gutenberg-zh/text/{bid}.txt"
        if not tp.exists():
            continue
        title = (r.get("title") or "").strip() or bid
        cat, sec = CAT1, ""
        if _CAT5_MINGUO.search(title) or (r.get("author_death_years") and max(r["author_death_years"]) > 1840):
            cat = CAT5
        elif _FO_BUDDHA.search(title) or _FO_DAO.search(title):
            cat = CAT4
        elif _CAT2_FIC.search(title):
            cat = CAT2
        elif _CAT3_POET.search(title):
            cat = CAT3
        elif _CAT6_KNOW.search(title):
            cat = CAT6
        death = r.get("author_death_years") or []
        era = "unknown"
        if death:
            d = max(death)
            era = "roc" if 1912 <= d <= 1945 else "late_qing" if 1840 <= d < 1912 else "ming_qing" if d < 1840 else "unknown"
        try:
            text = read_text_bytes(tp)
        except Exception as e:
            yield {"doc": {"doc_id": f"gutenberg|{bid}", "source": "gutenberg", "error": f"read: {e}"}}
            continue
        yield {
            "doc": {"doc_id": f"gutenberg|{bid}", "source": "gutenberg", "work": title,
                    "title": title, "author": "、".join(r.get("authors") or []), "category": cat,
                    "secondary": sec, "era": era, "path": str(tp), "license": "pd-us",
                    "time_evidence": r.get("reason", ""), "time_status": "pass",
                    "time_reason": r.get("reason", "")},
            "text": text,
        }


SOURCES = [
    ("daizhige", iter_daizhige),
    ("ccc_corpus", None),
    ("ccc_translate", None),
    ("poetry", iter_poetry),
    ("mingqing", iter_mingqing),
    ("erya", iter_erya),
    ("greathangpt", iter_greathangpt),
    ("wanli", iter_wanli),
    ("wikisource", iter_wikisource),
    ("gutenberg", iter_gutenberg),
]


def parse_doc(doc: dict, text: str) -> dict:
    if doc.get("error"):
        doc["parse_status"] = "error"
        doc["chars_raw"] = 0
        return doc
    text = normalize_newlines(text)
    doc["chars_raw"] = len(text)
    doc["sha1_raw"] = sha1_text(text)
    doc["cjk_ratio"] = round(cjk_ratio(text), 4) if text else 0.0
    doc["parse_status"] = "ok"
    return doc


def main() -> None:
    only = sys.argv[1:]
    recs = []
    n_err = 0
    ccc_cache = _ccc_loader()
    log.info("ccc loader: %s", {k: len(v) for k, v in ccc_cache.items()})
    for src, it in SOURCES:
        if only and src not in only:
            continue
        if src == "ccc_corpus":
            it = lambda: iter_ccc(ccc_cache)
        elif src == "ccc_translate":
            it = lambda: iter_ccc_translate(ccc_cache)
        src_n = 0
        for item in it():
            recs.append(parse_doc(item["doc"], item.get("text", "")))
            src_n += 1
            if src_n % 20000 == 0:
                log.info("%s: %d parsed (running)", src, src_n)
            if recs[-1]["parse_status"] == "error":
                n_err += 1
                log.error("%s parse error: %s", recs[-1]["doc_id"], recs[-1].get("error"))
        log.info("%s: %d docs", src, src_n)
    write_jsonl(DOC_MANIFEST, recs)
    log.info("total docs=%d, errors=%d, chars=%s",
             len(recs), n_err, f"{sum(r.get('chars_raw', 0) for r in recs):,}")


if __name__ == "__main__":
    main()
