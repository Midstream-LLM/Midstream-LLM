#!/usr/bin/env python3
"""Stage 3: clean every doc -> dataset/interim/clean/{source}.jsonl
(one line per kept doc: {"doc_id": ..., "text": ...}).

- source-specific cleaners (gutenberg header/footer, wiki markup, ming-qing
  page headers, poetry json, ccc, daizhige, erya, wanli, greathangpt)
- wikisource text-based 1945 gate + category inference from title
- universal quality filters (min length, cjk ratio, symbol ratio, junk)
- every dropped doc -> dataset/quarantine/ (JSONL, never silent)
- guards: per-source output bytes must stay <= 4x its raw chars, else abort
- output: dataset/interim/manifest/docs_clean.jsonl (atomic rewrite)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (DOC_MANIFEST, INTERIM, QUARANTINE, RAW, cjk_ratio,
                    normalize_newlines, read_jsonl, setup_logging, sha1_text,
                    strip_control, symbol_ratio, write_jsonl)

log = setup_logging("03_clean")
CLEAN_DIR = INTERIM / "clean"
OUT_MANIFEST = INTERIM / "manifest" / "docs_clean.jsonl"
QUAR = QUARANTINE / "rejected.jsonl"
BYTE_GUARD = 4.0

_WIKI_CONV = re.compile(r"-{([^{}]*?)}-")
_WIKI_TAGS = re.compile(r"<[^>]{1,60}>")
_GUT_START = re.compile(r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}", re.S)
_GUT_END = re.compile(r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.S)
_MQ_HEADER = re.compile(r"^(钦定四库全书|四库全书|集部|子部|史部|经部|提要|【明】|【清】|御制|御定)")
_MQ_SPACED = re.compile(r"^\s*(?:[一二三四五六七八九十百千]+)[集子史经部]\s*$")

_WI_CAT4 = re.compile(r"(佛說|佛經|金剛|般若|華嚴|法華|涅槃|楞嚴|維摩|阿彌陀|藥師|地藏|無量壽|大藏|禪宗|心經|楞伽|圓覺|瑜伽師地|道德經|南華|沖虛|抱朴|參同契|黃庭|道藏|本草|黃帝內經|素問|靈樞|難經|傷寒|金匱|針灸|脈經|千金|外臺)")
_WI_CAT3 = re.compile(r"(詩經|楚辭|樂府|全唐詩|全宋詞|元曲|詩集|詞集|詩話|詞話|曲譜|詩品|文心雕龍|聲律啟蒙|笠翁對韻|千家詩|唐詩|宋詩|詩鈔|詞鈔|賦集|駢文)")
_WI_CAT2 = re.compile(r"(小說|筆記|演義|話本|笑話|寶卷|傳奇|雜劇|南戲|彈詞|說部|世說|聊齋|紅樓|三國|水滸|西遊|金瓶梅|儒林外史|鏡花緣|老殘|官場現形|孽海花|海上花|西廂|牡丹亭|長生殿|桃花扇|琵琶記|竇娥|漢宮秋|閱微草堂)")
_WI_CAT5 = re.compile(r"(魯迅|呐喊|彷徨|朝花夕拾|野草|墳|熱風|華蓋|而已集|兩地書|梁啟超|飲冰室|嚴復|天演論|林紓|李伯元|吳趼人|劉鶚|曾樸|蘇曼殊|郁達夫|朱自清|背影|荷塘月色|徐志摩|老舍|沈從文|巴金|茅盾|張恨水|孽海花|老殘遊記|官場現形記|二十年目睹|海上花列傳|駱駝祥子|阿Q|狂人日記|祝福|傷逝|邊城|家|春|秋|雷雨|日出|茶館)")
_WI_CAT6 = re.compile(r"(教材|講義|教科書|辭典|字典|年鑑|條約|章程|公報|法令|法律|法規|憲法|新聞|報紙|社論|格致|算學|幾何|代數|聲學|光學|化學|物理|地學|博物|鐵路|電報|郵政|銀行|通商|度量衡|地理|歷史|國文|修身|啟蒙|三字經|百家姓|千字文|弟子規|增廣賢文|幼學瓊林|龍文鞭影|日知錄|通典|通志|文獻通考|會典|實錄|邸鈔|邸報|奏議|詔令|國策|戰國策|水經注|山海經|爾雅|說文|方言|釋名|廣韻|集韻|字彙|正字通)")
_WI_MODERN = re.compile(r"(中华人民共和国|中国共产党|习近平|毛泽东|邓小平|江泽民|胡锦涛|温家宝|李克强|改革开放|文化大革命|香港回归|澳门回归|一国两制|全国人民代表大会|国务院|最高人民法院|中国共产党中央|中国特色社会主义|三中全会|政协)")
_WI_ROCI = re.compile(r"中華民國(\d{1,3})年")
_WI_YEAR = re.compile(r"(?:18|19|20)\d{2}年")


def clean_wikisource(title: str, text: str) -> tuple[str, str, str, str]:
    """Return (clean_text, category, time_status, time_reason)."""
    text = _WIKI_CONV.sub(lambda m: m.group(1), text)
    text = _WIKI_TAGS.sub("", text)
    head = text[:3000] + text[-1500:]
    years = []
    for m in _WI_YEAR.finditer(head):
        years.append(int(m.group(0)[:-1]))
    for m in _WI_ROCI.finditer(head):
        years.append(1911 + int(m.group(1)))
    years = sorted(set(years))
    cat = "1"
    for rx, c in ((_WI_CAT4, "4"), (_WI_CAT3, "3"), (_WI_CAT2, "2"), (_WI_CAT5, "5"), (_WI_CAT6, "6")):
        if rx.search(title):
            cat = c
            break
    if _WI_MODERN.search(title + text[:2000]):
        return "", cat, "reject", "现代内容标记"
    late = [y for y in years if y >= 1946]
    if late:
        return "", cat, "reject", f"文本含1946年后年份 {late[:5]}"
    ok = [y for y in years if 1900 <= y <= 1945]
    if ok:
        return text, cat, "pass", f"文本年份证据 {ok[:5]}"
    return text, cat, "pass", "无年份标记，按古文本处理"


def clean_gutenberg(text: str) -> str:
    text = _GUT_START.sub("", text)
    text = _GUT_END.sub("", text)
    return "\n".join(l for l in text.split("\n") if not re.match(r"^\s*Produced by\b", l))


def clean_mingqing(text: str) -> str:
    out = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if _MQ_HEADER.match(s) or _MQ_SPACED.match(s):
            continue
        out.append(s)
    return "\n".join(out)


def clean_poetry(text: str) -> str:
    try:
        data = json.loads(text)
    except Exception:
        return text
    parts = []
    if isinstance(data, dict):
        data = [data]
    for poem in data:
        if not isinstance(poem, dict):
            continue
        title = poem.get("title") or ""
        author = poem.get("author") or ""
        paras = poem.get("paragraphs") or poem.get("strains") or []
        if isinstance(paras, str):
            paras = [paras]
        head = f"《{title}》 {author}".strip() if title else ""
        body = "\n".join(paras)
        parts.append(f"{head}\n{body}" if head else body)
    return "\n\n".join(parts)


# ---------- text access (streamed once per source) ----------

_LOADED: dict[tuple, dict] = {}  # (src, path) -> {id: text}
_ERYA_HANDLES: dict[str, list] = {}  # path -> [fh, done_chunks, seen]
_WANLI_CACHE: dict[str, str] = {}


def _preload_jsonl(path: str, id_key: str = "id", text_key: str = "text") -> dict:
    d: dict = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        d[r[id_key]] = r.get(text_key) or ""
    return d


def _preload_parquet(path: str, id_key: str = "id", text_key: str = "text") -> dict:
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=[id_key, text_key])
    return {r[id_key]: (r[text_key] or "") for r in t.to_pylist()}


def _lookup(src: str, key: str, path: str = "") -> str:
    cache_key = (src, path)
    if cache_key not in _LOADED:
        if src in ("ccc_corpus", "ccc_punctuate"):
            norm = RAW.parent / "normalized"
            name = "ccc_corpus.jsonl" if src == "ccc_corpus" else "ccc_punctuate.jsonl"
            keyname = "content" if src == "ccc_corpus" else "output"
            _LOADED[cache_key] = _preload_jsonl(str(norm / name), "id", keyname)
        elif src == "mingqing":
            _LOADED[cache_key] = _preload_jsonl(path, "id", "text")
        elif src in ("greathangpt", "wikisource"):
            _LOADED[cache_key] = _preload_parquet(path, "id", "text")
        else:
            _LOADED[cache_key] = {}
    return _LOADED[cache_key].get(key, "")


def _wanli_text() -> dict:
    if _WANLI_CACHE:
        return _WANLI_CACHE
    p = RAW / "wanli-dibao-corpus/wanli_shilu_corpus.jsonl"
    groups: dict[str, list[str]] = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        key = f"{r.get('year')}-{int(r.get('month') or 0):02d}"
        t = (r.get("raw_text") or "").strip()
        if t:
            groups.setdefault(key, []).append(t)
    for k, v in groups.items():
        _WANLI_CACHE[k] = "\n".join(v)
    return _WANLI_CACHE


def _erya_chunk(path: str, chunk: int) -> str:
    CHUNK = 100
    if path not in _ERYA_HANDLES:
        _ERYA_HANDLES[path] = [open(path, encoding="utf-8", errors="replace"), 0, 0]
    fh, _, seen = _ERYA_HANDLES[path]
    while seen < (chunk - 1) * CHUNK:
        line = fh.readline()
        if not line:
            break
        if line.strip():
            seen += 1
    lines = []
    while len(lines) < CHUNK:
        line = fh.readline()
        if not line:
            break
        if line.strip():
            lines.append(line)
    _ERYA_HANDLES[path] = [fh, 0, seen + len(lines)]
    return "".join(lines)


def source_text(doc: dict) -> str:
    src = doc["source"]
    if src == "daizhige":
        return Path(doc["path"]).read_text(encoding="utf-8", errors="replace")
    if src in ("ccc_corpus", "ccc_punctuate"):
        return _lookup(src, doc["doc_id"].split("|", 1)[1], doc["path"])
    if src == "mingqing":
        return _lookup(src, doc["doc_id"].split("|", 1)[1], doc["path"])
    if src == "greathangpt":
        return _lookup(src, doc["doc_id"].split("|", 2)[2], doc["path"])
    if src == "wikisource":
        return _lookup(src, doc["doc_id"].split("|", 1)[1], doc["path"])
    if src == "poetry":
        return Path(doc["path"]).read_text(encoding="utf-8", errors="replace")
    if src == "gutenberg":
        return Path(doc["path"]).read_text(encoding="utf-8", errors="replace")
    if src == "erya":
        _, name, chunk = doc["doc_id"].split("|")
        p = RAW / f"erya-dataset/extracted/monolingual/{name}"
        return _erya_chunk(str(p), int(chunk.replace("chunk", "")))
    if src == "wanli":
        return _wanli_text().get(doc["doc_id"].split("|", 1)[1], "")
    return ""


def safe_name(doc_id: str) -> str:
    return doc_id.replace("/", "__").replace("|", "_")


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    QUAR.parent.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    dropped: list[dict] = []
    counters: dict[str, int] = {}
    writers: dict[str, object] = {}
    src_bytes: dict[str, int] = {}
    src_raw: dict[str, int] = {}

    for i, doc in enumerate(read_jsonl(DOC_MANIFEST)):
        if i % 20000 == 0 and i > 0:
            log.info("progress %d docs (kept=%d dropped=%d)", i, len(kept), len(dropped))
        if doc.get("parse_status") != "ok":
            dropped.append({"doc_id": doc["doc_id"], "source": doc["source"], "reason": doc.get("error", "parse error")})
            continue
        src = doc["source"]
        src_raw[src] = src_raw.get(src, 0) + doc.get("chars_raw", 0)
        if doc.get("time_status") == "reject":
            dropped.append({"doc_id": doc["doc_id"], "source": src, "reason": doc.get("time_reason", "reject")})
            continue
        try:
            raw = source_text(doc)
        except Exception as e:
            log.error("read fail %s: %s", doc["doc_id"], e)
            dropped.append({"doc_id": doc["doc_id"], "source": src, "reason": f"read error: {e}"})
            continue
        text = strip_control(normalize_newlines(raw))

        cat, time_status, time_reason = doc["category"], doc.get("time_status"), doc.get("time_reason")
        if src == "wikisource":
            text, cat, time_status, time_reason = clean_wikisource(doc["title"], text)
        elif src == "gutenberg":
            text = clean_gutenberg(text)
        elif src == "mingqing":
            text = clean_mingqing(text)
        elif src == "poetry":
            text = clean_poetry(text)
        text = text.strip()

        if time_status == "reject":
            dropped.append({"doc_id": doc["doc_id"], "source": src, "reason": time_reason})
            continue
        if len(text) < 50:
            dropped.append({"doc_id": doc["doc_id"], "source": src, "reason": "短文档(<50字)", "chars": len(text)})
            continue
        cjk = cjk_ratio(text)
        sym = symbol_ratio(text)
        if cjk < 0.55:
            dropped.append({"doc_id": doc["doc_id"], "source": src, "reason": f"中文比例过低({cjk:.2f})", "chars": len(text)})
            continue
        if sym > 0.25:
            dropped.append({"doc_id": doc["doc_id"], "source": src, "reason": f"符号占比过高({sym:.2f})", "chars": len(text)})
            continue
        if "http://" in text or "www." in text:
            dropped.append({"doc_id": doc["doc_id"], "source": src, "reason": "含URL/网络残留", "chars": len(text)})
            continue

        rec = dict(doc)
        rec.update({
            "category": cat, "time_status": time_status, "time_reason": time_reason,
            "chars_clean": len(text), "sha1_clean": sha1_text(text),
            "cjk_ratio": round(cjk, 4), "clean_status": "ok",
        })
        if src not in writers:
            out = CLEAN_DIR / f"{src}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            writers[src] = out.open("w", encoding="utf-8")
        writers[src].write(json.dumps({"doc_id": doc["doc_id"], "text": text}, ensure_ascii=False) + "\n")
        src_bytes[src] = src_bytes.get(src, 0) + len(text) * 3 + 64
        if src_bytes[src] > BYTE_GUARD * max(src_raw.get(src, 1), 1):
            raise RuntimeError(
                f"ABORT: source {src} output {src_bytes[src]} bytes > {BYTE_GUARD}x raw {src_raw[src]}")
        kept.append(rec)
        counters[src] = counters.get(src, 0) + 1

    for w in writers.values():
        w.close()
    write_jsonl(OUT_MANIFEST, kept)
    write_jsonl(QUAR, dropped, append=False)
    log.info("kept=%d dropped=%d", len(kept), len(dropped))
    log.info("by source: %s", counters)
    log.info("output bytes by source: %s", src_bytes)


if __name__ == "__main__":
    main()
