#!/usr/bin/env python3
"""Shared helpers for the Jishui data production pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset/raw"
INTERIM = ROOT / "dataset/interim"
QUARANTINE = ROOT / "dataset/quarantine"
PROCESSED = ROOT / "dataset/processed"
REPORTS = ROOT / "dataset/reports/pipeline"
LOGS = ROOT / "dataset/reports/pipeline/logs"

DOC_MANIFEST = INTERIM / "manifest" / "docs.jsonl"
CLEAN_MANIFEST = INTERIM / "manifest" / "docs_clean_norm.jsonl"
FINAL_MANIFEST = INTERIM / "manifest" / "docs_final.jsonl"
DEDUP_DIR = INTERIM / "dedup"

TOKENIZER_DIR = ROOT / "tokenizer/ccc-bbpe-32k"

PRIORITY_ALL = ["ccc_corpus", "ccc_punctuate", "wanli", "greathangpt", "mingqing",
                "poetry", "erya", "daizhige", "wikisource", "gutenberg"]

CAT_NAMES = {
    1: "经史子集及散文",
    2: "小说笔记戏曲",
    3: "诗词曲赋",
    4: "佛道医",
    5: "晚清民国文学",
    6: "知识与公共文本",
}

ERA_NAMES = {
    "ancient": "上古(～220)",
    "medieval": "中古(220–907)",
    "early_modern": "近世(907–1368)",
    "ming_qing": "明清(1368–1840)",
    "late_qing": "晚清(1840–1911)",
    "roc": "民国(1912–1945)",
    "unknown": "未定",
}


def setup_logging(name: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, records, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]")


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    removed = len(_CJK_RE.sub("", text))
    return 1.0 - removed / len(text)


_KEEP_STR = "\n\r\t 　，。；：！？、（）《》「」『』〈〉【】—…·・"
_SYM_RE = re.compile(r"[^\w" + re.escape(_KEEP_STR) + r"]")


def symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    return 1.0 - len(_SYM_RE.sub("", text)) / len(text)


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def strip_control(text: str) -> str:
    return _CTRL_RE.sub("", text)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def fix_utf8(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    return text.replace("\ufffd", "")


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)


_YEARS = re.compile(r"(?:18|19|20)\d{2}年|民國(\d{1,3})年|(?:18|19|20)\d{2}")


def extract_years(text: str, window: str = "head") -> list[int]:
    """Extract explicit years from text. window: head/tail/all -> slice."""
    if window == "head":
        text = text[:3000]
    elif window == "tail":
        text = text[-1500:]
    years = []
    for m in _YEARS.finditer(text):
        y = m.group(0).rstrip("年")
        if y.startswith("民國"):
            years.append(1911 + int(y[2:]))
        else:
            years.append(int(y))
    return sorted(set(years))
