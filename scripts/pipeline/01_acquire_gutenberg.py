#!/usr/bin/env python3
"""Acquire Project Gutenberg Chinese-language books (all pre-1945 candidates).

Step 1: parse /browse/languages/zh -> book ids
Step 2: fetch each ebook page, extract title/author/author-death/original-publication
Step 3: time-gate (original pub <= 1945; fallback: all authors dead <= 1945)
Step 4: download UTF-8 text for passed books into dataset/raw/gutenberg-zh/

Resumable: skips ids already in books.jsonl. Logs every failure.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

RAW = Path("dataset/raw/gutenberg-zh")
MANIFEST = RAW / "books.jsonl"
TEXT_DIR = RAW / "text"
UA = {"User-Agent": "jishui-research/1.0 (classical chinese LLM pretraining)"}


def fetch(url: str, binary: bool = False, retries: int = 3) -> bytes:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def parse_book_page(book_id: str) -> dict:
    url = f"https://www.gutenberg.org/ebooks/{book_id}"
    html = fetch(url).decode("utf-8", "ignore")
    info = {"id": book_id, "url": url}
    m = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.S)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1))
        info["title"] = re.sub(r"\s+", " ", title).strip()
        m2 = re.search(r"^(.*?)\s+by\s+([^,]+)", info["title"], re.S)
        if m2 and " by " in info["title"]:
            info["title"] = m2.group(1).strip()
            if not info.get("authors"):
                info["authors"] = [m2.group(2).strip()]
    authors = []
    died = []
    for m in re.finditer(r'itemprop="creator">([^<]+?)</a>', html):
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        name = re.sub(r",\s*(?:[0-9]+|\[[^\]]+\])\s*[-–—]\s*(?:[0-9]+|\[[^\]]+\])", "", name).strip()
        authors.append(name)
        for dm in re.finditer(r"[-–—]\s*(\d{3,4})\s*(BCE|CE)?", m.group(1)):
            year, era = int(dm.group(1)), dm.group(2) or ""
            if era == "BCE":
                year = -year
            died.append(year)
    info["authors"] = authors
    info["author_death_years"] = died
    op = re.search(r"Original Publication\s*</td>\s*<td[^>]*>\s*(.*?)</td>", html, re.S)
    info["original_publication"] = re.sub(r"\s+", " ", op.group(1)).strip() if op else None
    rd = re.search(r"Release Date\s*</td>\s*<td[^>]*>\s*([^<]+)", html, re.S)
    info["release_date"] = rd.group(1).strip() if rd else None
    langs = re.findall(r'<a rel="dcterms:language"[^>]*>([^<]+)</a>', html)
    info["languages"] = [re.sub(r"\s+", " ", l).strip() for l in langs]
    return info


def gate(info: dict) -> tuple[str, str]:
    """Return (status, reason). status in {pass, quarantine}."""
    op = (info.get("original_publication") or "").lower()
    years = [int(y) for y in re.findall(r"(?:19|18|17|16|15)\d{2}", op)]
    if years and max(years) <= 1945:
        return "pass", f"original_publication={op}"
    death = info.get("author_death_years") or []
    if death and max(death) <= 1945:
        return "pass", f"authors died <=1945: {death}"
    if death and max(death) > 1945:
        return "quarantine", f"author died {max(death)} > 1945"
    return "quarantine", "unknown original publication; author death year unknown"


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(re.findall(r'ebooks/(\d+)', fetch("https://www.gutenberg.org/browse/languages/zh").decode("utf-8", "ignore"))))
    print(f"[gut] {len(ids)} chinese book ids found", flush=True)
    seen = set()
    if MANIFEST.exists():
        for line in MANIFEST.open():
            seen.add(json.loads(line)["id"])
    ok = 0
    for book_id in ids:
        if book_id in seen:
            continue
        try:
            info = parse_book_page(book_id)
        except Exception as e:
            print(f"[gut] FAIL page {book_id}: {e}", flush=True)
            continue
        info["status"], info["reason"] = gate(info)
        with MANIFEST.open("a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
        print(f"[gut] {book_id} {info.get('title','?')[:40]} -> {info['status']} ({info['reason'][:50]})", flush=True)
        ok += 1
        time.sleep(0.8)
    print(f"[gut] metadata done, new={ok}, total={len(ids)}", flush=True)

    # download texts for passed books
    passed = [json.loads(l) for l in MANIFEST.open() if json.loads(l).get("status") == "pass"]
    for info in passed:
        out = TEXT_DIR / f"{info['id']}.txt"
        if out.exists() and out.stat().st_size > 500:
            continue
        try:
            data = fetch(f"https://www.gutenberg.org/cache/epub/{info['id']}/pg{info['id']}.txt")
            if len(data) < 500:
                raise RuntimeError("empty download")
            out.write_bytes(data)
            print(f"[gut] DL {info['id']} {len(data)//1024}KB", flush=True)
        except Exception as e:
            print(f"[gut] FAIL dl {info['id']}: {e}", flush=True)
        time.sleep(0.8)


if __name__ == "__main__":
    sys.exit(main())
