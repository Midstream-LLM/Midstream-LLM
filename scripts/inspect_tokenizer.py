#!/usr/bin/env python3
"""Inspect a byte-level BPE tokenizer vocabulary in readable form.

Byte-level BPE vocab keys are raw UTF-8 byte strings (each byte mapped to a
Unicode char), so vocab.json looks like mojibake. This script decodes every
token back to real text and reports stats (fragments, single/multi-char,
CJK/ASCII), plus search / id-lookup / encode helpers.

Usage:
  python scripts/inspect_tokenizer.py --tokenizer tokenizer/ccc-bbpe-24k
  python scripts/inspect_tokenizer.py --tokenizer tokenizer/ccc-bbpe-24k --out tokenizer/ccc-bbpe-24k/vocab.readable.txt
  python scripts/inspect_tokenizer.py --tokenizer tokenizer/ccc-bbpe-24k --grep 学
  python scripts/inspect_tokenizer.py --tokenizer tokenizer/ccc-bbpe-24k --ids 595,598
  python scripts/inspect_tokenizer.py --tokenizer tokenizer/ccc-bbpe-24k --encode "学而时习之"
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="write full readable vocab dump to this file (id<TAB>decoded<TAB>key)")
    parser.add_argument("--max", type=int, default=None,
                        help="only print the first N entries when dumping")
    parser.add_argument("--ids", type=str, default=None,
                        help="comma-separated token ids to look up")
    parser.add_argument("--grep", type=str, default=None,
                        help="show ids whose decoded text contains this string")
    parser.add_argument("--encode", type=str, default=None,
                        help="show how this text is tokenized (ids + tokens + decoded)")
    return parser.parse_args()


def classify(decoded: str, is_special: bool) -> str:
    if is_special:
        return "special"
    if "\ufffd" in decoded:
        return "byte-fragment"
    if any(unicodedata.category(c).startswith("L") for c in decoded):
        return "cjk/letter" if all("\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" or c in "〇" for c in decoded if unicodedata.category(c).startswith("L")) else "mixed-text"
    return "symbol/ascii"


def main() -> None:
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    vocab = tok.get_vocab()
    special_ids = set(tok.all_special_ids)
    by_id = sorted(vocab.items(), key=lambda kv: kv[1])

    decoded: dict[int, str] = {}
    entries: list[tuple[int, str, str, str]] = []  # id, decoded, key, kind
    for key, tid in by_id:
        text = tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        decoded[tid] = text
        entries.append((tid, text, key, classify(text, tid in special_ids)))

    if args.encode:
        ids = tok.encode(args.encode, add_special_tokens=False)
        print(f"=== encode {args.encode!r}: {len(ids)} tokens ===")
        for tid in ids:
            key = [k for k, v in vocab.items() if v == tid][0]
            print(f"  {tid:>5}  {key!r:<16} -> {decoded[tid]!r}")
        print(f"  roundtrip: {tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == args.encode}")
        return

    if args.ids:
        print("=== ids ===")
        for s in args.ids.split(","):
            tid = int(s.strip())
            if tid in decoded:
                print(f"  {tid:>5}  key={vocab and [k for k, v in vocab.items() if v == tid][0]!r}  decoded={decoded[tid]!r}")
            else:
                print(f"  {tid:>5}  (not in vocab)")
        return

    if args.grep:
        print(f"=== tokens containing {args.grep!r} ===")
        hits = [(tid, text) for tid, text in decoded.items() if args.grep in text]
        for tid, text in hits[:50]:
            print(f"  {tid:>5}  {text!r}")
        print(f"  ({len(hits)} total)")
        return

    # ---- default: stats + dump ----
    kinds = {}
    for _, _, _, kind in entries:
        kinds[kind] = kinds.get(kind, 0) + 1

    print(f"=== vocab: {len(vocab):,} tokens ===")
    for kind, n in kinds.items():
        print(f"  {kind:<14} {n:>6}")
    print(f"  special ids: {sorted(special_ids)}")

    limit = args.max if args.max else 20
    print(f"\n=== first {limit} entries ===")
    for tid, text, key, kind in entries[:limit]:
        print(f"  {tid:>5}  {text!r:<12}  key={key!r:<14}  {kind}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for tid, text, key, kind in entries:
                f.write(f"{tid}\t{kind}\t{text}\t{key}\n")
        print(f"\nfull dump written to {args.out}")


if __name__ == "__main__":
    main()
