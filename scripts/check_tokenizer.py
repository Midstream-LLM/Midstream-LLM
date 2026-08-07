#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


TEST_STRINGS = [
    "學而時習之，不亦說乎？",
    "学而时习之，不亦说乎？",
    "𠡠令監造。",
    "九州□伯。",
    "鼓：○□○○□□○",
    "天地玄黃\n宇宙洪荒",
    "AI模型训练于2026年，loss=3.14159。",
    "URL: https://example.com/a?x=1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=12009)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
    )

    print("=== Round-trip tests ===")

    for text in TEST_STRINGS:
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        print(f"text:    {text!r}")
        print(f"tokens:  {len(ids)}")
        print(f"decoded: {decoded!r}")
        print(f"exact:   {decoded == text}")
        print()

        if decoded != text:
            raise RuntimeError("Tokenizer is not lossless.")

    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    unk_count = 0
    document_count = 0
    frequencies: Counter[int] = Counter()

    with args.corpus.open("r", encoding="utf-8") as file:
        for line in file:
            if document_count >= args.max_docs:
                break

            record = json.loads(line)
            text = record.get("content") or record.get("text")

            if not isinstance(text, str) or not text:
                continue

            text = text.replace("\r\n", "\n").replace("\r", "\n")

            ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            decoded = tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            if decoded != text:
                raise RuntimeError(
                    f"Round-trip failure in document {document_count}"
                )

            total_chars += len(text)
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(ids)
            unk_count += ids.count(tokenizer.unk_token_id)
            frequencies.update(ids)
            document_count += 1

    print("=== Corpus statistics ===")
    print(f"documents:       {document_count:,}")
    print(f"characters:      {total_chars:,}")
    print(f"UTF-8 bytes:     {total_bytes:,}")
    print(f"tokens:          {total_tokens:,}")
    print(f"chars/token:     {total_chars / total_tokens:.4f}")
    print(f"bytes/token:     {total_bytes / total_tokens:.4f}")
    print(f"UNK count:       {unk_count:,}")
    print(f"used token IDs:  {len(frequencies):,}/{len(tokenizer):,}")

    rare_tokens = sum(
        1 for count in frequencies.values()
        if count < 10
    )

    print(f"tokens used <10 times: {rare_tokens:,}")


if __name__ == "__main__":
    main()
