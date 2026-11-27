#!/usr/bin/env python3
"""Compare tokenizers on the v4 eval sets.

Per tokenizer and per (label, source): chars/token, share of corpus chars encoded as a
whole single-char token, share of tokens that are not whole characters (byte pieces),
round-trip failures. Also vocab composition by Han length.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer


def is_han(ch: str) -> bool:
    o = ord(ch)
    return 0x3400 <= o <= 0x9FFF or 0x20000 <= o <= 0x3FFFF or 0xF900 <= o <= 0xFAFF or o == 0x3007


def vocab_profile(tok: Tokenizer) -> dict:
    out = defaultdict(int)
    for i in range(tok.get_vocab_size()):
        s = tok.decode([i], skip_special_tokens=False)
        if "�" in s:
            out["byte_piece"] += 1
        elif s and all(is_han(c) for c in s):
            out[f"han_{min(len(s), 5)}"] += 1
        else:
            out["other"] += 1
    return dict(sorted(out.items()))


def single_char_set(tok: Tokenizer) -> tuple[set[str], set[int]]:
    chars, byte_ids = set(), set()
    for i in range(tok.get_vocab_size()):
        s = tok.decode([i], skip_special_tokens=False)
        if "�" in s:
            byte_ids.add(i)
        elif len(s) == 1:
            chars.add(s)
    return chars, byte_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", action="append", nargs=2, metavar=("NAME", "TOKENIZER_JSON"), required=True)
    p.add_argument("--eval", action="append", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=2000)
    args = p.parse_args()

    rows = []
    for path in args.eval:
        with path.open() as f:
            rows += [json.loads(l) for l in f]

    result = {"eval_rows": len(rows), "tokenizers": {}}
    for name, path in args.tokenizer:
        tok = Tokenizer.from_file(path)
        singles, byte_ids = single_char_set(tok)
        agg = defaultdict(lambda: {"chars": 0, "tokens": 0, "single_hit": 0, "byte_tokens": 0, "roundtrip_fail": 0})
        for start in range(0, len(rows), args.batch):
            batch = rows[start:start + args.batch]
            encs = tok.encode_batch([r["text"] for r in batch], add_special_tokens=False)
            for r, e in zip(batch, encs):
                text = r["text"]
                for key in ("ALL", r["label"], f"{r['label']}\t{r['source']}"):
                    a = agg[key]
                    a["chars"] += len(text)
                    a["tokens"] += len(e.ids)
                    a["single_hit"] += sum(c in singles for c in text)
                    a["byte_tokens"] += sum(t in byte_ids for t in e.ids)
                if start % (args.batch * 20) == 0 and tok.decode(e.ids) != text:
                    agg["ALL"]["roundtrip_fail"] += 1
        table = {}
        for key, a in sorted(agg.items(), key=lambda kv: -kv[1]["chars"]):
            table[key] = {
                "chars": a["chars"],
                "chars_per_token": round(a["chars"] / max(a["tokens"], 1), 4),
                "single_token_char_share": round(a["single_hit"] / max(a["chars"], 1), 5),
                "byte_piece_token_share": round(a["byte_tokens"] / max(a["tokens"], 1), 5),
                **({"roundtrip_fail_sampled": a["roundtrip_fail"]} if key == "ALL" else {}),
            }
        result["tokenizers"][name] = {"vocab": vocab_profile(tok), "single_chars": len(singles), "by": table}
        print(name, json.dumps(table["ALL"]), flush=True)

    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
