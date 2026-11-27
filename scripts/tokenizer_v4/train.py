#!/usr/bin/env python3
"""Train the v4 byte-level BPE tokenizer.

Differences from scripts/train_bbpe.py (ccc-bbpe-32k):
- Pre-tokenizer splits Han runs into chunks of at most --han-chunk characters before
  ByteLevel, so tokens span at most that many Han characters. Unpunctuated classical
  text otherwise yields line-long pre-tokens (memory blow-up, merges across clauses).
- Digits are single pre-tokens; full-width space runs and newline runs are their own
  pre-tokens; other symbols are isolated one by one.
- Special tokens keep ids 0-4 and add reserved tokens after them.
No Unicode / OpenCC / whitespace normalisation is applied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

BASE_SPECIAL = ["<|pad|>", "<|unk|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def split_pattern(han_chunk: int | None) -> str:
    han = r"\p{Han}+" if han_chunk is None else rf"\p{{Han}}{{1,{han_chunk}}}"
    return han + (                               # Han runs (chunked only for training)
        r"|\p{N}"                                # single digits
        r"|(?:(?!\p{Han})[\p{L}\p{M}])+"         # non-Han letter runs (Latin, kana, ...)
        r"|　+"                              # full-width space runs (indentation)
        r"|\n+"
        r"|[^\S\n　]+"                       # other whitespace runs
        r"|[^\s\p{L}\p{M}\p{N}]"                 # any other symbol, one at a time
    )


def pre_tokenizer(han_chunk: int | None):
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(split_pattern(han_chunk)), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])


def build(han_chunk: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizer(han_chunk)
    tok.decoder = decoders.ByteLevel()
    return tok


def iter_text(path: Path, every: int):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % every:
                continue
            text = json.loads(line)["text"].replace("\r\n", "\n").replace("\r", "\n")
            if text.strip():
                yield text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--han-chunk", type=int, default=4, help="Han chunk length while counting")
    p.add_argument("--every", type=int, default=1, help="use every N-th input line")
    p.add_argument("--reserved", type=int, default=11)
    p.add_argument("--model-max-length", type=int, default=2048)
    args = p.parse_args()

    special = BASE_SPECIAL + [f"<|reserved_{i}|>" for i in range(args.reserved)]
    tok = build(args.han_chunk)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(iter_text(args.input, args.every), trainer=trainer)
    # Encode whole Han runs: merges learned on chunks apply anywhere in the run, so token
    # boundaries are not tied to the arbitrary chunk grid used for counting.
    tok.pre_tokenizer = pre_tokenizer(None)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        eos_token="<|endoftext|>",
        additional_special_tokens=special[3:],
        model_max_length=args.model_max_length,
        clean_up_tokenization_spaces=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(args.output_dir)
    config = {
        "algorithm": "byte-level-bpe",
        "source": str(args.input),
        "input_every_nth_line": args.every,
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(fast),
        "min_frequency": args.min_frequency,
        "normalization": "none",
        "pre_tokenizer": {"split_regex": split_pattern(None),
                          "then": "ByteLevel(add_prefix_space=False, use_regex=False)",
                          "training_split_regex": split_pattern(args.han_chunk)},
        "special_tokens": special,
    }
    (args.output_dir / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print("saved", args.output_dir, len(fast))


if __name__ == "__main__":
    main()
