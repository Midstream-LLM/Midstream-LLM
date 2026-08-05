#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer on CCC corpus.jsonl."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to corpus.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to save the tokenizer.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16384,
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=2048,
    )
    return parser.parse_args()


def iter_corpus(path: Path) -> Iterator[str]:
    """Yield original text without Unicode or simplified/traditional conversion."""

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            text = record.get("content")

            # Compatibility fallback in case another corpus uses `text`.
            if text is None:
                text = record.get("text")

            if not isinstance(text, str):
                raise RuntimeError(
                    f"Missing text field at {path}:{line_number}; "
                    f"keys={sorted(record.keys())}"
                )

            # Only normalize the newline convention.
            # Do not apply NFKC, OpenCC or whitespace collapsing.
            text = text.replace("\r\n", "\n").replace("\r", "\n")

            if text.strip():
                yield text


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    if args.vocab_size <= len(SPECIAL_TOKENS) + 256:
        raise ValueError("Vocabulary size is too small.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<|unk|>",
        )
    )

    # GPT-2-style byte-level pre-tokenization.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,

        # Guarantee that all 256 bytes are present in the vocabulary.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(
        iter_corpus(args.input),
        trainer=trainer,
        length=12009,  # Only used for progress display.
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        eos_token="<|endoftext|>",
        additional_special_tokens=[
            "<|im_start|>",
            "<|im_end|>",
        ],
        model_max_length=args.model_max_length,
        clean_up_tokenization_spaces=False,
    )

    fast_tokenizer.save_pretrained(args.output_dir)

    config = {
        "algorithm": "byte-level-bpe",
        "source": str(args.input),
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(fast_tokenizer),
        "min_frequency": args.min_frequency,
        "normalization": "none",
        "add_prefix_space": False,
        "use_regex": True,
        "special_tokens": SPECIAL_TOKENS,
    }

    with (args.output_dir / "training_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print(f"Saved tokenizer to: {args.output_dir}")
    print(f"Vocabulary size: {len(fast_tokenizer)}")

    for token in SPECIAL_TOKENS:
        print(f"{token}: {fast_tokenizer.convert_tokens_to_ids(token)}")


if __name__ == "__main__":
    main()
