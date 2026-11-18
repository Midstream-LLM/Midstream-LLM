#!/usr/bin/env python3
"""Audit migration/validation using metadata and counts only; never decode text."""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from jishui.block_data import FixedValidation, SampleSpec, ShuffledBlockIterator
from jishui.data import PretrainIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.parent_checkpoint / "train_state.json").read_text())
    config = json.loads(args.config.read_text())
    train, pretrain = config["training"], config["pretraining"]
    index = PretrainIndex.load(args.index, args.data_dir)
    assert index.metadata == state["index_metadata"]
    spec = SampleSpec(train["seq_len"], (1,), train["eod_token_id"])
    sampler = ShuffledBlockIterator(index, spec, train["batch_size"], train["seed"])
    sampler.migrate_document_stream(state["sampler"])
    consumed = sampler.first_epoch*sampler.stream.size + sampler.first_offset - 1
    assert consumed == state["tokens_seen"], (consumed, state["tokens_seen"])
    assert np.unique(sampler.order).size == len(sampler.order)
    documents, sources, tokens = [], [], 0
    started = time.monotonic()
    for _ in range(64):
        docs = set()
        for _ in range(train["grad_accum_steps"]):
            batch = next(sampler)
            docs.update(map(int, np.unique(batch.document_ids[batch.document_ids >= 0])))
            tokens += int(batch.targets()[1].sum())
        documents.append(len(docs))
        sources.append(len({int(index.sources[d]) for d in docs}))
    elapsed = time.monotonic()-started
    evaluation = FixedValidation(index, spec, train["eval_batches"], train["batch_size"],
                                 pretrain["validation"]["seed"])
    print(json.dumps({"parent_step": state["step"], "parent_tokens": state["tokens_seen"],
        "migration": {"epoch": sampler.first_epoch, "unread_start": sampler.first_offset,
                      "remaining_targets": sampler.stream.size-sampler.first_offset,
                      "blocks": len(sampler.order)},
        "preview": {"updates": 64, "tokens": tokens, "seconds": elapsed,
            "one_document_updates": documents.count(1), "median_documents": statistics.median(documents),
            "median_sources": statistics.median(sources)}, "validation": evaluation.metadata}, indent=2))


if __name__ == "__main__":
    main()
