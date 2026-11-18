"""Length-independent token storage, shuffled sample views and fixed validation.

Only indices are shuffled. Underlying documents and token shards are immutable.
Targets partition the document stream once per epoch; each block also reads its
preceding input token and optional future targets from that same original stream.
No attention matrix or model-specific MTP implementation is built here.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class SampleSpec:
    seq_len: int
    prediction_offsets: tuple[int, ...] = (1,)
    eod_token_id: int = 2
    # Current NTP retains the existing EOD-separated, causal packing objective.
    cross_document_targets: bool = True

    def __post_init__(self):
        if type(self.seq_len) is not int or self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        if (not self.prediction_offsets or any(type(k) is not int or k < 1 for k in self.prediction_offsets)
                or tuple(sorted(set(self.prediction_offsets))) != self.prediction_offsets):
            raise ValueError("prediction_offsets must be sorted, unique positive integers")

    def identity(self):
        return {"version": VERSION, "seq_len": self.seq_len,
                "prediction_offsets": list(self.prediction_offsets), "eod_token_id": self.eod_token_id,
                "cross_document_targets": self.cross_document_targets, "packing": "document_stream_eod"}


@dataclass
class TokenBatch:
    tokens: np.ndarray                 # [B, L + max(prediction_offsets)]
    document_ids: np.ndarray           # -1 denotes padding; EOD belongs to its document
    target_counts: np.ndarray          # valid core target slots per row, excludes padding/lookahead
    target_starts: np.ndarray          # absolute target positions in the original epoch stream
    spec: SampleSpec

    @property
    def input_ids(self):
        return self.tokens[:, :self.spec.seq_len]

    @property
    def position_ids(self):
        return np.broadcast_to(np.arange(self.spec.seq_len, dtype=np.int64), self.input_ids.shape)

    @property
    def segment_offsets(self):
        """Per-row boundaries for future packed/variable-length attention backends."""
        result = []
        for ids, count in zip(self.document_ids, self.target_counts):
            end = int(count)
            cuts = np.flatnonzero(ids[1:end] != ids[:max(0, end-1)]) + 1
            result.append(np.asarray([0, *cuts, end], dtype=np.int64))
        return result

    def targets(self, offset=1, *, cross_document=None):
        if offset not in self.spec.prediction_offsets:
            raise ValueError("prediction offset is absent from sample specification")
        length = self.spec.seq_len
        target = self.tokens[:, offset:offset + length].copy()
        mask = np.arange(length)[None, :] < self.target_counts[:, None]
        mask &= self.document_ids[:, offset:offset + length] >= 0
        # MTP defaults to respecting real document boundaries, regardless of the
        # legacy NTP policy. Auxiliary architectures can request a stricter NTP too.
        allow = self.spec.cross_document_targets if offset == 1 else False
        if cross_document is not None:
            allow = cross_document
        if not allow:
            mask &= self.document_ids[:, :length] == self.document_ids[:, offset:offset + length]
        target[~mask] = -100
        return target, mask


class DocumentStream:
    """A virtual concatenation of complete documents with synthetic EOD tokens."""

    def __init__(self, index, doc_ids, eod_token_id):
        self.index = index
        self.doc_ids = np.asarray(doc_ids, dtype=np.int64)
        if not len(self.doc_ids) or np.any(index.lengths[self.doc_ids] <= 0):
            raise ValueError("document stream must contain positive-length documents")
        self.eod_token_id = eod_token_id
        self.ends = np.cumsum(index.lengths[self.doc_ids].astype(np.int64) + 1)
        self.size = int(self.ends[-1])

    def read(self, start, length):
        tokens = np.full(length, self.eod_token_id, dtype=np.int64)
        documents = np.full(length, -1, dtype=np.int64)
        pos = 0
        if start == -1:
            # A virtual BOS/EOD makes the first document's first token a target.
            documents[0] = self.doc_ids[0]
            pos, start = 1, 0
        if start < 0:
            raise ValueError("invalid stream offset")
        doc_pos = int(np.searchsorted(self.ends, start, side="right"))
        while pos < length and start < self.size:
            doc = int(self.doc_ids[doc_pos])
            begin = int(self.ends[doc_pos-1]) if doc_pos else 0
            within = start - begin
            body = int(self.index.lengths[doc])
            if within == body:
                tokens[pos] = self.eod_token_id
                count = 1
            else:
                count = min(body-within, length-pos)
                offset = int(self.index.offsets[doc]) + within
                tokens[pos:pos+count] = self.index.token_shards[int(self.index.shards[doc])][offset:offset+count]
            documents[pos:pos+count] = doc
            pos += count
            start += count
            if start == int(self.ends[doc_pos]):
                doc_pos += 1
        return tokens, documents

    def sample(self, start, count, spec):
        if not 0 <= start < self.size or not 0 < count <= min(spec.seq_len, self.size-start):
            raise ValueError("invalid sample bounds")
        tokens, docs = self.read(start-1, spec.seq_len + max(spec.prediction_offsets))
        return tokens, docs, count, start


def make_batch(rows, spec):
    tokens, documents, counts, starts = zip(*rows)
    return TokenBatch(np.stack(tokens), np.stack(documents), np.asarray(counts, dtype=np.int64),
                      np.asarray(starts, dtype=np.int64), spec)


class ShuffledBlockIterator:
    """Shuffle all fixed-length blocks; retain the final partial block with a mask.

    Optional migration starts with only the unread suffix of the old document
    epoch. Later epochs cover the entire dataset. Cached arrays are derivative,
    and a checkpoint contains the recipe and cursor needed to reconstruct them.
    """

    def __init__(self, index, spec, batch_size, seed, cache_dir=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.index, self.spec, self.batch_size, self.seed = index, spec, batch_size, seed
        self.ids = index.indices_for("train")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.dataset_id = digest({"metadata": index.metadata, "documents": len(index),
                                  "train_tokens": int(index.lengths[self.ids].astype(np.int64).sum())})
        self.epoch = self.cursor = self.first_epoch = self.first_offset = self.batches_emitted = 0
        self._loaded = None
        self._prepare()

    def _prepare(self):
        start = self.first_offset if self.epoch == self.first_epoch else 0
        key = (self.epoch, start)
        if self._loaded == key:
            return
        doc_order = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).permutation(self.ids)
        self.stream = DocumentStream(self.index, doc_order, self.spec.eod_token_id)
        if not 0 <= start <= self.stream.size:
            raise ValueError("invalid migrated stream offset")
        self.start = start
        n = math.ceil((self.stream.size-start) / self.spec.seq_len)
        recipe = dict(self.spec.identity(), dataset=self.dataset_id, seed=self.seed,
                      epoch=self.epoch, start=start, blocks=n)
        name = digest(recipe)
        path = self.cache_dir / f"blocks-{name}.npy" if self.cache_dir else None
        if path and path.exists() and path.with_suffix(".json").exists():
            self.order = np.load(path, mmap_mode="r", allow_pickle=False)
            if self.order.shape != (n,) or self.order.dtype != np.dtype("int64"):
                raise ValueError("invalid block index cache")
            saved = json.loads(path.with_suffix(".json").read_text())
            if (saved.get("recipe") != recipe or saved.get("sha256") !=
                    hashlib.sha256(self.order.tobytes()).hexdigest()):
                raise ValueError("block index cache checksum mismatch")
        else:
            self.order = np.arange(n, dtype=np.int64)
            np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 0xB10C])).shuffle(self.order)
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp.npy")
                np.save(temporary, self.order)
                temporary.replace(path)
                manifest = {"recipe": recipe, "sha256": hashlib.sha256(self.order.tobytes()).hexdigest()}
                temporary = path.with_suffix(".tmp.json")
                temporary.write_text(json.dumps(manifest, indent=2))
                temporary.replace(path.with_suffix(".json"))
        self._loaded = key

    def __iter__(self):
        return self

    def __next__(self):
        rows = []
        for _ in range(self.batch_size):
            if self.cursor == len(self.order):
                self.epoch += 1
                self.cursor = 0
                self._prepare()
            start = self.start + int(self.order[self.cursor]) * self.spec.seq_len
            count = min(self.spec.seq_len, self.stream.size-start)
            rows.append(self.stream.sample(start, count, self.spec))
            self.cursor += 1
        self.batches_emitted += 1
        return make_batch(rows, self.spec)

    def state_dict(self):
        return {"kind": "shuffled_blocks", "version": VERSION, "spec": self.spec.identity(),
                "dataset": self.dataset_id, "seed": self.seed, "batch_size": self.batch_size,
                "epoch": self.epoch, "cursor": self.cursor, "first_epoch": self.first_epoch,
                "first_offset": self.first_offset, "batches_emitted": self.batches_emitted}

    def load_state_dict(self, state):
        for key in ("kind", "version", "spec", "dataset", "seed", "batch_size"):
            if state.get(key) != self.state_dict()[key]:
                raise ValueError(f"block sampler changed {key}")
        for key in ("epoch", "cursor", "first_epoch", "first_offset", "batches_emitted"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid sampler {key}")
            setattr(self, key, state[key])
        if self.epoch < self.first_epoch:
            raise ValueError("sampler epoch precedes migration")
        self._prepare()
        if self.cursor > len(self.order):
            raise ValueError("invalid block cursor")

    def migrate_document_stream(self, state):
        if state.get("kind") != "document_stream" or state["seed"] != self.seed:
            raise ValueError("migration requires matching document stream seed")
        self.epoch = self.first_epoch = int(state["epoch"])
        self.first_offset = 0
        self._prepare()
        pos = int(state["doc_position"])
        if not 0 <= pos < len(self.stream.doc_ids):
            raise ValueError("invalid document cursor")
        doc = int(self.stream.doc_ids[pos])
        token_pos = int(state["token_position"])
        if not 0 <= token_pos <= int(self.index.lengths[doc]):
            raise ValueError("invalid document token cursor")
        if bool(state["need_eod"]) != (token_pos == int(self.index.lengths[doc])):
            raise ValueError("invalid document EOD cursor")
        self.first_offset = (int(self.stream.ends[pos-1]) if pos else 0) + token_pos
        if state["carry"] is not None:
            previous = self.stream.read(self.first_offset-1, 1)[0][0]
            if int(previous) != int(state["carry"]):
                raise ValueError("migration carry does not match token stream")
        self.cursor = 0
        self.batches_emitted = int(state["batches_emitted"])
        self._prepare()


class FixedValidation:
    """Fixed, source-stratified blocks spanning the whole held-out split.

    Every nonempty source receives samples. Per-source means are combined using
    that source's full validation token mass, not its quota of evaluation blocks.
    Counts and missing training sources are reported explicitly.
    """

    def __init__(self, index, spec, batches, batch_size, seed):
        self.index, self.spec = index, spec
        ids = index.indices_for("val")
        if not len(ids):
            raise ValueError("empty validation split")
        sources = sorted(map(int, np.unique(index.sources[ids])))
        budget = max(batches * batch_size, len(sources))
        self.streams, self.samples = {}, []
        self.source_tokens = {source: int((index.lengths[ids[index.sources[ids] == source]].astype(np.int64)+1).sum())
                              for source in sources}
        total = sum(self.source_tokens.values())
        # Reserve one sample per source, allocate the remainder by token mass.
        extra = {s: (budget-len(sources))*self.source_tokens[s]/total for s in sources}
        quotas = {s: 1+int(extra[s]) for s in sources}
        for s in sorted(sources, key=lambda s: (-(extra[s]-int(extra[s])), s))[:budget-sum(quotas.values())]:
            quotas[s] += 1
        details = {}
        for source in sources:
            source_ids = ids[index.sources[ids] == source]
            rng = np.random.default_rng(np.random.SeedSequence([seed, source, 0xEA1]))
            stream = DocumentStream(index, rng.permutation(source_ids), spec.eod_token_id)
            self.streams[source] = stream
            blocks = math.ceil(stream.size/spec.seq_len)
            count = min(quotas[source], blocks)
            # Jitter within equally spaced strata: cover the entire source, not
            # just its prefix, without scanning/reordering the corpus itself.
            chosen = [int(rng.integers(blocks*i//count, blocks*(i+1)//count)) for i in range(count)]
            for block in chosen:
                start = block*spec.seq_len
                self.samples.append((source, start, min(spec.seq_len, stream.size-start)))
            details[index.source_names[source]] = {"documents": len(source_ids),
                "tokens_with_eod": stream.size, "weight": stream.size/total, "blocks": count,
                "evaluated_tokens": sum(min(spec.seq_len, stream.size-b*spec.seq_len) for b in chosen)}
        train_sources = set(map(int, np.unique(index.sources[index.indices_for("train")])))
        self.metadata = {"version": "source_stratified_blocks_v1", "seed": seed,
            "spec": spec.identity(), "sources": details,
            "missing_training_sources": [index.source_names[s] for s in sorted(train_sources-set(sources))]}
        self.metadata["fingerprint"] = digest({"metadata": self.metadata, "index": index.metadata,
                                               "samples": self.samples})

    def __iter__(self):
        for source, start, count in self.samples:
            yield source, make_batch([self.streams[source].sample(start, count, self.spec)], self.spec)

    def weighted_mean(self, means):
        total = sum(self.source_tokens.values())
        return sum(means[source]*tokens/total for source, tokens in self.source_tokens.items())
