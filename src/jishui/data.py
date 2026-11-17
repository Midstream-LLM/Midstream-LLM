from __future__ import annotations

import json
import re
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


INDEX_FORMAT_VERSION = 1
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}

# These rules intentionally mirror scripts/pipeline/03_clean.py. Stage 5 rebuilt
# its manifest from pre-clean metadata and lost the inferred Wikisource category.
_WI_CATEGORY_RULES = (
    (re.compile(r"(佛說|佛經|金剛|般若|華嚴|法華|涅槃|楞嚴|維摩|阿彌陀|藥師|地藏|無量壽|大藏|禪宗|心經|楞伽|圓覺|瑜伽師地|道德經|南華|沖虛|抱朴|參同契|黃庭|道藏|本草|黃帝內經|素問|靈樞|難經|傷寒|金匱|針灸|脈經|千金|外臺)"), 4),
    (re.compile(r"(詩經|楚辭|樂府|全唐詩|全宋詞|元曲|詩集|詞集|詩話|詞話|曲譜|詩品|文心雕龍|聲律啟蒙|笠翁對韻|千家詩|唐詩|宋詩|詩鈔|詞鈔|賦集|駢文)"), 3),
    (re.compile(r"(小說|筆記|演義|話本|笑話|寶卷|傳奇|雜劇|南戲|彈詞|說部|世說|聊齋|紅樓|三國|水滸|西遊|金瓶梅|儒林外史|鏡花緣|老殘|官場現形|孽海花|海上花|西廂|牡丹亭|長生殿|桃花扇|琵琶記|竇娥|漢宮秋|閱微草堂)"), 2),
    (re.compile(r"(魯迅|呐喊|彷徨|朝花夕拾|野草|墳|熱風|華蓋|而已集|兩地書|梁啟超|飲冰室|嚴復|天演論|林紓|李伯元|吳趼人|劉鶚|曾樸|蘇曼殊|郁達夫|朱自清|背影|荷塘月色|徐志摩|老舍|沈從文|巴金|茅盾|張恨水|孽海花|老殘遊記|官場現形記|二十年目睹|海上花列傳|駱駝祥子|阿Q|狂人日記|祝福|傷逝|邊城|家|春|秋|雷雨|日出|茶館)"), 5),
    (re.compile(r"(教材|講義|教科書|辭典|字典|年鑑|條約|章程|公報|法令|法律|法規|憲法|新聞|報紙|社論|格致|算學|幾何|代數|聲學|光學|化學|物理|地學|博物|鐵路|電報|郵政|銀行|通商|度量衡|地理|歷史|國文|修身|啟蒙|三字經|百家姓|千字文|弟子規|增廣賢文|幼學瓊林|龍文鞭影|日知錄|通典|通志|文獻通考|會典|實錄|邸鈔|邸報|奏議|詔令|國策|戰國策|水經注|山海經|爾雅|說文|方言|釋名|廣韻|集韻|字彙|正字通)"), 6),
)


def infer_wikisource_category(title: str) -> int:
    for pattern, category in _WI_CATEGORY_RULES:
        if pattern.search(title):
            return category
    return 1


def _file_identity(path: Path) -> dict[str, int | str]:
    if not path.exists():
        return {"path": str(path.resolve()), "missing": True}
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_wikisource_categories(metadata_manifest: Path) -> dict[str, int]:
    categories: dict[str, int] = {}
    with metadata_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("source") == "wikisource":
                categories[record["doc_id"]] = infer_wikisource_category(
                    record.get("title", "")
                )
    return categories


@dataclass
class PretrainIndex:
    data_dir: Path
    shards: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    categories: np.ndarray
    splits: np.ndarray
    sources: np.ndarray
    source_names: tuple[str, ...]
    metadata: dict

    def __post_init__(self) -> None:
        self._token_shards: list[np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.lengths)

    @property
    def token_shards(self) -> list[np.ndarray]:
        if self._token_shards is None:
            shard_ids = sorted(int(value) for value in np.unique(self.shards))
            if shard_ids != list(range(max(shard_ids) + 1)):
                raise ValueError(f"non-contiguous shard ids: {shard_ids}")
            self._token_shards = [
                np.load(
                    self.data_dir / "tokens" / f"shard_{shard_id:05d}.npy",
                    mmap_mode="r",
                )
                for shard_id in shard_ids
            ]
        return self._token_shards

    def indices_for(self, split: str, known_categories_only: bool = True) -> np.ndarray:
        mask = self.splits == SPLIT_TO_ID[split]
        if known_categories_only:
            mask &= self.categories > 0
        return np.flatnonzero(mask)

    def token_totals(self, split: str) -> dict[int, int]:
        ids = self.indices_for(split)
        categories = self.category_ids
        totals = np.zeros(max(categories, default=0) + 1, dtype=np.uint64)
        selected = ids[self.categories[ids] < len(totals)]
        np.add.at(totals, self.categories[selected], self.lengths[selected].astype(np.uint64))
        return {category: int(totals[category]) for category in categories}

    @property
    def category_ids(self):
        if self.metadata.get("category_mode") == "labels":
            return sorted(int(k) for k in self.metadata["category_labels"])
        return range(1, 7)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez(
            temporary,
            shards=self.shards,
            offsets=self.offsets,
            lengths=self.lengths,
            categories=self.categories,
            splits=self.splits,
            sources=self.sources,
            source_names=np.asarray(self.source_names),
            metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, data_dir: str | Path) -> "PretrainIndex":
        with np.load(path, allow_pickle=False) as values:
            return cls(
                data_dir=Path(data_dir),
                shards=values["shards"],
                offsets=values["offsets"],
                lengths=values["lengths"],
                categories=values["categories"],
                splits=values["splits"],
                sources=values["sources"],
                source_names=tuple(str(value) for value in values["source_names"]),
                metadata=json.loads(str(values["metadata"])),
            )


def build_pretrain_index(
    data_dir: str | Path,
    metadata_manifest: str | Path | None = None,
    *, category_mode: str = "legacy",
) -> PretrainIndex:
    if category_mode not in {"legacy", "labels"}:
        raise ValueError("category_mode must be legacy or labels")
    data_dir = Path(data_dir)
    token_manifest = data_dir / "manifest" / "docs_tokens.jsonl"
    if metadata_manifest is None:
        metadata_manifest = data_dir.parent / "interim" / "manifest" / "docs_final.jsonl"
    metadata_manifest = Path(metadata_manifest)
    recovered_wikisource = (
        _load_wikisource_categories(metadata_manifest)
        if category_mode == "legacy" and metadata_manifest.exists()
        else {}
    )
    shards = array("I")
    offsets = array("Q")
    lengths = array("I")
    categories = array("I")
    splits = array("B")
    sources = array("I")
    source_to_id: dict[str, int] = {}
    label_to_id = {}
    recovered_count = 0
    unresolved_count = 0
    unresolved_tokens = 0

    with token_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            source = record["source"]
            if source not in source_to_id:
                source_to_id[source] = len(source_to_id)
            category_text = record.get("category", "")
            if category_mode == "labels":
                label = str(category_text).strip() if category_text is not None else None
                label = label or None
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id) + 1
                category = label_to_id[label]
            else:
                category = int(category_text) if category_text else 0
            if category_mode == "legacy" and category == 0 and source == "wikisource":
                category = recovered_wikisource.get(record["doc_id"], 0)
                recovered_count += category > 0
            if category == 0:
                unresolved_count += 1
                unresolved_tokens += int(record["tokens"])
            shards.append(int(record["shard"]))
            offsets.append(int(record["offset"]))
            lengths.append(int(record["tokens"]))
            categories.append(category)
            splits.append(SPLIT_TO_ID[record["split"]])
            sources.append(source_to_id[source])

    source_names = tuple(name for name, _ in sorted(source_to_id.items(), key=lambda x: x[1]))
    metadata = {
        "format_version": INDEX_FORMAT_VERSION,
        "token_manifest": _file_identity(token_manifest),
        "metadata_manifest": _file_identity(metadata_manifest),
        "category_recovery": (
            "wikisource-title-rules-v1" if metadata_manifest.exists() else "unavailable"
        ),
        "wikisource_categories_recovered": recovered_count,
        "unresolved_documents_excluded": unresolved_count,
        "unresolved_tokens_excluded": unresolved_tokens,
    }
    if category_mode == "labels":
        metadata.update(category_mode="labels", category_labels={str(v): k for k, v in label_to_id.items()})
        metadata["category_recovery"] = "not-needed"
        metadata["dataset_manifest"] = _file_identity(data_dir / "manifest.json")
    return PretrainIndex(
        data_dir=data_dir,
        shards=np.asarray(shards, dtype=np.uint32),
        offsets=np.asarray(offsets, dtype=np.uint64),
        lengths=np.asarray(lengths, dtype=np.uint32),
        categories=np.asarray(categories, dtype=np.uint32),
        splits=np.asarray(splits, dtype=np.uint8),
        sources=np.asarray(sources, dtype=np.uint32),
        source_names=source_names,
        metadata=metadata,
    )


def load_or_build_index(
    data_dir: str | Path,
    cache_path: str | Path,
    metadata_manifest: str | Path | None = None,
    *, category_mode: str = "legacy",
) -> tuple[PretrainIndex, bool]:
    data_dir = Path(data_dir)
    cache_path = Path(cache_path)
    token_manifest = data_dir / "manifest" / "docs_tokens.jsonl"
    if metadata_manifest is None:
        metadata_manifest = data_dir.parent / "interim" / "manifest" / "docs_final.jsonl"
    metadata_manifest = Path(metadata_manifest)
    expected = {
        "format_version": INDEX_FORMAT_VERSION,
        "token_manifest": _file_identity(token_manifest),
        "metadata_manifest": _file_identity(metadata_manifest),
    }
    if category_mode == "labels":
        expected.update(category_mode="labels", dataset_manifest=_file_identity(data_dir / "manifest.json"))
    if cache_path.exists():
        index = PretrainIndex.load(cache_path, data_dir)
        if (index.metadata.get("category_mode", "legacy") == category_mode
                and all(index.metadata.get(key) == value for key, value in expected.items())):
            return index, False
    index = build_pretrain_index(data_dir, metadata_manifest, category_mode=category_mode)
    index.save(cache_path)
    return index, True


def load_sampling_config(data_dir: str | Path) -> dict:
    with (Path(data_dir) / "sampling_weights.json").open(encoding="utf-8") as handle:
        return json.load(handle)


class ShuffledDocumentIterator(Iterator[np.ndarray]):
    """Shuffle documents each epoch and consume every token, including prefixes.

    Packing crosses document boundaries with EOD; it never randomly discards a
    document's beginning. State is small and reconstructs the epoch permutation.
    """

    def __init__(self, index, seq_len, batch_size, seed, eod_token_id=2):
        self.index, self.seq_len, self.batch_size = index, seq_len, batch_size
        self.seed, self.eod_token_id = seed, eod_token_id
        self.ids = index.indices_for("train")
        if not len(self.ids) or np.any(index.lengths[self.ids] <= 0):
            raise ValueError("document stream requires nonempty, positive-length training documents")
        self.epoch = self.doc_position = self.token_position = self.batches_emitted = 0
        self.need_eod, self.carry = False, None
        self._shuffle()

    def _shuffle(self):
        self.order = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).permutation(self.ids)

    def __iter__(self):
        return self

    def __next__(self):
        result = np.empty((self.batch_size, self.seq_len + 1), dtype=np.int32)
        for row in result:
            pos = 0
            if self.carry is not None:
                row[0], pos = self.carry, 1
            while pos < len(row):
                if self.need_eod:
                    row[pos] = self.eod_token_id
                    pos += 1
                    self.need_eod = False
                    self.doc_position += 1
                    self.token_position = 0
                    if self.doc_position == len(self.order):
                        self.doc_position = 0
                        self.epoch += 1
                        self._shuffle()
                    continue
                doc = int(self.order[self.doc_position])
                size = min(int(self.index.lengths[doc]) - self.token_position, len(row) - pos)
                offset = int(self.index.offsets[doc]) + self.token_position
                row[pos:pos + size] = self.index.token_shards[int(self.index.shards[doc])][offset:offset + size]
                pos += size
                self.token_position += size
                self.need_eod = self.token_position == int(self.index.lengths[doc])
            self.carry = int(row[-1])
        self.batches_emitted += 1
        return result

    def state_dict(self):
        return {"kind": "document_stream", "seed": self.seed, "epoch": self.epoch,
                "doc_position": self.doc_position, "token_position": self.token_position,
                "need_eod": self.need_eod, "carry": self.carry, "batches_emitted": self.batches_emitted}

    def load_state_dict(self, state):
        if state.get("kind") != "document_stream" or state["seed"] != self.seed:
            raise ValueError("document stream configuration changed")
        for key in ("epoch", "doc_position", "token_position", "need_eod", "carry", "batches_emitted"):
            setattr(self, key, state[key])
        self._shuffle()


class PackedBatchIterator(Iterator[np.ndarray]):
    """Length-aware, category-balanced packed batches over mmap token shards."""

    def __init__(
        self,
        index: PretrainIndex,
        sampling: dict,
        seq_len: int,
        batch_size: int,
        seed: int,
        mode: str = "target",
        eod_token_id: int = 2,
    ):
        self.index = index
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eod_token_id = eod_token_id
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.batches_emitted = 0
        train_ids = index.indices_for("train")
        self._groups: dict[int, np.ndarray] = {}
        self._cumulative: dict[int, np.ndarray] = {}
        totals: dict[int, int] = {}
        for category in index.category_ids:
            ids = train_ids[index.categories[train_ids] == category]
            if len(ids) == 0:
                continue
            self._groups[category] = ids
            cumulative = np.cumsum(index.lengths[ids], dtype=np.uint64)
            self._cumulative[category] = cumulative
            totals[category] = int(cumulative[-1])

        self.category_ids = np.asarray(sorted(self._groups), dtype=np.uint32)
        if mode == "target":
            raw = np.asarray(
                [float(sampling["target_mix"][str(cat)]) for cat in self.category_ids],
                dtype=np.float64,
            )
        elif mode == "weights":
            raw = np.asarray(
                [
                    totals[int(cat)]
                    * float(sampling["weights_by_category"][str(cat)])
                    for cat in self.category_ids
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"unknown sampling mode: {mode}")
        if not len(raw) or not np.isfinite(raw).all() or np.any(raw < 0) or raw.sum() <= 0:
            raise ValueError("training sampling probabilities must be finite, nonnegative and nonempty")
        self.category_probabilities = raw / raw.sum()

    def __iter__(self) -> "PackedBatchIterator":
        return self

    def __next__(self) -> np.ndarray:
        output = np.empty((self.batch_size, self.seq_len + 1), dtype=np.int32)
        selected = self.rng.choice(
            self.category_ids,
            size=self.batch_size,
            p=self.category_probabilities,
        )
        for row, category in enumerate(selected):
            self._fill_row(output[row], int(category))
        self.batches_emitted += 1
        return output

    def _fill_row(self, output: np.ndarray, category: int) -> None:
        ids = self._groups[category]
        cumulative = self._cumulative[category]
        position = 0
        while position < len(output):
            token_position = int(self.rng.integers(0, int(cumulative[-1])))
            local_doc = int(np.searchsorted(cumulative, token_position, side="right"))
            previous = int(cumulative[local_doc - 1]) if local_doc else 0
            start_in_doc = token_position - previous
            doc_index = int(ids[local_doc])
            available = int(self.index.lengths[doc_index]) - start_in_doc
            count = min(available, len(output) - position)
            shard = int(self.index.shards[doc_index])
            offset = int(self.index.offsets[doc_index]) + start_in_doc
            output[position : position + count] = self.index.token_shards[shard][
                offset : offset + count
            ]
            position += count
            if count == available and position < len(output):
                output[position] = self.eod_token_id
                position += 1

    def state_dict(self) -> dict:
        return {
            "batches_emitted": self.batches_emitted,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.batches_emitted = int(state["batches_emitted"])
        self.rng.bit_generator.state = state["rng_state"]


class SequentialBatchIterator(Iterator[np.ndarray]):
    """Deterministic, unweighted document stream for validation/test."""

    def __init__(
        self,
        index: PretrainIndex,
        split: str,
        seq_len: int,
        batch_size: int,
        eod_token_id: int = 2,
        category: int | None = None,
    ):
        self.index = index
        self.doc_ids = index.indices_for(split)
        if category is not None:
            self.doc_ids = self.doc_ids[index.categories[self.doc_ids] == category]
        if len(self.doc_ids) == 0:
            label = f"{split} category={category}" if category is not None else split
            raise ValueError(f"no usable documents in {label}")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eod_token_id = eod_token_id
        self.doc_position = 0
        self.token_position = 0

    def __iter__(self) -> "SequentialBatchIterator":
        return self

    def __next__(self) -> np.ndarray:
        output = np.empty((self.batch_size, self.seq_len + 1), dtype=np.int32)
        for row in output:
            self._fill_row(row)
        return output

    def _fill_row(self, output: np.ndarray) -> None:
        position = 0
        while position < len(output):
            doc_index = int(self.doc_ids[self.doc_position])
            available = int(self.index.lengths[doc_index]) - self.token_position
            count = min(available, len(output) - position)
            shard = int(self.index.shards[doc_index])
            offset = int(self.index.offsets[doc_index]) + self.token_position
            output[position : position + count] = self.index.token_shards[shard][
                offset : offset + count
            ]
            position += count
            self.token_position += count
            if self.token_position == int(self.index.lengths[doc_index]):
                self.doc_position = (self.doc_position + 1) % len(self.doc_ids)
                self.token_position = 0
                if position < len(output):
                    output[position] = self.eod_token_id
                    position += 1
