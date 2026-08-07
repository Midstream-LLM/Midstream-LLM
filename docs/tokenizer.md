# Tokenizer 训练记录（ccc-bbpe-24k / ccc-bbpe-32k）

v0 实验用 GPT-2 风格 byte-level BPE。本文档记录训练配置、验证结果与「单字 vs BPE merge」的取舍讨论。

## 1. 训练配置（见各目录 training_config.json）

| 项 | ccc-bbpe-24k | ccc-bbpe-32k |
|---|---|---|
| 算法 | byte-level BPE | byte-level BPE |
| 训练语料 | `dataset/normalized/ccc_corpus.jsonl`（12,009 条，点校本） | 同左 |
| vocab_size（请求/实际） | 24576 / 24576 | 32768 / 32768 |
| min_frequency | 2 | 2 |
| normalization | none | none |
| add_prefix_space | false | false |
| use_regex（ByteLevel 分词） | true | true |
| special tokens | `<\|pad\|> <\|unk\|> <\|endoftext\|> <\|im_start\|> <\|im_end\|>`（5 个，id 0–4） | 同左 |

脚本：`scripts/train_bbpe.py`（tokenizers 库，`initial_alphabet` = ByteLevel 256 字节表，保证全字节在词表中）。
语料预处理仅做 `\r\n → \n`，**不做** NFKC / OpenCC / 空白折叠 / 简繁转换。

## 2. 验证结果（scripts/check_tokenizer.py，CCC corpus 全量 12,009 篇）

| 指标 | ccc-bbpe-24k | ccc-bbpe-32k |
|---|---|---|
| chars/token | **1.37** | **1.42** |
| 无损往返（round-trip） | 全量通过 | 全量通过 |
| UNK | 0 | 0 |

实测单字编码（两词表行为一致，32k 为 24k merge 的超集）：
- 常用字单 token：为、集、說
- 繁体/异体按字节碎片：學 → 2 tok、為 → 2 tok、〇 → 2 tok
- 生僻字/特殊符单 token 无损：𠡠 → 1 tok、□ → 1 tok（byte-level 兜底保证任何字符无损，代价是碎片）

## 3. 关键观察

- **chars/token ≈ 1.4**：古汉语以单字表义，BPE merge 收益有限（对比现代中文 BPE 通常 1.5–2.0）。词表从 24k → 32k 提升有限（1.37 → 1.42）。
- **byte-level 兜底**：生僻字/异体字/□ 等任何字符都无损往返，但未 merge 的字符按 UTF-8 字节拆成多 token。这是 v0 的核心取舍：**保底无损，效率靠 merge**。
- **简繁不对称**：训练语料 CCC 以简体为主，繁体字（學/為/〇）多为字节碎片，而部分常用繁体（說）因在语料中出现而成为单 token。殆知阁语料简繁混杂（正史简体、佛藏/集藏繁体），对繁体侧效率有损。
- 词表结构（32k，inspect_tokenizer.py）：5 special + ~224 ASCII 单字节 + 31851 cjk/letter + 618 字节碎片 + 少量混合。

## 4. 决策

- **正式训练用 ccc-bbpe-32k**：更大词表、更低碎片率，embedding 增量（32k vs 24k ≈ 8M 参数）可忽略。
- 当前不重训。开放问题（决定重训时再评估）：
  1. **在殆知阁全量（1.8B 字符，繁简混杂）上重训**，改善繁体侧效率——需验证 chars/token 提升幅度。
  2. **简繁合并**（OpenCC 归一后再训 + 预测时归一）：无损性存疑，暂缓。
  3. **单字词表**（~20k 常用字，无 merge）：省 embedding、古汉语语义粒度天然单字，但与 Qwen3 系列 BPE 结构不一致，且碎片问题换成了 OOV 问题；暂不采用。
- 词表冻结后不可变，数据管线以「tokenizer 为唯一真理源」。
