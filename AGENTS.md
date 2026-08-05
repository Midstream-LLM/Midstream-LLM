# AGENTS.md — Jishui（集水）

古汉语原生 LLM 从零预训练项目。读这一节之前，agent 不会知道这些事。

## 项目性质
- **目标**：从零训练一个古汉语原生的小型 decoder-only LLM，先 ~600M，再 scale 到 ~1.5B。
- **技术栈**：PyTorch；标准 MHA + RoPE；结构参考 Qwen3-1.7B；自训 BPE tokenizer。
- **阶段**：当前只做 pretraining。后训练 (SFT/RL 等) 暂缓。
- **两部分分工**（团队 3 人）：①模型训练（当前）；②自研 AI 编译器 / infra（后续，不在本阶段范围）。
- 这是 hobby project，以学习为目的；用现成技术，不追新。

## 仓库布局（约定，部分已落地）
- `src/jishui/` —— 单 Python 包（src-layout，计划）；模型 / tokenizer / 数据 / 训练脚本都在其下（尚未创建）。
- `scripts/prepare_dataset.py` —— 数据管线：manifest / normalize / reports / extra 子命令。只读 `dataset/raw/`，产物写入 `manifests/`、`normalized/`、`reports/`。
- `scripts/train_bbpe.py` / `scripts/check_tokenizer.py` —— tokenizer 训练与验证（GPT-2 风格 Byte-level BPE，v0 实验用）。
- `tokenizer/` —— 已训练：ccc-bbpe-24k、ccc-bbpe-32k（CCC normalized corpus 训练，见 `training_config.json`）。
- `dataset/` —— 语料目录（见下节）。**已 gitignore**，不入版本库。
- `AGENTS.md` —— 本文件。
- 包管理：`venv` + `pip`（不用 uv/poetry/conda）。新增依赖写入 `requirements.txt`。当前 `.venv/` 已建，装有 opencc-python-reimplemented、huggingface_hub、pyarrow。

## 语料（dataset/ 目录结构，2026-08 整理）
```
dataset/
├── raw/                          # 原始语料，只读，不要改写/移动
│   ├── daizhigev20/              # 殆知阁 v20（原 daizhigev20-master/，zip 仍在 dataset 根）
│   ├── chinese-classical-corpus/ # CCC：corpus/translate/punctuate 三个 jsonl
│   ├── chinese-poetry/           # 中华诗词库（chinese-poetry 主仓库浅克隆）
│   ├── ming-qing-wenji-corpus/   # 明清文集 jsonl
│   ├── erya-dataset/             # 尔雅：monolingual/finetune/trans tgz + extracted/
│   └── greathangpt-classical-chinese/  # 大汉书 parquet（含全部时代分片）
├── manifests/                    # 每个数据集一个文件级 jsonl 清单
├── normalized/                   # CCC 清洗后 ccc_{corpus,translate,punctuate}.jsonl
├── reports/                      # schema/script/missing_char/overlap/extra 报告
└── processed/                    # 训练用产物（tokenized），空目录 + .gitkeep
```

### 殆知阁（daizhigev20）
plain UTF-8（部分带 BOM），**简繁混杂**（正史等多为简体，佛藏/集藏多为繁体），以**全角空格 `　` (U+3000)** 作缩进/段首。不要按 ASCII whitespace 清洗，否则会破坏版式语义。
十藏实际文件数（共 **31,388** 个 txt，非 15.7k）：
- 佛藏 5135 · 史藏 2043 · 集藏 1948 · 道藏 1721 · 子藏 1463
- 医藏 911 · 儒藏 908 · 诗藏 776 · 艺藏 446 · 易藏 343
- 统计：27,070,150 行 / 1,807,620,280 字符（manifest 结果）
- 史藏/正史/ 含二十四史各书 txt（四库本，简体，已断句）；编年/ 含资治通鉴。
- 遵守 `使用须知.md`：研究用途；不要把语料重新分发或商用。

### 新增数据集要点（manifest 与 extra 报告见 reports/）
- **CCC**：简体为主（1.86M/1.98M 记录），corpus 12,009 条（二十四史前 15 部+通鉴+十三经+说文，点校本），translate 1.92M 条，punctuate 46.5k 条。与殆知阁重叠：corpus 92.5%、punctuate 79.8%、translate 10.6%（24 字窗口 t2s 匹配估计，见 ccc_overlap_daizhige_report.json）。
- **chinese-poetry**：JSON 数组文件（全唐诗/宋词/元曲/蒙学等），共 251M 字符。
- **ming-qing-wenji**：单 jsonl，82M 字符，字段 id/dynasty/collection/page/juans/siku_category/author/char_count/text。
- **erya**：src/tgt 古译今平行文本 + monolingual 古文，解压后 2.1B 字符（含译文），训练配比需区分。
- **greathangpt**：9 个 parquet 分时代，550M 字符（char_count 列求和），字段 id/title/author/dynasty/era/content_category/source/text/paragraphs。
- 全库总字符量（6 数据集）：殆知阁 1.81B + CCC 0.15B + 新 4 个 ~3.0B。

## 硬件（关键 gotcha）
可用资源：
- 主力 **8×V100 32G**（持续可用）
- 本地开发 **1×3060**
- **2×A800**（仅短时可用）

**V100 不支持 bf16 tensor core**（Volta 架构，bf16 硬件加速从 Ampere 起）。在 V100 上用 `torch.autocast(bf16)` 会 fallback 到极慢的 fp32 模拟。
→ 训练用 **fp16 + `torch.cuda.amp` + `GradScaler`**；bf16 相关代码只能在 A800/3060 上开。
6×V100 32G 视角：600M–1.5B 用 **DDP + grad checkpoint** 即可放下，不需要 FSDP（DDP 会复制 params/optimizer/grads，1.5B + Adam-fp32 约 ~21GB/卡，32G 紧但可行）。FSDP 等超出 1.5B 再考虑。
启动多卡：`torchrun --nproc_per_node=N`。

## 待办：先把文档整理出来（当前阶段）
用户明确：先把整个项目的文档整理出来再动代码。本文件是第一步；后续应在 `src/jishui/` 落地前补：
- `docs/` 或 `src/jishui/` 下的模型配置说明（layers/heads/d_model 数值，需对齐 Qwen3-1.7B 的尺度）
- tokenizer 训练记录（词表大小、单字 vs BPE merge 的取舍——古汉语像素粒度与 BPE 的选择仍是开放问题；v0 实验数据：ccc-bbpe-24k chars/token=1.37、32k=1.42，生僻字/异体字/□ 均单 token 无损往返）
- 数据流水线设计（清洗/去重/分片/token 量级统计与 mix 配比）

在这些文档就位前，不要先写训练脚本。

## 不要做
- 不要在 `dataset/raw/` 下改写/移动语料（manifest/normalized/reports/processed 由脚本产出，可重建）。
- 不要给 V100 路径开 bf16。
- 不要 `git init` / 提交，除非用户明确要求。
- 不要为后训练或 AI 编译器部分提前建代码——本阶段只管 pretraining。