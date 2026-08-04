# AGENTS.md — Jishui（集水）

古汉语原生 LLM 从零预训练项目。读这一节之前，agent 不会知道这些事。

## 项目性质
- **目标**：从零训练一个古汉语原生的小型 decoder-only LLM，先 ~600M，再 scale 到 ~1.5B。
- **技术栈**：PyTorch；标准 MHA + RoPE；结构参考 Qwen3-1.7B；自训 BPE tokenizer。
- **阶段**：当前只做 pretraining。后训练 (SFT/RL 等) 暂缓。
- **两部分分工**（团队 3 人）：①模型训练（当前）；②自研 AI 编译器 / infra（后续，不在本阶段范围）。
- 这是 hobby project，以学习为目的；用现成技术，不追新。

## 仓库布局（约定，尚未落地）
- `src/jishui/` —— 单 Python 包（src-layout）；模型 / tokenizer / 数据 / 训练脚本都在其下。
- `dataset/` —— **只读**语料，不要在此目录写任何东西、不要重命名/移动文件。
- `AGENTS.md` —— 本文件。
- 还未 `git init`；当前没有任何代码、配置、锁文件。
- 包管理：`venv` + `pip`（不用 uv/poetry/conda）。新增依赖写入 `requirements.txt`。

## 语料（dataset/daizhigev20-master/）
殆知阁古代文献 txt 大全集（v20），plain UTF-8，**繁体/古汉语**，以 **全角空格 `　` (U+3000)** 作缩进/段首。不要按 ASCII whitespace 清洗，否则会破坏版式语义。
十藏分布（文件数，约 15.7k 个 txt）：
- 佛藏 5135 · 史藏 2043 · 集藏 1948 · 道藏 1721 · 子藏 1463
- 医藏 911 · 儒藏 908 · 诗藏 776 · 艺藏 446 · 易藏 343
- 同目录下的 `daizhigev20-master.zip` 是原始压缩包，勿删勿改。
- 遵守 `使用须知.md`：研究用途；不要把语料重新分发或商用。

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
- tokenizer 训练记录（词表大小、单字 vs BPE merge 的取舍——古汉语像素粒度与 BPE 的选择仍是开放问题）
- 数据流水线设计（清洗/去重/分片/token 量级统计与 mix 配比）

在这些文档就位前，不要先写训练脚本。

## 不要做
- 不要在 `dataset/` 下写文件或改语料。
- 不要给 V100 路径开 bf16。
- 不要 `git init` / 提交，除非用户明确要求。
- 不要为后训练或 AI 编译器部分提前建代码——本阶段只管 pretraining。