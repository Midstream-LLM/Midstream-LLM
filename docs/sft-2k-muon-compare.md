# 2K Muon 基座的 Muon SFT（2026-11-26）——严格按配置执行

目的：用 2K Muon 基座的最终权重（step 63,784，固定验证 loss 2.987）做一次 SFT，优化器用 Muon，超参按 Muon 调整。然后用 11 月 25 日的同一套评测，和两个已有 SFT 作参考对比：

| 已有结果（都是 AdamW SFT） | 基座 | 全测试集 CE | 材料摘录 | 余数题 |
|---|---|---:|---:|---:|
| 旧 2K SFT（11-24，step 1434） | 2K blocks，AdamW 预训练 | 1.7364 | 495/500 | 1/100 |
| 32K SFT（11-24，step 936） | Release3 32K | 1.8116 | 480/500 | 0/100 |

这次**不是**对旧配方的严格复现：基座、优化器和学习率都换了，对比结果只反映整套新流程的效果。但「严格按配置执行」这一点不变。

## 规则

1. 所有影响结果的参数都写在 `configs/sft-v1-simplified-2k-muon.json` 里，**不要改动这个文件**。命令行也不能覆盖其中的配方参数，`sft_torch` 会直接报错。只允许用 `PYTHON`、`GPU` 两个环境变量指定解释器和使用哪张卡。
2. 配置中的 `expect` 一节会在训练前逐项核对：数据指纹、分词器 sha256、train/val/test 条数、父 checkpoint 的 step 和模型结构。**任何一项不符就停下来报告，不要为了让它通过而改配置或改数据。**
3. 单卡训练（默认 `GPU=0`）。不要用 torchrun。
4. 评测脚本的输出中只要出现 `STOP:` 就必须停下来报告，不要绕过。
5. 训练中如果 loss 出现 NaN，或 val loss 连续两次评测上升超过 0.3，先停下来报告，不要自行改学习率重跑。

## 前置条件（在一号机仓库根目录）

- 代码：`git fetch origin && git checkout feat/doc-attention-8xv100 && git pull`。
- 父权重：`/data/jishui-docmask/runs/jishui-200m-v3-docmask-muon-8xv100/checkpoints/step_00063784/`，需要包含 `training.pt` 和 `train_state.json`。
- SFT 数据：`dataset/sft/sft_v1_simplified/compact_candidate_37_simplified/`，与 11-24 两次 SFT 用的是同一份。如果本机没有，从 corpus 服务器原样拷贝这个目录，不要重新导出。用 `check` 核对指纹，应为 `fc098eca…9662fd`。
- 分词器：仓库内 `tokenizer/ccc-bbpe-32k/tokenizer.json`，sha256 为 `7cac6a27…0e26`。
- Python 环境需要 torch、numpy、pyarrow、tokenizers。
- 磁盘：8 个 checkpoint，每个约 2.4 GB，共约 19 GB。

## 步骤

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv-torch/bin/python -m unittest tests.test_sft_torch tests.test_muon
bash scripts/sft/run_sft_2k_muon.sh check     # 必须打印 CHECK OK；记下打印的每轮步数
bash scripts/sft/run_sft_2k_muon.sh smoke     # 20 步；记录峰值显存、每步耗时、grad_norm
bash scripts/sft/run_sft_2k_muon.sh start     # 正式训练，约 1,434 步
bash scripts/sft/run_sft_2k_muon.sh eval      # 评最终 checkpoint，输出 runs/evaluations/sft-2k-muon/report.md
```

- 训练中断后用 `bash scripts/sft/run_sft_2k_muon.sh resume` 恢复，会严格接上；不要删掉 run 目录重新开始。
- 训练结束后，查 `metrics.jsonl` 里 val loss（`all`）最低的那一步。如果它不是最后一步，再评一次那个 checkpoint：`CKPT=runs/jishui-200m-sft-v1-simplified-2k-muon/checkpoints/step_000XXXXX bash scripts/sft/run_sft_2k_muon.sh eval`，结果会写到 `runs/evaluations/sft-2k-muon-step_000XXXXX/`。

## 配方（按 Muon 调整）

- 数据：sft_v1_simplified / compact_candidate_37_simplified，train 140,011 条，val 6,859 条，test 6,965 条。loss 只计入助手正文和 `<|im_end|>`。
- 训练量：2 个 epoch，每个 micro-batch 补齐后 4,096 个 token，grad_accum 8，约 1,434 次更新（具体步数由 seed 1234 决定的 batch 规划算出）。
- 优化器：
  - transformer 层里的 2D 矩阵用 Muon：momentum 0.95、Nesterov、Newton–Schulz 5 步，与预训练相同。更新幅度已对齐到 AdamW，所以两组共用一个学习率。
  - tied embedding 用 AdamW，带 weight decay；norm 和 bias 用 AdamW，不带 weight decay。AdamW 的 betas 为 0.9/0.95，eps 1e-5。
- 学习率：峰值 **1e-4**，warmup 50 步（约 3.5%），之后按 cosine 降到 **1e-5**。峰值取预训练峰值 5e-4 的 1/5，是旧 AdamW SFT 的 2 倍；Muon 的更新幅度与 AdamW 对齐，这个量级可以沿用 AdamW 的经验。
- weight decay **0.1**，与 Muon 预训练相同；grad clip 1.0；fp16 AMP + GradScaler。
- 每 200 步评一次 val、保存一次 checkpoint，全部保留（8 个）。学习率比上次高，可能更早过拟合：上次数学类在第 600 步就到了最低点，所以保留全部 checkpoint，方便事后挑 val loss 最低的那个。
- 这组参数是根据经验定的，没有做过扫描。如果结果明显过拟合或欠拟合，下一轮再调学习率。

## 需要回传的内容

- `runs/jishui-200m-sft-v1-simplified-2k-muon/`：`metrics.jsonl`、`final.json`、`val_generations.json`，以及第 0 步的 val loss。旧 2K 基座当时是 3.027，32K 基座是 3.143。
- `runs/evaluations/sft-2k-muon/`（如果还评了 val 最低的 checkpoint，连同那个目录）：`report.md`、`summary.json`、`info.json`（含权重 sha256、GPU 型号、SFT 优化器）、`generations.json`、`remainder-audit.json`。
- smoke 记录的峰值显存、速度和 grad_norm，以及任何 `STOP:` 或报错的完整输出。

## 解读注意

- 旧的两个评测是在 3060 上算的，这次在 V100 上算。fp16 autocast 在不同显卡上的数值误差约在 1e-3 量级，远小于要比较的差异。
- 2K Muon 基座只训了 2 轮（41.8 亿 tokens）。两个基座的预训练验证 loss 不能直接比，因为验证协议可能不完全一致（这次计分 260,118 个 token，旧 run 是 259,906 个）。以 SFT 测试集上的结果为准。
