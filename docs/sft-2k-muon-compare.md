# 2K Muon 基座 SFT 对比（2026-11-26）——严格执行

目的：用 2K Muon 基座的最终权重（step 63,784，固定验证 loss 2.987），按 **11 月 24 日 2K SFT 的原配方**原样做一次 SFT，再用 11 月 25 日的同一套评测，与两个已有 SFT 对比：

| 已有结果 | 基座 | 全测试集 CE | 材料摘录 | 余数题 |
|---|---|---:|---:|---:|
| 旧 2K SFT（11-24，step 1434） | 2K blocks，AdamW，step 239,757 | 1.7364 | 495/500 | 1/100 |
| 32K SFT（11-24，step 936） | Release3 32K，step 15,946 | 1.8116 | 480/500 | 0/100 |

**只换基座，其他一律不变。** SFT 优化器仍是 AdamW，与两个已有 SFT 保持一致，不改用 Muon。

## 规则

1. 所有影响结果的参数都写在 `configs/sft-v1-simplified-2k-muon.json` 里，**不要改动这个文件**。命令行也不能覆盖其中的配方参数，`sft_torch` 会直接报错。只允许用 `PYTHON`、`GPU` 两个环境变量指定解释器和使用哪张卡。
2. 配置中的 `expect` 一节会在训练前逐项核对：数据指纹、分词器 sha256、train/val/test 条数、每轮步数 `[717, 717]`、父 checkpoint 的 step 和模型结构。**任何一项不符就停下来报告，不要为了让它通过而改配置或改数据。**
3. 单卡训练（默认 `GPU=0`），与原配方的单卡 batch 规划一致。不要用 torchrun。
4. 评测脚本的输出中只要出现 `STOP:` 就必须停下来报告，不要绕过。

## 前置条件（在一号机仓库根目录）

- 代码：`git fetch origin && git checkout feat/doc-attention-8xv100 && git pull`。
- 父权重：`/data/jishui-docmask/runs/jishui-200m-v3-docmask-muon-8xv100/checkpoints/step_00063784/`，需要包含 `training.pt` 和 `train_state.json`。
- SFT 数据：`dataset/sft/sft_v1_simplified/compact_candidate_37_simplified/`，与 11-24 两次 SFT 用的是同一份。如果本机没有，从 corpus 服务器原样拷贝这个目录，不要重新导出。用 `check` 命令核对指纹，应为 `fc098eca…9662fd`。
- 分词器：仓库内 `tokenizer/ccc-bbpe-32k/tokenizer.json`，sha256 为 `7cac6a27…0e26`。
- Python 环境需要 torch、numpy、pyarrow、tokenizers。

## 步骤

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv-torch/bin/python -m unittest tests.test_sft_torch
bash scripts/sft/run_sft_2k_muon.sh check     # 必须打印 CHECK OK ... steps [717, 717]
bash scripts/sft/run_sft_2k_muon.sh smoke     # 20 步；记录峰值显存和每步耗时
bash scripts/sft/run_sft_2k_muon.sh start     # 正式训练，1,434 步
bash scripts/sft/run_sft_2k_muon.sh eval      # 生成 runs/evaluations/sft-2k-muon/report.md
```

训练中断后用 `bash scripts/sft/run_sft_2k_muon.sh resume` 恢复，会严格接上；不要删掉 run 目录重新开始。

## 配方（与 11-24 旧 2K SFT 相同）

- 数据：sft_v1_simplified / compact_candidate_37_simplified，train 140,011 条，val 6,859 条，test 6,965 条。
- 训练量：2 个 epoch，每个 micro-batch 补齐后 4,096 个 token，grad_accum 8，共 1,434 次更新。
- 优化：AdamW（betas 0.9/0.95，eps 1e-5），weight decay 0.01（不作用于 norm 和 bias）。学习率 warmup 50 步到 5e-5，再按 cosine 降到 5e-6；grad clip 1.0；fp16 AMP + GradScaler。
- 其他：seed 1234，每 200 步评一次 val，每 400 步保存一次，保留 2 个 checkpoint。
- loss 只计入助手正文和 `<|im_end|>`。

## 需要回传的内容

- `runs/jishui-200m-sft-v1-simplified-2k-muon/`：`metrics.jsonl`、`final.json`、`val_generations.json`，以及第 0 步的 val loss。旧 2K 基座当时是 3.027，32K 基座是 3.143。
- `runs/evaluations/sft-2k-muon/`：`report.md`、`summary.json`、`info.json`（含权重 sha256、GPU 型号）、`generations.json`、`remainder-audit.json`。
- smoke 记录的峰值显存和速度，以及任何 `STOP:` 或报错的完整输出。

## 解读注意

- 旧的两个评测是在 3060 上算的，这次在 V100 上算。fp16 autocast 在不同显卡上的数值误差约在 1e-3 量级，远小于要比较的差异。
- 2K Muon 基座只训了 2 轮（41.8 亿 tokens），比旧 2K 基座（约 40 亿 tokens、24 万次更新）的总量相当、更新次数少。基座 loss 的对比要注意两者验证协议可能不完全一致（这次计分 260,118 个 token，旧 run 是 259,906 个），以 SFT 测试集为准。
