# MLX 本机预训练

## 验收结果

```bash
.venv/bin/python scripts/validate_pretrain_data.py --data-dir dataset/processed
```

当前冻结数据验收通过：21 个 `uint16` shard、4,155,137,755 tokens，manifest 无越界、无重叠、无 shard 缺口，抽样 token id 在 0–32767。原始 `docs_tokens.jsonl` 有 252,532 篇（1,243,095,050 tokens）分类字段为空；这是 Stage 5 重建 manifest 时丢失清洗阶段推断元数据造成的，验收脚本会保留这个 warning，不把它伪装成完全干净的数据。

训练索引首次启动时写入 `runs/<run>/cache/pretrain_index.npz`。它不改动冻结 shard 或 manifest：

- 从 `dataset/interim/manifest/docs_final.jsonl` 按原有标题规则恢复 207,544 条 Wikisource 分类；
- 排除仍处于 `quarantine`、无法可靠分类的大汉书 `未分类` 44,988 篇（4,620,051 tokens）；
- 按 `sampling_weights.json` 的六类目标配比重新归一化 sampler 概率；
- 以文档为边界读取 mmap shard，跨文档插入 `<|endoftext|>`（id 2），产出 `(batch, seq_len + 1)` 的输入/标签序列。

因此这里需要一个 MLX 数据加载器，但不需要 PyTorch 多进程 `DataLoader`。MLX 的数据侧只做 NumPy memmap、随机位置抽样和定长 packing，batch 转成 `mx.array` 后直接交给 Metal。

## 安装与启动

本机 `.venv` 已有 MLX 0.32；在仓库根目录执行：

```bash
source .venv/bin/activate
pip install -e . --no-deps
```

两步真实训练冒烟已经通过：

```bash
# 极小模型，验证数据、Metal、optimizer、validation、checkpoint、resume 全链路
PYTHONPATH=src python -m jishui.train_mlx \
  --config configs/jishui-debug-mlx.json \
  --data-dir dataset/processed \
  --run-dir runs/jishui-debug

# Stage 0 正式 200M 配置。可先用短序列确认 Metal/速度，再逐步恢复 seq_len=2048。
PYTHONPATH=src python -m jishui.train_mlx \
  --config configs/jishui-200m-mlx.json \
  --data-dir dataset/processed \
  --run-dir runs/jishui-200m \
  --seq-len 512 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --max-steps 10000

# 本机建议的 1B-token Stage 0 预算（seq_len=2048 时约 61,036 steps）。
# 以当前实测吞吐约 9.2 天理论值，实际按 10–12 天预留。
PYTHONPATH=src python -m jishui.train_mlx \
  --config configs/jishui-200m-mlx.json \
  --data-dir dataset/processed \
  --run-dir runs/jishui-200m-stage0-1b \
  --max-steps 61036
```

正式 Stage 0 配置的默认值是 `seq_len=2048`、`batch_size=1`、`grad_accum_steps=8`、fp16、单 microbatch `mx.compile`，且关闭 gradient checkpointing。本机是 24GB RAM；该配置实测不开 checkpoint 峰值约 6.7 GiB，因此单独运行时不需要 gradient checkpointing。每 `400` 个 optimizer steps 保存一次，代码和配置只保留最近 `3` 份，并让新生成的三份都包含 `model.safetensors`、`optimizer.npz` 和 `train_state.json`。MLX 正式进程和 retention sidecar 当前均已停止，本机不会自动重启；历史 `step_00002000` 是 model-only，`2400`、`2800` 可精确恢复。更长序列、更大 batch 或后续 600M/1.5B 仍应重新评估。MLX/Metal 不受 V100 的 bf16 限制；当前 fp16 Adam 的 `eps=1e-5` 已显式配置，避免 `1e-8` 下溢导致 NaN。

## 训练预算（本机实测）

seq 2048、batch 1、gradient accumulation 8、fp16、单 microbatch `mx.compile`、不开 gradient checkpointing 时，稳定吞吐约 **1,261 tok/s**。下面是按此稳态速度的理论值，未计入降频、验证和 checkpoint I/O：

| 预算 | 训练 tokens | 理论耗时 |
|---|---:|---:|
| 短测覆盖 `max_steps=10,000` | 163.84M | 约 1.5 天 |
| Stage 0 短预算 | 0.1B | 约 0.92 天 |
| 中等预算 | 1B | 约 9.2 天 |
| 全部可用 train split | 4,119,474,408 | 约 37.8 天 |

当前正式配置 `max_steps=61,036` 覆盖约 1B tokens；`max_steps=10,000` 只覆盖约 0.164B tokens，并不等于把 4.1B train split 跑完。无风扇 Mac 的持续降频和周期性验证/checkpoint 会拉长实际墙钟时间；因此 60 天问题是本机持续算力和热约束，不是 24GB RAM 不足。完整 4.1B 训练不建议在本机执行，应迁移到持续可用的 8×V100 32G 或 A800；本机用于 0.1–1B tokens 的 Stage 0 超参、数据配比和断点恢复实验。

这里的 `4,119,474,408` 是索引中可用训练 token 的总量，不是 sampler 的“唯一 token 预算”。默认 `sampling_mode=target` 按六类目标比例放回抽样，类别 5/6 因数据缺口会反复采样；因此 `tokens_seen` 达到 4.1B 时不保证每个原始训练 token 都恰好出现一次。若要做接近自然频率的对照实验，可把配置中的 `sampling_mode` 改为 `weights`，但它同样不是严格无重复 epoch。

断点恢复：

```bash
PYTHONPATH=src python -m jishui.train_mlx \
  --config configs/jishui-200m-mlx.json \
  --data-dir dataset/processed \
  --run-dir runs/jishui-200m \
  --resume latest
```

每个 checkpoint 保存 `model.safetensors`、`optimizer.npz`、`train_state.json`（包含 sampler RNG 状态和已见 tokens），可恢复模型、Adam 和数据位置。正式配置的 `max_checkpoints=3`、`optimizer_checkpoints=3` 会自动只保留最近 3 份，并让新生成的三份都支持精确恢复。历史 `step_00002000` 的 optimizer 已被删除且无法重建；等 `step_00003200` 生成后，保留窗口才会全部是完整 checkpoint。ExFAT 上不使用 Python 3.12 的 fd-relative `shutil.rmtree`，项目采用路径式删除以避免 pruning 失败。

## 实现边界

- `src/jishui/model.py`：标准 MHA/GQA、RoPE、SwiGLU、LayerNorm/RMSNorm、pre-norm、tied embedding。
- `src/jishui/data.py`：索引缓存、类别目标采样、mmap shard、EOD 边界、确定性 validation iterator。
- `src/jishui/train_mlx.py`：MLX `value_and_grad`、梯度累积/裁剪、warmup+cosine、验证、checkpoint/resume。
- 本机训练使用 MLX/Metal；V100 训练路径仍按项目约束使用另行实现的 PyTorch fp16/DDP 脚本，本次没有混用或打开 bf16。

## Checkpoint 续写测试

`generate_mlx` 直接读取 checkpoint 中的模型配置和 dtype，不加载 optimizer，也不会修改训练状态。模型当前只有 pretraining 能力，没有 chat template；提示词应使用自然的古文前缀，而不是问答指令。

```bash
PYTHONPATH=src .venv/bin/python -m jishui.generate_mlx \
  --run-dir runs/jishui-200m-stage0-1b \
  --checkpoint latest \
  --prompt '太史公曰：' \
  --prompt '夫兵者，國之大事，死生之地，存亡之道，' \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-p 0.9 \
  --top-k 40 \
  --repetition-penalty 1.05
```

默认屏蔽未在 pretraining 中出现的 pad/unk/chat special tokens，保留 `<|endoftext|>` 作为停止符。设置 `--temperature 0` 可做 greedy decode；为了比较不同 checkpoint，固定 prompts、seed 和采样参数。

## Apple ANE 实验路径

`native/ane/training_dynamic/` 是对 maderix/ANE 的 Jishui-200M 适配。它通过
Apple 私有 `_ANE*` API 把矩阵密集的 forward/backward-dx 放到 ANE，并可用
`--metal-norm --metal-silu` 把 LayerNorm 和 SiLU backward 放到 Metal GPU；
full-vocab softmax、dW 和 AdamW 仍在 CPU。这是同一 SoC 上的算子级
ANE+GPU+CPU 异构训练，不是模型副本数据并行。Jishui-200M 当前是 MHA
（`num_key_value_heads=11`），虽然内核保留 GQA 支持，但不能把 GQA 的节省
外推到这个模型。

先导出约 34MB 的文档索引，再在线 mmap 原始 NPY shard：

```bash
PYTHONPATH=src .venv/bin/python scripts/export_ane_index.py \
  --data-dir dataset/processed --output /private/tmp/jishui_ane_train.index
cd native/ane/training_dynamic
make MODEL=jishui_200m_2048
./train --scratch --index /private/tmp/jishui_ane_train.index \
  --data-dir /Volumes/PS2000/LLM/Jishui/dataset/processed \
  --optimizer-steps 100 --accum 10 \
  --metal-norm --metal-silu
```

ANE checkpoint v6 默认每个 `--save-interval` optimizer update 原子写入，并只保留最近
3 份；`--resume --checkpoint PREFIX` 会自动选择最新文件并恢复模型、Adam、
accumulation 和 schedule 超参。`--warmup` 按 optimizer update 计数。短测必须
加 `--no-checkpoint`，否则结束时仍会写约 2.4GB 的完整状态。生产长度的
`--metal-shadow` 已通过真实 index 的 LayerNorm/SiLU 逐算子容差校验和
`accum=2` Adam gate；Metal/CPU 首步 loss 为 `10.4381294/10.4381037`，不是
bitwise equal。token-major classifier 与有界 dW 队列落地后，冷机同输入
配对为 5.6265/4.0101 秒（约 1.40x）。MPS classifier/dW GEMM 实测慢于
Accelerate，且异步 CPU dW 没有暴露等待，未接入。长期运行需保持 ExFAT
外置盘在线。
