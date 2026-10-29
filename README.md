# 击水 · Midstream AI / 中流 AI

古汉语原生 decoder-only 语言模型，从零预训练。代码包和 checkpoint 模型标识沿用 `Jishui`。

200M 模型先在 Apple Silicon / MLX 上训练，再迁移至 RTX 3060，以 fp16 AMP + GradScaler
继续训练。2026-10-07 已完成 **244,141 steps / 4,000,006,144 tokens**，完整 checkpoint
位于本地 `runs/jishui-200m-cuda-4b/checkpoints/step_00244141/`。语料和训练产物不随仓库发布。

## 训练代码

| 文件 | 职责 |
|---|---|
| `src/jishui/config.py` / `data.py` | 模型与训练配置、mmap 数据索引、可恢复采样器 |
| `src/jishui/model_torch.py` | PyTorch 模型、RoPE、因果注意力、梯度重算 |
| `src/jishui/pretrain_torch.py` | 单卡 CUDA 续训、验证、梯度累积、溢出重试与温和停止 |
| `src/jishui/checkpoints_torch.py` | MLX 权重及 Adam 状态导入、完整 CUDA checkpoint 保存与恢复 |
| `src/jishui/schedules_torch.py` | 按绝对更新步数计算余弦重启学习率 |
| `src/jishui/tracking.py` | 可选 W&B 指标追踪和恢复关联 |
| `src/jishui/generate_torch.py` | CUDA checkpoint 推理与古文提示词套件 |
| `scripts/train_watch.py` | 只读训练巡检，默认输出本地简报 |
| `scripts/plot_trainlog.py` | MLX / CUDA 日志转 SVG 曲线 |
| `scripts/push_training_dashboard.py` | 独立推送允许公开的训练指标 |

MLX 训练入口为 `jishui.train_mlx`；`jishui.train_torch` 负责分布式 runtime 探测，
正式多卡 trainer 尚未接入。V100 路径使用 fp16，不启用 bf16。

## Linux CUDA 环境

使用 venv + pip；以下命令适用于本项目的 PyTorch 2.6 / CUDA 12.4 环境：

```bash
python3 -m venv .venv-torch
.venv-torch/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
.venv-torch/bin/python -m pip install -e '.[torch,tracking]'
```

准备本地冻结数据、tokenizer 和完整恢复点后，按 [CUDA 续训说明](docs/cuda-continuation.md)
启动。该 trainer 要求已有完整 MLX / CUDA checkpoint；4B 配置是已完成实验的记录，
不能直接从最终 checkpoint 再次续训同一目标。`scripts/start_cuda_4b.sh` 使用 Linux systemd
用户服务托管训练；本次代码整理不启动训练。

```bash
# 查看现有训练状态，不发送通知
.venv-torch/bin/python scripts/train_watch.py --brief

# 从最终 checkpoint 续写古文（CPU 不占用训练 GPU）
.venv-torch/bin/python -m jishui.generate_torch --device cpu --prompt '太史公曰：' --max-new-tokens 64

# 单元测试：Linux 上跳过需要 Apple Silicon MLX 的模块
PYTHONPATH=src .venv-torch/bin/python -m unittest discover -s tests -v
```

## 数据与复现

- [数据生产运行说明](docs/data-production-run.md)：清洗、去重、划分、tokenize 与采样权重。
- [模型配置](docs/model-config.md)：模型结构及规模。
- [CUDA 续训说明](docs/cuda-continuation.md)：迁移、日程、恢复、推理和 W&B。
- [分布式 runtime](docs/torch-distributed.md)：设备探测与 collective smoke。
- [历史交接记录](HANDOFF.md)：MLX / ANE 阶段背景。

`dataset/raw/` 只读，语料的研究用途与使用约束见 [AGENTS.md](AGENTS.md)。
`runs/`、本地通知配置、凭据及网盘元数据均不入版本库。
