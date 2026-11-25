# 文档内注意力隔断 + Muon + 8×V100 DDP 预训练（2026-11-25）

## 为什么

Release3 32K 基座（step 15,946）做 SFT 后不如 2K 基座。两个基座在 SFT 第 0 步的验证 loss 就差约 0.12（3.143 vs 3.027）。可能的原因有两个：

1. 优化器更新太少。32K 每步约 26 万 tokens，4.18B tokens 只更新了 15,946 次；2K 更新了约 24 万次。
2. 跨文档注意力。语料平均每篇约 1,760 tokens，一个 32K 窗口里约有 18 篇，用 EOD 拼接，原先只有因果 mask，不同文档可以互相看到。8K→32K 窗口的 loss 只改善了 0.011。

本轮两点都改：每篇文档只看自己（文档内注意力隔断），并在 8 卡上用 2K 窗口跑足够多的更新次数。

## 改了什么

- `src/jishui/model_torch.py`
  - `forward(input_ids, segment_ids=None)`：`None` 时与原先完全相同。SFT、推理和旧 checkpoint 不受影响，模型参数也没有变化。
  - 传入 `segment_ids` 后，只允许同一 segment 内的因果注意力。
  - 后端由 `JISHUI_ATTENTION_BACKEND=auto|dense|flex` 选择：
    - `auto` 在长度 ≤ 8192 时用 dense（SDPA 加布尔 mask，在 V100 上走 mem-efficient kernel），更长时用 FlexAttention。
    - dense 超过 8192 会直接报错，因为 mask 需要 B×L×L 字节。
- `src/jishui/pretrain_torch.py`
  - 配置项 `pretraining.attention: "causal" | "document"`，默认 `causal`，只能与 `shuffled_blocks` 一起使用。
  - 该字段属于 `pretraining`，resume 时会比对，不能中途切换。
  - **segment 定义**：每个输入位置的 segment 取它预测目标所在的文档（`document_ids[:, 1:L+1]`）。效果是上一篇结尾的 EOD 输入位置被并入下一篇，充当下一篇的 BOS：下一篇的第一个字只由这个 EOD 预测，与推理时从头生成一致。
  - **DDP**（用 `torchrun` 启动，NCCL 后端；CPU 测试用 gloo）：
    - 所有 rank 用同一个确定性 sampler 抽取全局 micro-batch（`batch_size × world_size` 行），各取自己那一段。因此 sampler 状态、token 计数、学习率日程、checkpoint 都与单进程读同一个 batch 完全相同。
    - loss 按全局有效 token 加权。
    - 梯度累积期间用 `no_sync`。
    - 固定验证集按样本轮流分给各 rank，再 all-reduce 汇总。
    - 只有 rank 0 写 checkpoint、日志、索引缓存和 W&B。
    - 收到停止信号后，所有 rank 在同一步一起停。
    - 改变 world_size 的 resume 会被拒绝。
- `src/jishui/checkpoints_torch.py`：DDP 下只保存和恢复本卡的 CUDA RNG，避免 rank 0 在所有卡上都建立 CUDA context。单卡行为不变。

- `src/jishui/optim_torch.py`（Muon）：
  - 配置项 `pretraining.optimizer`，取 `{"name": "muon", ...}`；省略或 `adamw` 时与原先的 fused AdamW 相同。
  - 同一个优化器内部分两组：transformer 层里的 2D 矩阵（q/k/v/o、gate/up/down）用 Muon；tied embedding（同时是输出层）、LayerNorm 权重和 bias 用 AdamW，超参取 `training` 里的 beta、eps。
  - Muon 对 Nesterov 动量做 5 步五次 Newton–Schulz 正交化，更新量乘 0.2·√max(行,列)，使其 RMS 与 AdamW 对齐（Moonlight 论文的做法）。因此两组共用 `learning_rate`、学习率日程和解耦的 weight decay，不需要单独设 Muon 学习率。
  - Newton–Schulz 在 CUDA 上用 fp16（V100 张量核；矩阵先归一化，数值范围安全），CPU 上用 fp32。200M 模型每步约 3.2 TFLOP，fp32 在 V100 上约 0.25 秒，fp16 约几十毫秒。DDP 下各 rank 的梯度相同，各自重复计算同一份更新，没有拆分到不同 rank。
  - 仍然只有一个 optimizer state_dict，GradScaler、checkpoint 和严格 resume 都不变。resume 时如果换了优化器会被拒绝；换优化器需要用 `--init-from` 开新 run，或者从零开始。

## 测试（`tests/test_document_attention.py`，10 项）

- 打包后加隔断的前向，与每篇文档单独前向（`[EOD, 正文...]`）的 logits 一致；不加隔断时两者不一致（证明测试夹具确实会泄漏）。
- 梯度不会跨文档；只有一篇文档时与纯因果结果相同；padding 输出有限值，且不影响真实 token。
- flex 与 dense 的结果一致（CPU eager）。
- 2 进程 gloo DDP（每卡 1 行）与单进程（2 行）训练到结束：权重一致（rtol 1e-5），step、tokens、sampler、验证协议完全相同。causal 和 document 两种模式都做了。
- `tests/test_muon.py`（13 项）：
  - Newton–Schulz 正交化：长方形和方阵、两种朝向都测了；fp16 与 fp32 结果的相对误差 < 2%。
  - 更新 RMS 约为 0.2。
  - AdamW 分支与 `torch.optim.AdamW` 结果一致。
  - 参数分组覆盖全部参数，且每个参数只出现一次。
  - state_dict 往返后继续训练逐位相同。
  - 2 进程 DDP + Muon 与单进程结果一致。
  - 训练中途恢复后逐位相同；换优化器的 resume 会被拒绝。
- 本地还跑了原有的 39 项测试，全部通过（合计 62 项，跳过 1 项）。**尚未在 V100 上实测**：dense 后端的速度和显存、NCCL 多卡运行都还没验证。

## 配置 `configs/jishui-200m-v3-docmask-8xv100.json`

- 模型与 2K 基座相同：200M，2048 上下文，RoPE θ=1e4。从零开始训练。
- 每卡 batch 4 × 2048，accum 1，全局每步 65,536 tokens。
- 预算 4,180,100,928 tokens（与 32K run 相同，约 2 epoch），约 63,784 次更新，是 32K run 的 4 倍。
- Muon（momentum 0.95，Nesterov，5 步）+ AdamW（embedding 和 norm），峰值学习率 5e-4 → 5e-5 cosine，warmup 1000，weight decay 0.1，fp16 AMP + GradScaler。因为 Muon 的更新幅度已和 AdamW 对齐，5e-4 沿用的是 AdamW 的量级。**这是第一次在本项目里用 Muon，前几千步要盯紧 loss 和 grad_norm。** 如果发散，先把学习率降到 3e-4 重新开始。
- 固定验证：`eval_batches 32 × batch 4 = 128` 块，seed 20261118，SampleSpec 与 2K blocks run 相同，所以验证协议指纹应当一致，可以直接和 2K run 的 3.237 比较。**启动日志里打印的 protocol 指纹要核对一下。**
- 显存估计：每卡约 15 GiB（3060 上 batch 1 的峰值是 6.3 GiB）。如果 OOM，把 `batch_size` 改成 2、`grad_accum_steps` 改成 2，全局 batch 不变；但验证块数会跟着变，需要同时把 `eval_batches` 改成 64。

## 服务器操作

```bash
git fetch origin && git checkout feat/doc-attention-8xv100 && git pull
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv-torch/bin/python -m unittest \
  tests.test_document_attention tests.test_muon tests.test_corpus_pretraining tests.test_block_data tests.test_torch_continuation
bash scripts/train_8xv100_docmask.sh smoke     # 20 步，写入带时间戳的临时 run 目录
bash scripts/train_8xv100_docmask.sh start     # 正式开始
bash scripts/train_8xv100_docmask.sh resume    # 中断后严格恢复
```

- 可用环境变量覆盖：`PYTHON`、`DATA_DIR`（默认 `dataset/corpus-date-open-v3`）、`RUN_DIR`（默认 `runs/jishui-200m-v3-docmask-8xv100`）、`NPROC`。
- smoke 阶段要核对：
  - 8 个 rank 都正常启动；
  - `world_size=8 attention=document`；
  - 初始验证 loss 有限；
  - tok/s 和 `peak_vram_gib` 合理；
  - checkpoint 能写出，`--resume latest` 能接上。
- 不要改动已结束的旧 run。

## 未做

- 32K 长上下文阶段：应在本 run 之后，用很少的 token 做长度扩展，并上采样长书。长文档的采样还没实现。
- FlexAttention 在 V100 上没有验证过（依赖 Triton）。2K 用不到它。
- 已有的 32K 基座用加隔断与不加隔断各评一次验证 loss（不重训的诊断），没有做。
