# HANDOFF — Jishui 数据管线完成，MLX Stage 0 已落地（2026-08-09）

> 本文件是给下一个 agent 的交接说明。先读 `AGENTS.md`、本文件、`dataset/processed/README.md`、`docs/data-production-run.md`，再动手。

## 1. 当前状态（一句话）

Jishui 古汉语预训练**数据生产管线已全量跑通并冻结**（2026-08-07）：inventory → clean → 五级去重 → 作品级 split → tokenize（ccc-bbpe-32k，21 shard，**41.55 亿 tokens**）→ 15 项统计 → 采样权重。MLX 本机训练链路已落地并验证（200M MHA + RoPE + SwiGLU + LayerNorm、mmap sampler、梯度累积、`mx.compile`、验证、checkpoint/resume），但本机正式进程已由用户停止，不要自动重启。Apple ANE 路径已实现 ANE+Metal+CPU 算子级异构训练并通过真实 shard 的 forward/backward/Adam gate；短测提速随热与换页状态从近乎持平到约 **1.40x**，尚无稳定长跑结论。完整 4.1B train-token 训练仍应迁移到 V100/A800。

## 2. 最终产物（训练直接可用）

| 路径 | 内容 |
|---|---|
| `dataset/processed/tokens/shard_00000.npy … 00020.npy` | uint16 token 数组，每 shard ~200M tokens，文档不跨 shard |
| `dataset/processed/tokens/shard_info.jsonl` | shard 清单（路径 + token 数） |
| `dataset/processed/manifest/docs_tokens.jsonl` | 每文档：doc_id/source/category/era/tokens/shard/offset/split |
| `dataset/processed/split/{train,val,test}.jsonl` + `split_manifest.json` | 划分（train 1,867,362 / val 3,671 / test 1,123） |
| `dataset/processed/sampling_weights.json` | 六类目标配比 → 类别/源采样权重 + 缺口 |
| `/private/tmp/jishui_ane_train.index`（可重建） | ANE v2 train 文档索引；不复制 token payload，约 34MB |
| `dataset/processed/README.md` | 数据读取说明（含代码示例） |
| `dataset/reports/pipeline/token_stats.json` | 15 项统计（全量流式计算） |
| `dataset/reports/pipeline/report.md` | 报告 + 人工审核清单 |
| `dataset/interim/final/*.jsonl` | 可读最终文本（9 源，48.73 亿字符；中间产物可重建） |
| `dataset/interim/dedup/l{1,2,3,3n,4}*.jsonl` | 五级去重审计记录 |

关键数字：inventory 4,373,240 文档 / 62.6 亿字符 → clean 保留 1,872,156 → 去重后 1,872,156 文档 / 48.73 亿字符；tokenize 后 1,836,149 篇（其余为空文本）。L1 8 / L2 1,947 / L3 段落 41,281,261 条 / L3n 80 段（66 篇）/ L4 0 篇；L5 交叉去重 27 篇移回 train。

## 3. 硬件与环境（必须先知道）

- **本机是 24GB RAM、无风扇 Mac**：MLX Stage 0（200M、seq 2048、batch 1、不开 gradient checkpointing）实测峰值约 6.7 GiB，内存不是当前瓶颈；持续满载仍会降频，速度会周期性波动。数据管线/报告继续采用流式处理，避免一次性加载大数据。
- **外置盘 `/Volumes/PS2000` 是 ExFAT**：休眠唤醒后可能掉挂载或只读。掉挂载用 `diskutil mount /dev/disk4s1`（需提权）；只读时 `mount -u -o rw`（本机曾靠用户手动处理）。**工作前先 `touch` 验证可写。**
- **sqlite 索引放内置盘比外置盘快 2–3 倍**：用环境变量 `JISHUI_PARA_DB` / `JISHUI_WIN_DB` 指到 `/private/tmp/`（本机 05_dedup 就是这么跑的）。
- **`.venv` 现状**：python3.12；已装 numpy 2.5.1、transformers 5.14.1、MLX 0.32、opencc-python-reimplemented、huggingface_hub、pyarrow。**torch 未安装**；若启用 V100 的 PyTorch 训练路径，再 `pip install torch` 并更新 `requirements.txt`。
- **2026-08-09 磁盘审计**：`/private/tmp/` 早先有 4 份 disposable ANE 一步 checkpoint（每份 2,402,695,776 bytes，合计约 9.0GiB），当前已不在目录中且不可恢复；`/private/tmp` 约 204MiB。native trainer 已增加 `--no-checkpoint`，后续 disposable gate 必须使用。“突然又满”的直接原因是这 4 份文件叠加 ANE/MLX 并发造成的 swap 膨胀：swap 曾达 13.31GiB，审计时仍用约 7.7GiB；释放压力后内置 Data 约 420/460GiB、可用约 9–11GiB（swap 释放有延迟）。长期占用主要是 `~/Desktop` 197G（`kitti_data` 97G、`Huawei Challenge` 67G、`ugpnet_weights` 25G）和 `~/Library` 78G（Application Support 40G、Containers 23G、Caches 6.8G）。另有 `/private/var/folders` 约 2.2G，其中 Chrome code-sign clone 约 1.3G、Metal/cache 约 0.7G；APFS 还有一个 Apple 系统更新 sealed snapshot，不能手工删除。外置 PS2000 为 ExFAT、已用 99%、约剩 22GiB；大项是 `datasets` 533G、`FlightGear` 310G、`LLM` 246G、`地下停车场数据` 171G、`Huawei Challenge` 165G，仓库本身约 92G。未发现 GiB 级“已删除但仍被进程占用”的文件。
- **第三方 ANE probe 禁区（2026-08-09）**：不要在本机运行 `imperatormk/ane-train` 的 `modules/test_bwd` 或其他面向 macOS 26 的私有 ANE binary。本机是 macOS 15.7.1；静态符号映射显示用户态崩溃点为 `test_bwd.m:163` 对空 `IOSurface` 映射的写入（lock/base 返回值被忽略），但同一时间窗的 AMCC 非法 DMA 记录更可能是先发生的根因，空写是次生故障。随后出现系统级 `SOCD report detected: (iBoot async abort)` / `AMCC error` panic；15:50 与 16:00 两次 panic 的 AMCC 地址相同。issue #47 只作为源码/约束参考；后续新边界先做 Metal-only 测试，禁止直接执行未经过本仓库兼容性约束的外部 ANE kernel。
- 内置 SSD 也曾被 OpenCode 的 snapshot 机制写满（崩溃遗留 ~13GB `tmp_pack_*`，已清理）；当前 `~/.local/share/opencode/` 约 2.7G。

## 4. 管线脚本与断点机制

脚本在 `scripts/pipeline/`：`01_acquire_gutenberg.py`、`02_inventory.py`、`03_clean.py`、`05_dedup.py`、`06_split.py`、`07_tokenize_stats.py`、`08_report.py`、`common.py`。

- 全量复现：`bash scripts/pipeline/run_all.sh`；断点续跑：`bash scripts/pipeline/resume_dedup.sh`（05→06→07→08）。
- **05_dedup 断点**：`interim/dedup/l3_done.json`（完成的源）+ `l3_progress.json`（每 5000 文档落盘的 last_doc）。中途被杀只重做最后几千文档。
- **05 幂等**：`l4_doc_near.jsonl` + `docs_final.jsonl` 存在时 L4 自动跳过；L1/L2 记录存在时跳过重算。
- 07 分两阶段：A 多进程编码写 `.bin`（流式，勿改成全量内存），B memmap 组装 shard。
- 08_report 必须流式统计（41M 条 L3 记录全量载入会 OOM）；这条是管线的内存约束，与 MLX Stage 0 的 24GB 训练余量无关。

## 5. 本轮踩过的坑（避免重犯）

1. **OpenCC 逐段落转换是 L3 瓶颈**：改成按文档 `"\n".join(paras)` 一次转换再 split，已验证与逐段转换 50k 段 0 差异。worker 内 OpenCC 实例各建一份。
2. **L3 跳过断点的 bug**：曾写错 skip 逻辑导致从头重跑并写坏 final 文件；正确做法是 `waiting` 标志——跳过直到遇到 checkpoint doc（含它本身），之后才处理。
3. **macOS multiprocessing 是 spawn**：worker 必须是模块级函数；不要用 heredoc/stdin 跑 `mp.Pool`（会 FileNotFoundError）。
4. **L4 窗口索引不要放外置盘**（曾 2 小时+）；内置盘 ~40 分钟。L4 只验证采样窗口命中 ≥8 的前 12000 对（近似，已写入报告已知问题）。
5. **docs_clean_norm.jsonl 没有 chars_clean 字段**（05 重写时丢失）：08 的“清洗后字符”改为 `最终字符 + 各级去重删除字符` 推导，勿回退到读字段。
6. **磁盘容量**：全流程峰值需 ~30GB 外置盘余量；完成后 `dataset/interim/clean/`（16GB）与 sqlite 索引可删（可重建）。当前外置盘约 22GiB；内置盘会随 swap 在接近 0 到约 11GiB 间波动。不要让短 gate 默认写 2.4GB native checkpoint，也不要在正式 MLX 旁长期并发跑 native trainer。

## 6. 下一步任务（当前训练阶段）

1. MLX 本机链路到此收口，不要自动继续实验或重启训练。正式配置仍是 `max_steps=61036`（约 1B tokens）、`save_interval=400`、`max_checkpoints=3`、`optimizer_checkpoints=3`；以后最近三份都会保留模型、Adam 和训练状态。
2. MLX trainer 和 retention sidecar 当前均未运行。现存 `2000` 是历史 model-only checkpoint，`2400`、`2800` 是完整 checkpoint；应从最新的 `2800` 精确恢复。`2000` 的 optimizer 无法补造，等未来产生 `3200` 后，保留窗口才会自然变成三份完整 checkpoint。本机稳态约 **1,261 tok/s**；0.1B ≈ 0.92 天，1B ≈ 9.2 天，4,119,474,408 个可用 train tokens ≈ 37.8 天理论值。
3. **60 天不是 24GB 内存问题，而是本机 Metal 算力/持续降频问题。** 若目标是完整 4.1B train-token 预训练，迁移到持续可用的 8×V100 32G 或短时 A800；V100 路径用 fp16 + `torch.cuda.amp` + `GradScaler`，600M/1.5B 用 DDP + grad checkpoint。
4. MLX 链路已包含 sampler、模型、验证和 checkpoint/resume；PyTorch V100 训练脚本仍需另行实现，新增依赖写 `requirements.txt`。
5. ANE 入口在 `native/ane/training_dynamic/`：`scripts/export_ane_index.py` 导出索引，`./train --index ... --data-dir ...` 在线 mmap 21 个 NPY shard；checkpoint v6 原子写入并只保留最近 3 份，保存 accumulation、warmup、LR/Adam/clip/loss-scale 等恢复关键配置，旧 v5（含错误 residual 实验）会被拒绝。`--warmup` 单位是 optimizer update，末尾 partial accumulation 已正确按有效 microstep 数归一化；短测必须用 `--no-checkpoint`。当前 NPY/index 是 uint16 token 格式，因此 151,936 词表的 Qwen3 reference header 会在编译期明确拒绝，不能拿来做实际 GQA 训练。
6. ANE+GPU 使用 `make MODEL=jishui_200m_2048` 后加 `--metal-norm --metal-silu`。映射为 ANE transformer/attention+dx、Metal LayerNorm/SiLU、CPU classifier/softmax/dW/Adam；这是单 SoC 算子级异构，不是 DDP。token-major classifier 与有界 dW 队列落地后，冷机配对为 **5.6265s → 4.0101s（1.40x）**；另一次真实 index 三更新 gate 为 **5.555s → 4.680s（15.8%）**，第三步 loss 相对差 `7.4e-4`、采样权重最大绝对差 `1.41e-5`。重 swap 场景曾只有 20.654s → 20.467s，不能据短测推算 overnight 吞吐。逐算子 LayerNorm/SiLU shadow gate 已通过；MPS classifier/dW GEMM 比 Accelerate 慢，且现有异步 CPU dW 可隐藏，未接入 GPU。完整模型双副本 data parallel 会复制训练状态并争用统一内存/散热，不建议在这台 24GB 无风扇 Mac 上继续实现。

## 7. 约束（AGENTS.md）

- `dataset/raw/` 只读，不改写/移动；manifest/normalized/reports/processed 可重建。
- 不要 `git init`/提交（除非用户明确要求）。
- 本阶段只管 pretraining；不要提前写 SFT/RL 或 AI 编译器代码。

## 8. 参考

- `AGENTS.md`（项目性质/语料/硬件 gotcha）
- `docs/model-config.md`（200M/600M/1.5B 配置）
- `docs/tokenizer.md`（ccc-bbpe-32k 决策）
- `docs/data-pipeline.md`（流水线设计，§6 落地顺序）
- `docs/data-production-run.md`（运行说明 + 2026-08-07 实测结果）
- `dataset/reports/pipeline/report.md`（15 项统计 + 人工审核清单）
- `native/ane/README.md` / `native/ane/hybrid/README.md`（ANE+Metal 运行方式、A/B 与限制）
- `docs/ane-issues-47-49.md`（issue #47/#49 对混合训练的适用性与本机 panic 结论）
- `session-ses_02dc.md`（上一轮 OpenCode 会话记录，含大量调试上下文）
