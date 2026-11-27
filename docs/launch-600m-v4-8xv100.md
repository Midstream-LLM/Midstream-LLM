# 600M v4 + MTP：8×V100 启动说明（2026-11-27）

按顺序做完 0–6 步。每步都有“核对”，不符就停下，不要硬往下跑。

| 项 | 值 |
|---|---|
| 分支 | `feat/doc-attention-8xv100`，至少到 `bcf335e` |
| 配置 | `configs/jishui-600m-v4-docmask-8xv100.json` |
| 模型 | Jishui-600M-Base，主模型 700,665,600 参数 + MTP 25,238,016（D=1，λ=0.3） |
| 数据 | `dataset/corpus-date-open-v4`（中流 v4 原文 + 繁体书简体副本，繁简都练，不转简） |
| tokenizer | `tokenizer/v4-bbpe-32k`（随仓库提交，vocab 32,768） |
| run 目录 | `runs/jishui-600m-v4-docmask-muon-mtp1-8xv100` |
| 预算 | 14,028,477,524 tokens（4 epoch），214,058 次更新，每次 65,536 tokens |

以下命令都在服务器上击水仓库根目录执行。

## 0. 前置检查

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
df -h .
zstd --version
.venv-torch/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

核对：
- 8 张 V100 32G，都空闲；显卡被别的任务占着就先不要开。
- 磁盘至少留 40 GB：压缩包 5.4 GB + 解压后 9.5 GB + checkpoint（按 fp32 权重和 Muon/Adam 状态估算，每个完整 checkpoint 约 6 GB，保留 3 个约 18 GB，以实际为准）。数据解压完可以删掉压缩包。
- 没有 `zstd` 就先装上（或请管理员装），下面解压要用。

## 1. 更新代码

```bash
git fetch origin
git checkout feat/doc-attention-8xv100
git pull
git log --oneline -1
```

核对：最新提交是 `bcf335e Enable DeepSeek-V3-style MTP for the 600M v4 run`，或更新。

tokenizer 指纹要和配置一致：

```bash
sha256sum tokenizer/v4-bbpe-32k/tokenizer.json
```

应为 `1dfcf66ab75d71aaa3ec35d4941fc9449a89fd2350feee0645af0610a42a26dc`。

## 2. 下载数据

数据在阿里云北京区 `oss://fgprc/dataset/jishui-600m-v4-20261127/`，公共读，直接下载，不需要密钥。

```bash
mkdir -p dataset/downloads && cd dataset/downloads
B=https://fgprc.oss-cn-beijing.aliyuncs.com/dataset/jishui-600m-v4-20261127
wget -c "$B/corpus-date-open-v4.tar.zst" "$B/corpus-date-open-v4.tar.zst.sha256"
sha256sum -c corpus-date-open-v4.tar.zst.sha256
cd ../..
```

核对：
- `corpus-date-open-v4.tar.zst` 为 5,404,477,322 字节。
- `sha256sum -c` 输出 `OK`（sha256 `a7f729b28c5437d7d8a750dc3dca2a53b653a60472d11ba73d96b8915ffbbd84`）。
- 下载中断时重跑同一条 `wget -c` 会续传。

## 3. 解压并校验数据

```bash
test ! -e dataset/corpus-date-open-v4 || { echo "dataset/corpus-date-open-v4 已存在，先确认再处理"; exit 1; }
zstd -dc dataset/downloads/corpus-date-open-v4.tar.zst | tar -xf - -C dataset/
du -sh dataset/corpus-date-open-v4
```

然后用配置里的条件做一次完整校验（逐个文件核对 sha256、年代条件、tokenizer 指纹、不转简），大约几分钟：

```bash
PYTHONPATH=src .venv-torch/bin/python - <<'EOF'
import json
from jishui.prepare_corpus import validate_training_data
c = json.load(open("configs/jishui-600m-v4-docmask-8xv100.json"))
d = c["data"]
m = validate_training_data("dataset/corpus-date-open-v4", before_year=d["before_year"],
    vocab_size=c["model"]["vocab_size"], tokenizer_sha256=d["tokenizer_sha256"],
    date_policy=d["date_policy"], script_conversion=d["script_conversion"])
print("ok", m["by_split"])
EOF
```

核对：输出 `ok`，且
- train：6,166,111 行，3,500,953,270 tokens
- val：12,102 行，4,590,962 tokens
- test：11,096 行，4,660,025 tokens

通过后可以删掉 `dataset/downloads/corpus-date-open-v4.tar.zst`。

## 4. 跑测试（CPU）

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv-torch/bin/python -m unittest \
  tests.test_mtp tests.test_muon tests.test_jishui_600m_arch \
  tests.test_document_attention tests.test_corpus_pretraining tests.test_block_data
```

核对：`OK`（Mac 上为 60 项，跳过 2 项）。

## 5. smoke（20 步）

```bash
export CONFIG=configs/jishui-600m-v4-docmask-8xv100.json
export DATA_DIR=dataset/corpus-date-open-v4
export RUN_DIR=runs/jishui-600m-v4-docmask-muon-mtp1-8xv100
bash scripts/train_8xv100_docmask.sh smoke 2>&1 | tee runs/smoke-600m-v4-mtp1.log
```

smoke 写到带时间戳的临时目录 `${RUN_DIR}-smoke-…`，不会占用正式 run 目录，跑完可删。

核对（从日志里找）：
1. 8 个 rank 都起来了；`world_size=8 attention=document optimizer=muon mtp_depth=1`。
2. 参数量 `parameters=725903616`（主模型加 MTP）。
3. `usable_train_tokens=` 后面是按分类拆开的训练 token 数，各项加起来应为 3,500,953,270；启动行里应有 `final_step=214058 target_tokens=14028477524`。**不一致就停下**（第 3 步校验已核对过行数 6,166,111），把 `target_tokens` 改成 4 ×（训练 tokens + 训练行数），`max_steps` 改成它除以 65,536 向上取整，再重跑 smoke。开训后再改预算会被 resume 拒绝。
4. 初始验证 loss 是有限值；记下打印的验证 protocol 指纹。它和 v3 不同（词表和验证来源都变了），loss 不能和 v3 直接比。
5. `train/loss`（纯 NTP）和 `train/mtp_loss` 都有，且是有限值；`grad_norm` 没有 inf/nan 连发。
6. 记下 tok/s 和 `peak_vram_gib`。MTP 估计慢 5–8%、每卡多约 1 GiB。
7. 最后打印 `smoke run complete`，checkpoint 写出成功。

如果 OOM：把配置里 `training.batch_size` 改成 2、`grad_accum_steps` 改成 2（全局 batch 不变），同时把 `eval_batches` 从 32 改成 64（保持 128 个验证块），重跑 smoke。

## 6. 正式开训

正式 run 要跑很多天，放在 tmux 里（或用 nohup），断开 SSH 不影响：

```bash
tmux new -s jishui-600m-v4
export CONFIG=configs/jishui-600m-v4-docmask-8xv100.json
export DATA_DIR=dataset/corpus-date-open-v4
export RUN_DIR=runs/jishui-600m-v4-docmask-muon-mtp1-8xv100
test ! -e "$RUN_DIR" || echo "RUN_DIR 已存在：要续训用 resume，不要 start"
bash scripts/train_8xv100_docmask.sh start 2>&1 | tee -a runs/jishui-600m-v4-docmask-muon-mtp1-8xv100.log
```

按 `Ctrl-b d` 离开 tmux，`tmux attach -t jishui-600m-v4` 回来。

中断后（掉电、OOM、手动停）用同一组环境变量严格恢复，不要再用 start：

```bash
bash scripts/train_8xv100_docmask.sh resume 2>&1 | tee -a runs/jishui-600m-v4-docmask-muon-mtp1-8xv100.log
```

resume 会核对配置、数据和 tokenizer，与开训时不同就拒绝。

## 开训后盯什么

- **前几千步**：这是第一次在 600M 上同时用 Muon 和 MTP。盯紧 `train/loss`、`train/mtp_loss`、`grad_norm` 和 loss scale。loss 发散或 scale 一路降到很小，就停下，把 `learning_rate` 从 4e-4 降到 3e-4，换新的 RUN_DIR 从零重开。
- `train/mtp_loss` 应高于 `train/loss`（预测更远的 token 更难），两者同步下降。
- 验证只评主头的 NTP loss，可以和以后不开 MTP 的 v4 run 直接比。
- tracking 是 W&B offline 模式，数据在 run 目录里。

## 训练完之后

给 SFT 或推理用之前，用 `checkpoints_torch.strip_mtp` 去掉 `mtp.*` 权重，得到与 Jishui-600M-Base 结构完全相同的主模型（见 `docs/mtp-3060.md`）。
