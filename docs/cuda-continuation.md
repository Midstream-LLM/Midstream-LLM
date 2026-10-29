# 200M CUDA 续训到累计 4B tokens

状态更新（2026-10-29）：日志确认本轮已于 **2026-10-07 20:52（Asia/Shanghai）**完成，
最终为 step **244,141 / 4,000,006,144 tokens**，完整 checkpoint 为
`runs/jishui-200m-cuda-4b/checkpoints/step_00244141/`。下方启动流程保留用于复现；
不要用已完成 checkpoint 重启相同 4B 目标，trainer 会拒绝已达目标的恢复点。

2026-10-02：Linux / RTX 3060 12GB，沿用 Stage 0 模型和 ccc-bbpe-32k。
原始 MLX 恢复点是 `runs/jishui-200m-stage0-1b/checkpoints/step_00055200`，
累计 904,396,800 tokens。原始 checkpoint 保留，CUDA 产物写入独立目录。

## 训练配置

- 入口：`python -m jishui.pretrain_torch`，单 CUDA GPU；多卡入口仍待实现。
- 配置：`configs/jishui-200m-cuda-4b.json`。
- 总目标：4,000,000,000 tokens，包含恢复点以前已经训练的部分。
- seq 2048、microbatch 1、累积 8；每次更新 16,384 tokens。
- 完成于 step 244,141，实际累计 4,000,006,144 tokens，最多多出一个 batch。
- fp16 autocast + GradScaler；参数主副本、梯度、Adam 动量用 fp32。
- 3060 已测无梯度重算峰值分配约 6.31 GiB，稳定短测约 7,300–7,400 tok/s；
  该显存值不含全部 CUDA 预留空间和桌面显存。持续速度、保存与验证开销会影响总耗时。
- 每 200 步验证整体和六类 loss；每 400 步保存，保留最近 3 份完整 checkpoint。
- 新 run 开始训练前保存导入状态；SIGTERM / Ctrl-C 会在当前更新完成后保存。
- checkpoint 包含模型、Adam、GradScaler、sampler、RNG、训练步数、累计 tokens、
  学习率日程、数据索引信息与采样配比。恢复时检查架构、关键训练参数和日程一致性。

## 学习率：余弦退火加重启

用户选择 cosine annealing with restarts。采用 SGDR 的余弦公式，按绝对 optimizer
update 计算，避免恢复时重置周期。三个周期大致等长，末周期在总目标处结束。

| 位置 | 累计 tokens | 行为 |
|---|---:|---|
| step 55,200 恢复 | 0.904397B | 从已保存的 `3.6086549e-5` 接续 |
| 接下来 1,000 次更新 | 到 0.920781B | 线性升至 `1e-4`，只在首周期做一次 |
| 首周期末：step 118,181 | 1.936278B | 余弦降至 `1e-5` |
| 下一次更新 | — | 学习率重启至 `1e-4`，保留 Adam 动量 |
| 第二周期末：step 181,161 | 2.968142B | 再次降至 `1e-5` |
| 下一次更新 | — | 再次重启至 `1e-4` |
| 第三周期末：step 244,141 | 4.000006B | 降至 `1e-5` 并完成训练 |

每个周期的首/末次更新分别落在峰值/谷值；首周期先做上述 warmup。
后续两次重启直接回到峰值，不重置模型、Adam 或 GradScaler。
初次迁移后，数值精度从 MLX 的 fp16 参数/动量提升为 fp32 主副本，
因此是保留训练状态的跨框架续训，不保证此后的更新逐位等同于旧 MLX。

公式参考：[PyTorch CosineAnnealingWarmRestarts](https://docs.pytorch.org/docs/2.6/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)。
代码在 `src/jishui/schedules_torch.py`；测试对照 PyTorch 的 SGDR 公式，覆盖周期边界、末步和恢复相位。

## 扩训采样

从恢复点起改为按六类可用训练 token 数比例采样，继续使用原 mmap sampler 和恢复的 RNG 状态。
这样降低此前 5/6 类的过采样。配比冻结在 CUDA checkpoint 中：

| 类别 | 比例 |
|---|---:|
| 1 | 75.6081% |
| 2 | 4.3747% |
| 3 | 1.8528% |
| 4 | 14.6868% |
| 5 | 0.6967% |
| 6 | 2.7809% |

这是有放回抽样；累计 4B tokens 不等于语料无重复遍历一次。
本次没有改写冻结的原始数据或 `sampling_weights.json`。

## 环境与启动

`.venv-torch` 是 Python 3.12 的独立 venv。使用现有 Python 解释器创建，
通过 pip 安装 PyTorch 2.6.0+cu124，与当前 550.163.01 驱动配套。
从 Mac 拷贝来的 `.venv` 不适用于 Linux，继续保留。

```bash
# 首次启动从 MLX checkpoint 迁移；以后自动从本 run 的 latest 续训。
bash scripts/start_cuda_4b.sh

# 查看日志
tail -f runs/jishui-200m-cuda-4b/trainlog.txt

# 温和停止：当前更新完成后保存完整 checkpoint。
systemctl --user stop jishui-200m-cuda-4b.service
```

启动脚本支持一个可选参数：已验证的 CUDA checkpoint 路径，作为新 run 的起点。
首次保存后，正式 run 独立恢复，不依赖短测目录。`trainer.lock` 防止同一 run 重复启动。
后台训练由 systemd 用户级临时服务托管，命令结束后继续运行；不设置自动重启或开机启动。
服务停止时最多等待 300 秒，让当前更新、checkpoint 和 W&B 同步完成。
GPU 进程需要能访问主机的 NVIDIA 设备；受限沙箱内的 `nvidia-smi` 失败不代表主机驱动损坏。

## 已完成的验证

- 真实 MLX checkpoint 导入到 CUDA 后，固定验证 loss 为 **4.466994**，旧日志为 **4.4670**；
  六类验证 perplexity 也与旧日志一致到显示精度附近。
- 所有参数及 Adam 一、二阶动量逐项验证名称、形状和有限性，Adam step 恢复为 55,200。
- 验证 causal mask、RoPE 半维配对、梯度重算一致性、Adam 更新公式和完整保存/恢复。
- CUDA 短训、保存、恢复已跑通；恢复不会清空 Adam 或 sampler。
- 调度测试覆盖 warmup、两次重启、最后一步、序列化后的相位一致性。
- 长期收敛效果仍需观察每类验证 loss；短测不证明 `1e-4` 峰值最优。

```bash
PYTHONPATH=src .venv-torch/bin/python -m unittest \
  tests.test_torch_continuation tests.test_tracking tests.test_data tests.test_checkpoints tests.test_distributed
```

## 推理（CUDA checkpoint）

`generate_torch` 是 `generate_mlx` 的 PyTorch 对应物，采样函数直接复用 `generate_mlx.sample_next_token`，
因此固定 prompt/seed/采样参数后，两条路径的输出可比。它只读 `training.pt` 里的模型权重，
不加载 optimizer，也不写任何训练状态；模型仍是纯 pretraining 能力，没有 chat template，
提示词应写自然古文前缀而不是问答指令。默认屏蔽未参与预训练的 pad/unk/chat special tokens，
保留 `<|endoftext|>` 作为停止符。

```bash
PYTHONPATH=src .venv-torch/bin/python -m jishui.generate_torch \
  --run-dir runs/jishui-200m-cuda-4b \
  --checkpoint latest \
  --prompt '太史公曰：' \
  --max-new-tokens 128 \
  --temperature 0.8 --top-p 0.9 --top-k 40 --repetition-penalty 1.05
```

- 权重默认 CUDA 上 fp16、CPU 上 fp32；`--device cpu` 可避开正在训练的 GPU（本机 3060 训练时约占 7.8/12 GiB）。
- `--temperature 0` 为 greedy；`--top-logits N` 打印 prompt 后继 top-N token 及其 logit，便于对比 checkpoint。
- `--prompt-file` 跑提示词套件，`--output-json` 落盘每条结果，便于跨 checkpoint 比较。
- 没有 KV cache：每生成一个 token 重算整段 context，速度随上下文变长而下降。

2026-10-03 实测（step 87,600 / 1.435B tokens，3060）：greedy `太史公曰：` → `太史公曰：今夫世，而世`，
CUDA fp16 与 CPU fp32 逐 token 一致，top-3 logit 差 ≤0.002。

### 提示词套件

`tests/prompts/classical-inference-suite.json` 是 22 例古汉语续写探针：**9 对简繁平行**（奏议／诸子／诗／佛经／
医书／奏疏／词／尺牍／易类，两版等义同字数）＋ 空 prompt 冷启动、简繁混排、以及一对约 50 字长前缀。
平行对是刻意的：否则无法区分「繁体本身难」与「模型退化」。

```bash
PYTHONPATH=src .venv-torch/bin/python -m jishui.generate_torch \
  --run-dir runs/jishui-200m-cuda-4b --checkpoint latest \
  --prompt-file tests/prompts/classical-inference-suite.json \
  --max-new-tokens 24 --output-json runs/inference-suite/step_00088000-t24.json
```

判据分两层（详见套件文件的 `criterion`）：

- **flags（失败，退出码非零）**：文字内部出现 U+FFFD、泄漏未屏蔽 special token、一个 token 都没生成。
- **notes（提示，需人工看）**：`trailing-partial-char`（`stop=length` 时末字被切一半，byte-level BPE 的截断伪影）、
  `immediate-eod`（模型判定 prompt 已是完整文档）、`low-distinctness`（生成段 ≥40 字且 4 字去重率 <0.7，启发式阈值）。

开跑前还会强制校验每条 prompt 能 tokenizer 往返；套件同时打印简繁 prompt 的 token 成本比。

2026-10-03 实跑（step 88,000，temp 0.8 / seed 42 / 24 token）：22 例，**flags 0、notes 5**，总生成 634 token、约 60 tok/s。
三个值得注意的结果：

- 简繁 token 成本比 **1.00–1.64（10 对平均 1.35）**，因为 ccc-bbpe-32k 训练语料 CCC 以简体为主；繁体 `聞/賢/樂/為` 要拆成
  两个 byte 片段，而 `國/黃/風/東/陰` 本身在词表里。这个差异是语料属性，不是模型对繁体的额外损失。
- `p03-simp`（`国破山河在，城春草木深。`）与 `p07-trad`（`大江東去，浪淘盡，千古風流人物。`）都是
  `generated=1 stop=eod`——模型判定名句已收尾即文档结束；而等价的另一半却继续写了 24 token，说明 EOD 概率卡在决策边界。
- `x01-empty` 空 prompt 冷启动立刻陷入 `元丰县学` 复读（4 字去重率 0.59，全批唯一一条低去重）；`p09-simp` 也出现
  `而感通天下之至` 复读。这是 1.44B tokens 阶段的正常表现，但说明「无条件下自由生成」比「给古文前缀续写」明显更差。

## W&B 实验追踪

使用 W&B Models 的训练实验追踪。配置文件中的 `tracking` 指定：

- entity：`liangsidi-university-of-electronic-science-and-technolog`
- project：`Jishui`
- name：`jishui-200m-cuda-4b-cosine-restarts`
- mode：`online`

W&B SDK 安装在 `.venv-torch` 中。首次在本机终端运行 `.venv-torch/bin/wandb login`；
API key 由 SDK 的本机凭据存储管理，不放进项目配置、checkpoint 或聊天。

记录内容：

- `train/loss`、`train/perplexity`；
- `val/loss`、`val/perplexity` 和 `val/c1_loss` 至 `val/c6_loss`，各类也有 perplexity；
- `optimizer/learning_rate`、梯度范数、loss scale、重启周期和溢出重试次数；
- `perf/*`：即时/平均 tok/s、已分配/预留显存、剩余时间；SDK 同时采集 GPU 系统指标；
- `progress/*`：累计 optimizer step、tokens 和目标完成比例；
- 模型、训练、采样、学习率日程、运行环境和 MLX checkpoint 来源；
- checkpoint 步数与本地路径，以及实验完成/停止/失败状态。

训练和验证曲线默认以累计 tokens 为横轴，优化器曲线以 optimizer step 为横轴。
W&B run ID 保存在 run 目录的 `wandb_run.json`；同一目录恢复时使用相同 ID。
如果进程异常退出后从较早 checkpoint 恢复，重放的训练点会带有新的 `resume/attempt`，
SDK 的内部 history step 继续递增，不会因训练步数回退而静默丢弃指标。

checkpoint 中也保存 W&B 关联信息。复制 checkpoint 到一个新 run 目录时创建新的实验，
记录父实验引用，避免改写原实验。默认只记录指标和配置，不上传训练文本或大型模型 checkpoint。

在线模式初始化失败会在训练开始前报错。需要离线测试时，在训练命令后加 `--wandb-mode offline`；
完全关闭用 `--wandb-mode disabled`。离线模式每次启动生成独立 session（同一 group），
因为 W&B 不支持 offline resume；之后按 SDK 提示运行 `wandb sync <offline-run-dir>`。

当前已建立的在线实验：
[jishui-200m-cuda-4b-cosine-restarts](https://wandb.ai/liangsidi-university-of-electronic-science-and-technolog/Jishui/runs/96a54509595f)。
运行是否仍在进行，以进程及日志为准。
