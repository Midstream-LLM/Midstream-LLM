# 使用中流语料库重新预训练 200M

2026-11-17：训练与语料管理均使用 corpus 服务器。新数据准备、动态分类和独立
PyTorch 训练入口已实现。服务器执行进展见下方 2026-11-18 记录。

2026-11-18（UTC+8）浏览器执行记录：已保存配方 `jishui-pre1945-v2`，
提交 corpus 数据集发布任务 **#25**。试算候选 1,579,859 篇，去重后
954,533 篇 / 480,999,375 字；train 935,760 篇 / 472,134,835 字，
val 9,672 篇 / 4,448,538 字，test 9,101 篇 / 4,416,002 字。
任务 #25 已完成，实际 **release_id=2**；不要把 job_id 当作 release_id。

OSS 已通过 Chrome 登录，目标为北京 `fgprc` 桶的 `dataset/` 目录，
其中现有 `processed.7z` 是 2026-08-15 的旧产物，不覆盖。
2026-11-18 已通过 `corpusserverproxy` 恢复 SSH。使用 CLI 提交完整文档导出任务
**#26**，已完成，实际 **export_id=1**，包含 `composed_years` 和完整模板。
服务器目录为
`/media/sidi762/b41354bc-9442-475d-9c51-ad8d110f2139/midstream/exports/jishui-pre1945-v2/1`。
导出为 9 个 JSONL 分片及 manifest；压缩包命名
`jishui-pre1945-release2-export1-20261118.tar.gz`，另有同名 `.sha256` 文件。
两者已通过 Chrome 原生文件选择窗口上传至北京区 `oss://fgprc/dataset/`；
用户明确要求保留公共读，文件继承 Bucket 权限。压缩包为 726,593,476 bytes，
SHA-256 为 `bc64f64bfadab9f3626b624c8ac44a20fc5ae4a8dcddee2ced322f15081da311`。
从训练服务器访问 OSS 返回 200，文件大小和校验清单一致。
本机交付文件及截图在 `dataset/deliveries/jishui-pre1945-release2-20261118/`。

完整导出审计：954,533 篇、26 类（含未分类）、带模板 496,130,552 字符，
成书年代上界最大 1944，越界或未知为 0；954,300 条非空标题前缀全部核验通过。
3,581 篇超过 8,192 字符，最长 14,422,084 字符，完整文档导出未按 `max_chars` 截断。
模板渲染会 strip 标题等元数据字段的首尾空白，不 strip 正文。

本次代码部署在服务器击水根目录下 `deployments/corpus-pre1945-20261118/`，
使用原 `.venv-torch`，已在服务器通过 28 项测试。修正了原依赖文件中不存在的
`opencc-python-reimplemented>=2.4`，实际安装版本为 `0.1.7`。
用户已授权暂停占用显存的 Qwen 服务；启动参数保存在部署目录 `qwen-paused.json`。
训练数据生成至独立的 `dataset/corpus-pre1945-v2/`；旧实验与 checkpoint 保留。

训练分片已完成：954,533 篇 / 436,023,259 tokens；train 935,760 篇 /
427,815,186 tokens，val 9,672 篇 / 4,084,182 tokens，test 9,101 篇 /
4,123,891 tokens。年代复检排除数为 0，发布筛选已完成全部排除。
2026-11-18 00:34（UTC+8）真实 CUDA 短测完成 20 步 / 327,680 tokens，
完整 checkpoint 为 `runs/jishui-200m-corpus-pre1945/checkpoints/step_00000020/`。
初始验证 loss=10.5528；step 10/20 的训练 loss 为 10.4621/9.6174，
短测约 7,715 tokens/s，峰值分配显存 6.31 GiB，无梯度溢出。
本轮从 seed=42 随机初始化，200,224,640 参数；沿用配置的 4B token 预算。
正式训练从这份 step 20 完整状态恢复，启动脚本为部署目录的 `train-launch.sh`。
W&B 在线初始化返回 403，在更新前退出；随后使用 `--wandb-mode offline` 恢复，
指标保存在 run 目录的 `wandb/`，标准输出 `train-offline.stdout.log`，
训练日志为 run 目录的 `trainlog.txt`。在线网络/权限恢复后可再同步离线指标。

独立在线看板已接入 <https://www.fgprc.org.cn/training>，实验 ID 固定为
`jishui-200m-corpus-pre1945`，开始时间为 `2026-11-18T00:32:41.068+08:00`。
服务器用户服务 `jishui-corpus-pre1945-dashboard.service` 已启用，每 30 秒读取
本次 run 日志并推送公开指标；复用训练服务器已有的看板凭据，不与 W&B 互相依赖。
公开 API 已验证 `live` 状态与步数持续更新。监控服务可用
`systemctl --user status jishui-corpus-pre1945-dashboard.service` 查看，详细接入说明见
[在线训练看板](online-dashboard-integration.md)。

代码发布在 GitHub `Midstream-LLM/Midstream-LLM` 的 `codex/corpus-pre1945` 分支，
基于远端 main 单独整理，未包含本机无关的未提交改动及其他尚未推送的历史提交。

## 数据口径

- 本轮“1945 年以前”按 **成书年代上界 < 1945** 执行，最晚 1944 年。
  1945 年及以后、跨越截止年份、年代未知或上界未知的记录不进入训练/验证/测试。
  起年未知但已知上界早于 1945 的记录可进入。成书年代采用导出中的结构化
  `composed_years`（含首尾年份）；不按扫描、抓取或现代再版年份推断正文年代。
- `configs/corpus-pre1945-recipe.json` 使用 quality=ok、段落去重、按来源分层、
  按字符数 98/1/1 划分。`year_to=1944` 的相交筛选再配合排除 `year_from=1945`，
  去掉跨界及无上界记录。配方不要加入绕过筛选的手动 document add。
- 配方冻结原文版本；训练准备阶段对**已经带模板的完整正文**执行 OpenCC t2s，
  包括前缀字形。保留原始语料及发布版本，转换配置与包版本写入训练 manifest。
- 新分类直接按字符串建立动态 ID，null 独立保留为未分类，原字段仍在文档索引中。
  默认每个 epoch 打乱文档后完整消费正文，用 EOD 分隔并打包成 2048 token 序列。
  每个文档的模板前缀每轮出现一次；长文后续片段不重复前缀。

## 1. 固定发布并导出带模板文本

使用服务器现有 midstream 环境。先对上述配方运行 `midstream export-preview`，
查看选材和划分，再用工作台导入配方发布；CLI `midstream export <配方路径>`
也会生成发布和一份 Parquet。后续导出使用它返回的发布编号，不使用示例编号。

```bash
# 在 midstream-core 环境，JISHUI_ROOT 指向服务器现有击水工作区
.venv/bin/midstream export-preview "$JISHUI_ROOT/configs/corpus-pre1945-recipe.json"
```

固定发布后，使用现有 `export-release`。以下模板是可修改的起点，已知字段才输出，
`--template-ratio` 控制套用比例。字段必须包含 source、category 和 composed_years，
不能沿用只有正文的默认字段列表。

```bash
.venv/bin/midstream export-release "$RELEASE_ID" \
  --format jsonl --unit document --shard-mb 256 \
  --fields ordinal version_id doc_id split title text authors era category genre source composed_years tags \
  --template '[《{title}》\n][{authors}\n][【{era}·{genre}】\n]{text}' \
  --template-ratio 1 --template-seed 42
```

本训练路径消费这份带模板的 JSONL/Parquet 导出。语料工具现有 `midstream tokenize`
仍编码原正文，不用于本轮；本次修改全部位于击水仓库，无需修改语料工具或其数据库。
关闭 JSONL 外层压缩，保留根 `manifest.json` 与列出的所有分片。

## 2. 制作训练分片

在击水工作区，`EXPORT_DIR` 指向上一步导出的 manifest 所在目录。
分词器沿用 `ccc-bbpe-32k`，不添加 BOS/EOS、不截断、不 padding。
训练时由 sampler 插入 EOD=2。

```bash
PYTHONPATH=src .venv-torch/bin/python -m jishui.prepare_corpus \
  --export-dir "$EXPORT_DIR" \
  --output-dir dataset/corpus-pre1945-v2 \
  --tokenizer tokenizer/ccc-bbpe-32k \
  --before-year 1945
```

准备程序先校验导出文件的 SHA-256，再按固定 split 逐条检查年代、转简、编码。
模板已包含在 text 中，只编码一次。缺少年代字段会报错；未知或越界的记录写入
`reports/excluded.jsonl`，按原因、来源汇总。过滤后 train/val 为空时拒绝发布。
已有输出目录不会被覆盖；失败会清理本次创建的临时目录。

产物包括 `tokens/shard_*.npy`、`manifest/docs_tokens.jsonl`、`manifest.json`、
原导出 manifest 和排除清单。索引保留 version_id、原分类、年代、split 和训练文本哈希；
根 manifest 记录模板配置、发布身份、转换配置、tokenizer 指纹及分片哈希。
训练启动与恢复都会核验文件、词表、tokenizer 和年代条件。

## 3. 新训练与恢复

`configs/jishui-200m-corpus-pre1945.json` 保留旧 200M 模型结构与 32k tokenizer。
示例预算仍为 4B tokens，学习率为从零训练的初始建议：warmup 1000 步到 3e-4，
单周期 cosine 降到 3e-5；训练前根据实际入选 token 数、短测和目标调整。
CUDA 使用 fp32 参数/Adam + fp16 AMP/GradScaler。
每次验证保留总体指标，并评估验证 token 最多的 12 类，避免数百类逐类验证拖慢训练；
`pretraining.eval_categories` 可调整，设为 0 时只计算总体指标。

```bash
# 从零初始化；短测后在同一目录恢复，短测更新会计入总预算。
PYTHONPATH=src .venv-torch/bin/python -m jishui.pretrain_torch \
  --config configs/jishui-200m-corpus-pre1945.json \
  --data-dir dataset/corpus-pre1945-v2 \
  --run-dir runs/jishui-200m-corpus-pre1945 \
  --steps 20 --evaluate-first --wandb-mode disabled

# 完整恢复同一任务，包括 optimizer、sampler、RNG 和学习率相位
PYTHONPATH=src .venv-torch/bin/python -m jishui.pretrain_torch \
  --config configs/jishui-200m-corpus-pre1945.json \
  --data-dir dataset/corpus-pre1945-v2 \
  --run-dir runs/jishui-200m-corpus-pre1945 --resume latest
```

若要从旧模型继续学习新数据，使用**另一个新 run 目录**，加
`--init-from <旧CUDA checkpoint目录>`。仅导入权重，Adam、GradScaler、采样器、
步数和 token 预算从零开始，并记录父 checkpoint；需要按继续预训练调整学习率。
旧 `--resume` 保持严格检查，同一任务恢复时不能改变数据、年代门槛、采样方式或日程。
旧 4B 配置及完整状态迁移路径保持可用。

正式运行前在 corpus 上检查显存、内存、磁盘和 llama-server 等现有进程的资源占用。
新导出和 checkpoint 写入独立目录；已有数据副本可能使用硬链接，不原地覆盖。

## 验证

`tests/test_corpus_pretraining.py` 使用合成导出和 tiny CPU 模型验证完整链路：
模板及繁转简进入 NPY、1944/1945/跨界/未知年份、未知分类与 300 类索引、
正文前缀和 EOD、从零训练、连续训练与恢复后的权重逐项一致、权重初始化重置 Adam，
以及输入文件校验和失败清理。CPU 验证使用 float32；正式 CUDA 使用 float16。

```bash
PYTHONPATH=src .venv-torch/bin/python -m unittest \
  tests.test_corpus_pretraining tests.test_data tests.test_torch_continuation \
  tests.test_tracking tests.test_checkpoints
```
