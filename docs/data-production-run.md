# 数据生产 Pipeline 运行说明（2026-08-06 全量运行）

复现命令：`bash scripts/pipeline/run_all.sh`（各阶段独立、幂等、可恢复；全部日志在 `dataset/reports/pipeline/logs/`）。
断点续跑（dedup 以后的全部阶段）：`bash scripts/pipeline/resume_dedup.sh`。

## 阶段与产物

| 阶段 | 脚本 | 输入 | 输出 |
|---|---|---|---|
| 01 获取 | `01_acquire_gutenberg.py` | 网络（Gutenberg） | `dataset/raw/gutenberg-zh/`（444 本中文书元数据 + 248 本下载） |
| 02 inventory | `02_inventory.py` | 全部 11 个 raw 数据源 | `dataset/interim/manifest/docs.jsonl`（437 万文档，62.6 亿原始字符，0 解析错误） |
| 03 clean | `03_clean.py` | docs.jsonl + raw | `dataset/interim/clean/{source}.jsonl`（保留 187.2 万文档）、`docs_clean.jsonl`、`dataset/quarantine/rejected.jsonl` |
| 05 dedup | `05_dedup.py` | clean 产物 | `interim/final/{source}.jsonl`、`interim/dedup/l{1,2,3,3n,4}*.jsonl`（审计记录）、`docs_final.jsonl` |
| 06 split | `06_split.py` | final 产物 | `dataset/processed/split/{train,val,test}.jsonl` + `split_manifest.json` |
| 07 tokenize | `07_tokenize_stats.py` | final 产物 | `dataset/processed/tokens/shard_*.npy`（uint16）+ `manifest/docs_tokens.jsonl` + `shard_info.jsonl` |
| 08 report | `08_report.py` | 全部上述 | `dataset/reports/pipeline/token_stats.json`、`report.md`、采样权重 |

## 目录约定

- `dataset/raw/`：只读，不改写。
- `dataset/interim/`：中间产物（manifest / clean / final / dedup / token_tmp），可重建。
- `dataset/quarantine/`：被排除文档清单（时间/质量/年代审核），一条不删不留痕。
- `dataset/processed/`：训练直接可用的 tokenized 数据 + 划分 + 权重。
- `dataset/reports/pipeline/`：统计、报告、日志。

## 数据源清单（2026-08 新增 3 个）

1. **wanli-dibao-corpus**（HF `dibao-research/wanli-dibao-corpus`，cc-by-4.0）：万历邸钞/实录，按年月聚合为 262 个文档，类别⑥，1573–1620 全过 1945 门控。
2. **wikimedia/wikisource**（HF，20231201.zh 快照，3.1GB/11 parquet/26.5 万页）：渲染后文本；标题+首尾年份启发式做 1945 门控；作者页/模板/分类页剔除。
3. **Project Gutenberg 全量中文**：`/browse/languages/zh` 444 本 → 逐本抓元数据（作者卒年/原出版时间）→ 门控通过 248 本 → 下载 UTF-8 全文。非单一作者页。

## 关键设计

- **时间门控（≤1945-12-31）**：古代典籍按成书年代；报刊/小说按首发；译本按中文译本首发；整理本的现代标点/导读/译文标记或删除（CCC punctuate/corpus 标为"现代点校本"，CCC translate 全部拒绝）；无法确定 → quarantine。
- **去重 5 级 + 全量审计**：L1 文件级精确（原始字节 sha1）→ L2 文档级精确（清洗文本 sha1）→ L3 段落级精确（t2s 归一 sha1）→ L3n 段内近重复（24 字窗口覆盖 ≥0.9）→ L4 文档级近重复（采样窗口索引 + Jaccard ≥0.8）→ L5 划分后跨集交叉去重。每条删除都写 `dup_source / reason / similarity / chars`。
- **去重优先级**：ccc_corpus > ccc_punctuate > wanli > greathangpt > mingqing > poetry > erya > daizhige > wikisource > gutenberg（保留高质量点校本，剔重版本）。
- **划分**：作品级（非句子）；作者不相交；val/test 覆盖六类；seed=42；划分后再 L5 交叉去重。
- **统计**：全部最终 token 数由冻结的 **ccc-bbpe-32k** 实测编码得出；字符数仅作辅助指标。
- **稳健性**：单源失败记日志继续；阶段输出原子写；字节守卫（输出 ≤4× 源字符）防膨胀；不静默忽略异常。

## 已知问题 / 人工审核事项

见 `dataset/reports/pipeline/report.md` 末尾清单（wikisource 启发式门控、点校本现代标点、erya 切块边界、大汉书未分类待审、Gutenberg 译者卒年误判风险、L4 仅验证前 12000 候选对、类别⑤⑥缺口大等）。

## 2026-08-07 全量运行结果（实测）

| 阶段 | 结果 |
|---|---|
| inventory | 4,373,240 文档，62.6 亿原始字符，0 解析错误 |
| clean | 保留 1,872,156 文档；排除 2,501,084（ccc_translate 现代译文 192 万 + 质量/年代过滤） |
| dedup | L1 8 / L2 1,947 / L3 段落 41,281,261 条 / L3n 80 段（66 篇）/ L4 0 篇；最终 1,872,156 文档，48.73 亿字符 |
| split | train 1,867,362 / val 3,671 / test 1,123 文档（作品级、作者不相交、六类覆盖、L5 交叉去重 27 篇） |
| tokenize | 21 shard（每 ~200M tokens，uint16 npy），共 **4,155,137,755 tokens**，1,836,149 文档 |
| report | `dataset/reports/pipeline/token_stats.json` + `report.md`；采样权重 `dataset/processed/sampling_weights.json` |

断点/幂等说明：
- L3 按数据源断点（`interim/dedup/l3_done.json` + `l3_progress.json`），中途被杀只会重做最后数千文档；
- L4 完成标志 = `l4_doc_near.jsonl` + `docs_final.jsonl` 存在，重跑自动跳过；
- L3/L4 的 sqlite 索引默认写 `interim/dedup/`，可用环境变量 `JISHUI_PARA_DB` / `JISHUI_WIN_DB` 指到内置盘提速（本机实测内置盘快 2 倍以上）。
