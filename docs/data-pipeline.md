# 数据流水线设计（清洗 / 去重 / 分片 / 配比）

状态：设计稿。现存脚本 `scripts/prepare_dataset.py`（manifest / extra / normalize / reports 子命令）已覆盖"入库存档"阶段；本文档描述从 manifest 到 tokenized 训练产物的完整设计。

## 1. 语料总览（manifest + extra 报告实测）

| 数据集 | 文件数 | 字符量 | 说明 |
|---|---|---|---|
| 殆知阁 v20 | 31,388 txt | 1,807,620,280 | 十藏，简繁混杂；全角空格 U+3000 缩进；正史简体、佛/集藏繁体 |
| CCC | 3 jsonl | 147,406,752 | corpus 12,009 条（点校本）/ translate 1.92M / punctuate 46.5k |
| chinese-poetry | 2,285 | 251,504,488 | 全唐诗/宋词/元曲/蒙学等，JSON 数组 |
| ming-qing-wenji | 12 | 82,211,094 | 单 jsonl，含 dynasty/juans 等元数据 |
| erya | 64 | 2,106,398,227 | 古译今平行 + monolingual；**2.1B 含现代译文** |
| greathangpt | 9 parquet | 550,695,054 | char_count 列求和；字段含 dynasty/era/content_category |
| **合计** | | **≈ 4.84B 字符** | 按 chars/token ≈1.3–1.45 → **≈ 3.4–3.7B tokens**（去重前） |

## 2. 清洗规则（各数据集）

- **殆知阁**：UTF-8（部分带 BOM，需去 BOM）；**全角空格 U+3000 是段首/缩进语义，禁止按 ASCII whitespace 折叠**；`\r\n → \n`；不做 NFKC/OpenCC。
- **CCC**：normalize 已完成（`normalized/ccc_*.jsonl`），沿用其规则。
- **chinese-poetry**：JSON 数组 → 展平为记录，保留 author/title 可选拼接（策略待定：正文优先，标题作者作为段落前缀）。
- **ming-qing-wenji / greathangpt**：已结构化，取 text 字段；greathangpt 有 paragraph 边界可直接保留。
- **erya**：split src(古文)/tgt(现代译文)；**训练只取 monolingual 古文 + trans 的 src 侧**，tgt 译文不进入预训练（或降权极少量，见 §5 开放问题）。

## 3. 去重（跨数据集重叠，已有报告）

24 字窗口 t2s 匹配估计（ccc_overlap_daizhige_report.json）：

| CCC 子集 | 与殆知阁重叠 |
|---|---|
| corpus（点校本） | 92.5%（史记/汉书/后汉书/通鉴等 100%） |
| punctuate | 79.8% |
| translate | 10.6% |

策略（去重方向：**保留 CCC 点校本，在殆知阁侧剔除匹配段落**）：
- 理由：CCC corpus 是点校本（质量标注好、已断句），殆知阁为四库本简体转写，文本基本同源。
- 实施：窗口级（24 字，t2s 归一）匹配，在殆知阁文本中删除匹配区段；记录删除比例报告。
- punctuate 与殆知阁 79.8% 重叠，同样窗口级去重（保留 punctuate 版）。
- translate 重叠低（10.6%），不做跨库去重。
- 库内去重（殆知阁内部十藏重复、诗词库内重复）后续补全文档级 simhash 去重。

## 4. 分片与 tokenize（计划，未实现）

- **格式**：每 shard = 一个 tokenized `.npy`（uint32 数组）+ manifest jsonl 记录（shard 号、token 数、来源占比）。训练时直接 memmap 读取，避免 jsonl 重复解析。
- **shard 大小**：固定 ~200M tokens/shard（约 280MB npz），便于多卡并行读取与断点恢复。
- **流程**：manifest → normalize → dedup → shard（按数据集/类别分桶，保证类别连续性）→ tokenize（`ccc-bbpe-32k`）→ token 统计报告。
- **必跑统计**：全量 chars/token（简体 vs 繁体侧分开报）、每数据集 token 量、去重前后总量对比——产出 `dataset/reports/token_stats.json`，作为配比的依据。
- **splitting**：训练/验证按 shard 切分（如留 0.1% 验证），不做文档级交叉，简单可复现。

## 5. 配比（定稿方案）与数据缺口

### 5.1 目标配比（初版训练，tokens 计）

| 数据类别 | 比例 | 累计 tokens |
|---|---:|---:|
| 经史子集及古代散文 | 35% | 2.10B |
| 古典小说、笔记、戏曲 | 15% | 0.90B |
| 诗词曲赋 | 8% | 0.48B |
| 佛藏、道藏、医藏等专门文献 | 12% | 0.72B |
| 晚清与民国公版文学 | 15% | 0.90B |
| 现代中文百科、说明与通识文本 | 15% | 0.90B |
| **合计** | 100% | **6.00B** |

### 5.2 现有语料可支撑量（chars/token ≈ 1.4 估算，去重前）

| 类别 | 现有来源（实测字符量） | ≈ tokens | 目标 | 缺口 |
|---|---|---|---:|---:|---:|
| ① 经史子集及散文 | 史藏 488M + 儒藏 140M + 易藏 35M + 子藏诸子/类书 158M + 集藏别集/总集/文评 232M + 艺藏 19M + CCC corpus 17M + greathangpt 先秦~两宋 80M + ming-qing-wenji 82M + **erya monolingual 1870M** | ~2.2B | 2.10B | ✓ |
| ② 小说、笔记、戏曲 | 集藏小说/演义/话本/笑话/宝卷 119M + 子藏笔记 56M + 诗藏剧曲 17M + greathangpt 明清 31M | ~0.16B | 0.90B | **-0.74B** |
| ③ 诗词曲赋 | 诗藏 100M + chinese-poetry 251M | ~0.25B | 0.48B | **-0.23B** |
| ④ 佛道医专门 | 佛藏 219M + 道藏 46M + 医藏 114M + greathangpt 佛经 258M（与佛藏重叠未知） | ~0.27–0.46B | 0.72B | **-0.26B+** |
| ⑤ 晚清民国公版文学 | **无** | 0 | 0.90B | **-0.90B** |
| ⑥ 现代百科/通识 | **无** | 0 | 0.90B | **-0.90B** |
| 合计 | 现有全库 ≈ 4.94B 字符 | ~3.5B | 6.00B | **-2.5B** |

要点：
- ① 达标依赖 erya monolingual（1.87B 字符，上古训诂类，与四库经部互补）；
- ②③④ 合计缺口 ≈ 1.2B tokens：② 缺明清小说（四大名著等已含于演义 22.8M，但体量远不够）、③ 缺全唐诗/全宋词更大版本、④ 可补完整道藏；
- ⑤⑥ 完全无数据源，合计需 ≈ 2.5B 字符（≈1.8B tokens）新语料；
- 整体缺口 ≈ 2.5B tokens（≈3.5B 字符），全部需要新获取语料。

### 5.3 新增数据源（待定，需用户确认）

- ⑤ 晚清民国公版文学：民国小说/报刊、鲁迅、老舍、沈从文等公版作品集、古登堡/维基文库相关条目；
- ⑥ 现代百科/通识：中文维基百科 dump、知乎精选/教科书类（注意：与"古汉语原生"目标的关系需确认，含现代文本的动机是通用能力兜底）。

### 5.4 当前阶段（Stage 0，200M 本机试验）处理方式

- Stage 0 在本机的训练量预算为 **0.1–1B tokens**：从**现有语料**按 ①②③④ 目标比例取子集（⑤⑥ 缺数据，暂以 ① 余量补齐或按现有量归一化），配比偏差记录在 token_stats.json 中；1B 约需 9.2 天理论时间，完整 4.1B 预算应迁移到 V100/A800；
- ⑤⑥ 数据到位后，Stage 1（600M）再按 6.00B 定稿配比执行。

### 5.5 开放问题

- 繁体语料占比是否需刻意平衡（繁体 chars/token 更低，碎片更多）；
- erya 译文（tgt 侧）是否完全排除（现计划：monolingual 全收，译文对不用）；
- 是否对殆知阁内部重复（十藏与正史/编年重叠）与 greathangpt 佛经/殆知阁佛藏重叠做文档级去重——③④ 缺口可能因去重进一步扩大。

## 6. 落地顺序

1. 实现 tokenize + shard 脚本（tokenizer 已冻结，`ccc-bbpe-32k`）；
2. 跑全量 token 统计 → 产出 `token_stats.json`；
3. 按统计定稿配比 → 冻结训练集 manifest；
4. 之后才写训练脚本。
