# Jishui 数据生产报告

生成时间: 2026-08-07 | 阶段: inventory→clean→dedup→split→tokenize

## 总览
- 原始文档/文件数: 4,373,240（各源 {'daizhige': 15694, 'ccc_corpus': 12009, 'ccc_punctuate': 46546, 'ccc_translate': 1924378, 'poetry': 2246, 'mingqing': 20868, 'erya': 882368, 'greathangpt': 1202968, 'wanli': 262, 'wikisource': 265653, 'gutenberg': 248}）
- 解析成功: 4,373,240 / 4,373,240 | 排除: 2,501,084
- 清洗后保留: 1,872,156 | 去重后保留: 1,872,156 | 已 tokenize: 1,836,149
- 最终字符数: 4,872,871,188 | BBPE32k tokens: 4,155,137,755

## 1–7 字符流水统计（按源）
| 源 | 原始 | 解析 | 清洗前字符 | 清洗后字符 | 精确去重后 | 近重去重后 |
|---|---:|---:|---:|---:|---:|---:|
| ccc_corpus | 12,009 | 12,009 | 17,229,693 | 17,078,073 | 17,078,073 | 17,050,994 |
| ccc_punctuate | 46,546 | 46,546 | 9,787,842 | 9,780,162 | 9,754,343 | 7,158,616 |
| wanli | 262 | 262 | 1,518,585 | 1,518,585 | 1,518,585 | 1,515,890 |
| greathangpt | 1,202,968 | 1,202,968 | 354,516,409 | 343,204,956 | 335,760,999 | 326,046,185 |
| mingqing | 20,868 | 20,868 | 77,545,672 | 78,204,310 | 77,515,337 | 73,727,122 |
| poetry | 2,246 | 2,246 | 251,457,166 | 241,673,097 | 241,673,097 | 211,509,951 |
| erya | 882,368 | 882,368 | 1,938,230,721 | 1,932,776,058 | 1,932,656,170 | 1,630,189,287 |
| daizhige | 15,694 | 15,694 | 1,743,336,364 | 1,725,752,982 | 1,725,594,872 | 1,395,781,885 |
| wikisource | 265,653 | 265,653 | 1,710,903,802 | 1,518,105,925 | 1,513,472,714 | 1,209,891,258 |
| gutenberg | 248 | 248 | 41,079,872 | 0 | 0 | 0 |

## 8–10 Token 统计（ccc-bbpe-32k 实测编码）
| 源 | tokens |
|---|---:|
| ccc_corpus | 11,843,378 |
| ccc_punctuate | 4,999,652 |
| wanli | 1,598,492 |
| greathangpt | 351,539,994 |
| mingqing | 59,395,716 |
| poetry | 15,919,868 |
| erya | 1,346,662,849 |
| daizhige | 1,124,702,807 |
| wikisource | 1,238,474,999 |
| gutenberg | 0 |

按类别：
| 类 | 名称 | tokens | 占比 | 目标占比 | 缺口 vs 6B | 权重 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 经史子集及散文 | 2,120,550,445 | 51.0% | 35% | +20,550,445 | 0.6858 |
| 2 | 小说笔记戏曲 | 153,441,349 | 3.7% | 15% | -746,558,651 | 4.0619 |
| 3 | 诗词曲赋 | 61,402,457 | 1.5% | 8% | -418,597,543 | 5.4136 |
| 4 | 佛道医 | 572,943,175 | 13.8% | 12% | -147,056,825 | 0.8703 |
| 5 | 晚清民国文学 | 2,106,787 | 0.1% | 15% | -897,893,213 | 100.0 |
| 6 | 知识与公共文本 | 1,598,492 | 0.0% | 15% | -898,401,508 | 100.0 |

## 11–12 重复矩阵（被删字符，按 源→重复源）
```
ccc_corpus->ccc_corpus: 27,079
ccc_punctuate->ccc_corpus: 2,591,274
ccc_punctuate->ccc_punctuate: 4,453
daizhige->ccc_corpus: 10,046,742
daizhige->ccc_punctuate: 1,594
daizhige->daizhige: 39,270,895
daizhige->erya: 193,734,920
daizhige->greathangpt: 10,448,910
daizhige->mingqing: 71,984,551
daizhige->poetry: 4,214,520
daizhige->wanli: 110,855
erya->ccc_corpus: 949,081
erya->ccc_punctuate: 217,668
erya->erya: 274,900,064
erya->greathangpt: 604,631
erya->mingqing: 25,627,411
erya->poetry: 5,405
erya->wanli: 162,623
greathangpt->ccc_corpus: 69,897
greathangpt->ccc_punctuate: 2
greathangpt->greathangpt: 9,644,915
mingqing->ccc_corpus: 6,033
mingqing->ccc_punctuate: 173
mingqing->greathangpt: 164,727
mingqing->mingqing: 3,617,281
mingqing->wanli: 1
poetry->ccc_corpus: 72,513
poetry->ccc_punctuate: 4
poetry->greathangpt: 18,742,545
poetry->mingqing: 1,502
poetry->poetry: 11,346,582
wanli->wanli: 2,695
wikisource->ccc_corpus: 2,445,506
wikisource->ccc_punctuate: 3,580
wikisource->daizhige: 150,727,577
wikisource->erya: 95,727,493
wikisource->greathangpt: 3,498,002
wikisource->mingqing: 11,855,902
wikisource->poetry: 976,468
wikisource->wanli: 1,375
wikisource->wikisource: 38,345,553
```
## 13 1945 时间门控（按 源/状态）
```
ccc_corpus/pass: 12,009
ccc_punctuate/pass: 46,546
ccc_translate/reject: 1,924,378
daizhige/pass: 15,694
erya/pass: 882,368
greathangpt/pass: 1,130,119
greathangpt/quarantine: 72,849
gutenberg/pass: 248
mingqing/pass: 20,868
poetry/pass: 2,246
wanli/pass: 262
wikisource/pending_text: 265,653
```
## 14 许可证
```
cc-by-4.0: 262
pd-or-ccbysa: 265653
pd-research: 4107077
pd-us: 248
```
## 15 质量
```
cjk_ratio_min: 0.0029
docs_with_fffd: 0
avg_doc_chars_final: 2603
dropped_parse_errors: 0
```
## 去重明细
```
l1_file_exact: 8
l2_doc_exact: 1947
l3_para_exact: 41281261
l3n_para_near: 80
l4_doc_near: 0
```
## 划分
```
train: 1867362 docs / 4,124,094,271 tokens
val: 3671 docs / 22,500,907 tokens
test: 1123 docs / 8,542,577 tokens
```
## 已知问题与人工审核事项

1. wikisource 年代门控基于标题与头尾年份标记启发式，`待审核`与误判需人工抽查。
2. ccc_corpus/punctuate 为现代点校本（现代标点），按规则标记为整理本保留正文。
3. erya monolingual 按 100 行切块，块边界不保证篇章完整。
4. 大汉书 未分类 语料全部进入待审核（时间与类别未知）。
5. L4 只检查了采样窗口命中>=8 的前 12000 对文档，覆盖面为采样近似。
6. 类别 5/6 数据源有限（万历邸钞 262 个月档、wikisource 少量），缺口大。
7. Gutenberg 无保留文档（时间门控/来源问题，待核查）。
8. 现代标点版 CCC punctuate 的 output 字段含跨句粘连（首字符为句号）等小瑕疵。
