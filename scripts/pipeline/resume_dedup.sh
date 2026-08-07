#!/bin/bash
# 断点续跑整条数据生产链路：05_dedup(断点续跑) → 06_split → 07_tokenize → 08_report。
# 用法（在本机终端，保持窗口开启）：bash scripts/pipeline/resume_dedup.sh
set -e
cd "$(dirname "$0")/../.."
export JISHUI_PARA_DB=/private/tmp/jishui_para.sqlite
echo "== [05] dedup (断点续跑) =="
.venv/bin/python scripts/pipeline/05_dedup.py
echo "== [06] split =="
.venv/bin/python scripts/pipeline/06_split.py
echo "== [07] tokenize + shard =="
.venv/bin/python scripts/pipeline/07_tokenize_stats.py all
echo "== [08] report =="
.venv/bin/python scripts/pipeline/08_report.py
echo "== 完成。产物见 dataset/processed/ 与 dataset/reports/pipeline/ =="
