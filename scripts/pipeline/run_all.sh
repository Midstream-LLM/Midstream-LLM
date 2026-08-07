#!/bin/bash
# Jishui 数据生产 pipeline —— 全链路复现命令
# 用法: bash scripts/pipeline/run_all.sh   (幂等可重跑，各阶段独立)
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH=scripts/pipeline

echo "== [01] 获取新数据源 (Gutenberg 全量中文; wanli/wikisource 已在 raw/) =="
[ -f dataset/raw/gutenberg-zh/books.jsonl ] || .venv/bin/python scripts/pipeline/01_acquire_gutenberg.py

echo "== [02] inventory =="
.venv/bin/python scripts/pipeline/02_inventory.py

echo "== [03] clean =="
.venv/bin/python scripts/pipeline/03_clean.py

echo "== [05] dedup =="
.venv/bin/python scripts/pipeline/05_dedup.py

echo "== [06] split =="
.venv/bin/python scripts/pipeline/06_split.py

echo "== [07] tokenize =="
.venv/bin/python scripts/pipeline/07_tokenize_stats.py all

echo "== [08] report =="
.venv/bin/python scripts/pipeline/08_report.py

echo "== 完成。产物见 dataset/processed/ 与 dataset/reports/pipeline/ =="
