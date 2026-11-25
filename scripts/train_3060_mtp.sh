#!/usr/bin/env bash
# Single-GPU (RTX 3060) pretraining with DeepSeek-V3-style MTP (depth 1).
# Everything except MTP matches the 2K blocks baseline: causal attention, AdamW.
#
#   bash scripts/train_3060_mtp.sh smoke      # 20 updates into a throwaway run dir
#   bash scripts/train_3060_mtp.sh start      # new run from scratch
#   bash scripts/train_3060_mtp.sh resume     # exact resume of the same run
#
# Override with env vars: PYTHON, DATA_DIR, RUN_DIR, CONFIG.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
PYTHON="${PYTHON:-.venv-torch/bin/python}"
CONFIG="${CONFIG:-configs/jishui-200m-v3-mtp1-3060.json}"
DATA_DIR="${DATA_DIR:-dataset/corpus-date-open-v3}"
RUN_DIR="${RUN_DIR:-runs/jishui-200m-v3-mtp1-3060}"

export PYTHONPATH=src

launch() {
  "$PYTHON" -m jishui.pretrain_torch --config "$CONFIG" --data-dir "$DATA_DIR" "$@"
}

case "$MODE" in
  smoke)
    SMOKE_DIR="${RUN_DIR}-smoke-$(date +%Y%m%d-%H%M%S)"
    launch --run-dir "$SMOKE_DIR" --steps 20 --evaluate-first --wandb-mode disabled
    echo "smoke run complete: $SMOKE_DIR (safe to delete)"
    ;;
  start)
    launch --run-dir "$RUN_DIR" --evaluate-first
    ;;
  resume)
    launch --run-dir "$RUN_DIR" --resume latest
    ;;
  *)
    echo "usage: $0 smoke|start|resume" >&2
    exit 2
    ;;
esac
