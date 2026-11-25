#!/usr/bin/env bash
# 8×V100 DDP pretraining with document-masked attention and Muon.
#
#   bash scripts/train_8xv100_docmask.sh smoke      # 20 updates into a throwaway run dir
#   bash scripts/train_8xv100_docmask.sh start      # new run from scratch
#   bash scripts/train_8xv100_docmask.sh resume     # exact resume of the same run
#
# Override with env vars: PYTHON, DATA_DIR, RUN_DIR, NPROC, CONFIG.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
PYTHON="${PYTHON:-.venv-torch/bin/python}"
CONFIG="${CONFIG:-configs/jishui-200m-v3-docmask-8xv100.json}"
DATA_DIR="${DATA_DIR:-dataset/corpus-date-open-v3}"
RUN_DIR="${RUN_DIR:-runs/jishui-200m-v3-docmask-muon-8xv100}"
NPROC="${NPROC:-8}"

export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Dense SDPA mask at 2K (mem-efficient kernel, works on V100). See docs.
export JISHUI_ATTENTION_BACKEND="${JISHUI_ATTENTION_BACKEND:-auto}"

launch() {
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m jishui.pretrain_torch --config "$CONFIG" --data-dir "$DATA_DIR" "$@"
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
