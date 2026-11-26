#!/usr/bin/env bash
# SFT of the 2K Muon base (2026-11-24 2K recipe; Muon SFT by default), then the comparison eval.
# Everything that affects the result is pinned in the config JSON selected by VARIANT.
#
#   bash scripts/sft/run_sft_2k_muon.sh check    # config + data + parent checks only, no training
#   bash scripts/sft/run_sft_2k_muon.sh smoke    # 20 updates into a throwaway run dir
#   bash scripts/sft/run_sft_2k_muon.sh start    # the real run (~1,434 updates)
#   bash scripts/sft/run_sft_2k_muon.sh resume   # exact resume after an interruption
#   bash scripts/sft/run_sft_2k_muon.sh eval     # score the final checkpoint (final.json) against the 11-24 SFTs
#
# Only PYTHON, GPU and VARIANT may be set from the environment.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-}"
PYTHON="${PYTHON:-.venv-torch/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONPATH=src
# VARIANT=muon (main: Muon SFT) or adamw (control: the original AdamW recipe).
VARIANT="${VARIANT:-muon}"
case "$VARIANT" in
  muon)  SUFFIX="" ;;
  adamw) SUFFIX="-adamw" ;;
  *) echo "VARIANT must be muon or adamw" >&2; exit 2 ;;
esac
CONFIG="configs/sft-v1-simplified-2k-muon${SUFFIX}.json"
RUN_DIR="runs/jishui-200m-sft-v1-simplified-2k-muon${SUFFIX}"
EVAL_DIR="runs/evaluations/sft-2k-muon${SUFFIX}"

train() { "$PYTHON" -m jishui.sft_torch --config "$CONFIG" --device cuda "$@"; }

case "$MODE" in
  check)
    "$PYTHON" - "$CONFIG" <<'EOF'
import json, math, sys
from pathlib import Path
from jishui.sft_torch import CompactSFT, batch_plan, check_expectations, parse_args
args = parse_args(["--config", sys.argv[1], "--run-dir", "/nonexistent", "--device", "cpu"])
data = CompactSFT(args.data_dir)
plans = [batch_plan(data.lengths, data.splits["train"], args.micro_tokens, args.seed, e) for e in range(args.epochs)]
steps = [math.ceil(len(p) / args.grad_accum) for p in plans]
parent = json.loads((args.init_from / "train_state.json").read_text())
check_expectations(args, data, sum(steps), steps, parent)
print("CHECK OK: data, tokenizer, parent step/model and batch plan match", sys.argv[1], "steps", steps)
EOF
    ;;
  smoke)
    train --run-dir "${RUN_DIR}-smoke-$(date +%Y%m%d-%H%M%S)" --max-steps 20
    ;;
  start)
    train --run-dir "$RUN_DIR"
    ;;
  resume)
    train --run-dir "$RUN_DIR" --resume
    ;;
  eval)
    FINAL="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$RUN_DIR/final.json")"
    "$PYTHON" scripts/sft/eval_sft_compare.py --config "$CONFIG" \
      --checkpoint "$FINAL" --out "$EVAL_DIR" --device cuda:0
    ;;
  *)
    echo "usage: $0 check|smoke|start|resume|eval" >&2
    exit 2
    ;;
esac
