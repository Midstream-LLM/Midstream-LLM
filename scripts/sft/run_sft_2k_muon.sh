#!/usr/bin/env bash
# Muon SFT of the 2K Muon base, then the comparison eval.
# Everything that affects the result is pinned in configs/sft-v1-simplified-2k-muon.json.
#
#   bash scripts/sft/run_sft_2k_muon.sh check    # config + data + parent checks only, no training
#   bash scripts/sft/run_sft_2k_muon.sh smoke    # 20 updates into a throwaway run dir
#   bash scripts/sft/run_sft_2k_muon.sh start    # the real run (~1,434 updates)
#   bash scripts/sft/run_sft_2k_muon.sh resume   # exact resume after an interruption
#   bash scripts/sft/run_sft_2k_muon.sh eval     # score the final checkpoint (final.json) against the 11-24 SFTs
#   CKPT=<checkpoint dir> bash scripts/sft/run_sft_2k_muon.sh eval   # score another saved checkpoint
#
# Training runs on 8 GPUs with torchrun (DDP; each update sees the same batch as one GPU).
# Only PYTHON, NPROC (1, 2, 4 or 8; must divide grad_accum 8) and GPUS may be set.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-}"
PYTHON="${PYTHON:-.venv-torch/bin/python}"
NPROC="${NPROC:-8}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
export PYTHONPATH=src
CONFIG=configs/sft-v1-simplified-2k-muon.json
RUN_DIR=runs/jishui-200m-sft-v1-simplified-2k-muon
EVAL_DIR=runs/evaluations/sft-2k-muon

train() {
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m jishui.sft_torch --config "$CONFIG" --device cuda "$@"
}

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
    if [ -n "${CKPT:-}" ]; then
      TARGET="$CKPT"; OUT="$EVAL_DIR-$(basename "$CKPT")"
    else
      TARGET="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$RUN_DIR/final.json")"
      OUT="$EVAL_DIR"
    fi
    "$PYTHON" scripts/sft/eval_sft_compare.py --config "$CONFIG" \
      --checkpoint "$TARGET" --out "$OUT" --device cuda:0   # eval: first visible GPU only
    ;;
  *)
    echo "usage: $0 check|smoke|start|resume|eval" >&2
    exit 2
    ;;
esac
