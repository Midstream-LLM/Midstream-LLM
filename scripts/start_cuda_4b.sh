#!/usr/bin/env bash
# Run from any directory. Optional first argument: bootstrap checkpoint.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
run_dir="runs/jishui-200m-cuda-4b"
mkdir -p "$run_dir"
resume="${1:-runs/jishui-200m-stage0-1b/checkpoints/step_00055200}"
if [[ -f "$run_dir/latest.json" ]]; then
    resume="latest"
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unit="jishui-200m-cuda-4b"
extra_env=()
# Inherit proxy values without putting their contents on this command line.
for variable in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
    if [[ -v "$variable" ]]; then
        export "$variable"
        extra_env+=("--setenv=$variable")
    fi
done
systemd-run --user --collect --unit="$unit" --service-type=exec \
    --working-directory="$PWD" --setenv=PYTHONPATH --setenv=PYTHONUNBUFFERED \
    "${extra_env[@]}" \
    --property=Restart=no --property=TimeoutStopSec=300 --property=KillMode=mixed \
    --property="StandardOutput=append:$PWD/$run_dir/console.log" \
    --property="StandardError=append:$PWD/$run_dir/console.log" \
    "$PWD/.venv-torch/bin/python" -m jishui.pretrain_torch \
    --config configs/jishui-200m-cuda-4b.json \
    --run-dir "$run_dir" --resume "$resume" --evaluate-first
systemctl --user show "$unit.service" --property=ActiveState --property=MainPID
printf 'Training log: %s/trainlog.txt\n' "$run_dir"
