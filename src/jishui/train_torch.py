from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from typing import Sequence

from .distributed import (
    DistributedCapabilityError,
    LaunchEnvironment,
    build_runtime_plan,
    check_runtime,
    initialize_runtime,
    probe_accelerators,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe and validate Jishui PyTorch distributed runtimes"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--probe",
        action="store_true",
        help="report CUDA, MPS, Apple ANE, and Ascend torch_npu capabilities",
    )
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="resolve the torchrun rank/device plan without importing PyTorch",
    )
    mode.add_argument(
        "--check-runtime",
        action="store_true",
        help="resolve the plan and verify that its runtime is usable",
    )
    mode.add_argument(
        "--collective-smoke",
        action="store_true",
        help="initialize, barrier, and destroy the requested process group",
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        help=(
            "auto, cpu, cuda, mps, apple-ane, or ascend-npu. Comma-separated "
            "heterogeneous requests are rejected explicitly."
        ),
    )
    return parser.parse_args(argv)


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.probe:
            probes = probe_accelerators()
            _print_json(
                {
                    "host": {
                        "system": platform.system(),
                        "machine": platform.machine(),
                    },
                    "accelerators": {
                        name: probe.to_dict() for name, probe in probes.items()
                    },
                }
            )
            return 0

        launch = LaunchEnvironment.from_environ()
        plan = build_runtime_plan(args.accelerator, launch)
        result = {"launch": asdict(launch), "plan": plan.to_dict()}
        if args.plan_only:
            _print_json(result)
            return 0

        probe = check_runtime(plan)
        result["probe"] = probe.to_dict()
        if args.check_runtime:
            _print_json(result)
            return 0

        with initialize_runtime(plan) as runtime:
            if runtime.plan.distributed:
                runtime.torch.distributed.barrier()
            result["collective_smoke"] = "passed"
            _print_json(result)
        return 0
    except DistributedCapabilityError as exc:
        print(f"runtime configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
