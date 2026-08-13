#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ANE_RESULT_RE = re.compile(r"^BENCHMARK_JSON\s+(\{.*\})$", re.MULTILINE)


@dataclass(frozen=True)
class Result:
    worker: str
    microsteps: int
    tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "worker": self.worker,
            "microsteps": self.microsteps,
            "tokens": self.tokens,
            "seconds": self.seconds,
            "tokens_per_second": self.tokens_per_second,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure checkpoint-free MLX, ANE, and concurrent throughput"
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--records", type=Path, default=Path("/private/tmp/jishui_ane_smoke.bin"))
    parser.add_argument("--mlx-microsteps", type=int, default=16)
    parser.add_argument("--ane-microsteps", type=int, default=4)
    parser.add_argument("--warmup-microsteps", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--mode", choices=("mlx", "ane", "concurrent", "all"), default="all")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def run_checked(
    command: list[str], *, cwd: Path, timeout: float, environment: dict[str, str]
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}"
        )
    return completed.stdout


def parse_mlx(output: str) -> Result:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("worker") == "mlx":
            return Result(
                worker="mlx",
                microsteps=int(value["microsteps"]),
                tokens=int(value["tokens"]),
                seconds=float(value["seconds"]),
            )
    raise ValueError("MLX benchmark output did not contain a result record")


def parse_ane(output: str, microsteps: int, seq_len: int = 2048) -> Result:
    match = ANE_RESULT_RE.search(output)
    if not match:
        raise ValueError("ANE benchmark output did not contain BENCHMARK_JSON")
    value = json.loads(match.group(1))
    if int(value["microsteps"]) != microsteps or int(value["tokens"]) != microsteps * seq_len:
        raise ValueError("ANE benchmark result does not match the requested work")
    return Result(
        worker="ane",
        microsteps=microsteps,
        tokens=microsteps * seq_len,
        seconds=float(value["seconds"]),
    )


def mlx_command(args: argparse.Namespace, ready: Path | None = None, start: Path | None = None) -> list[str]:
    command = [
        str(args.repo / ".venv/bin/python"),
        "-m",
        "jishui.benchmark_mlx",
        "--config",
        "configs/jishui-200m-mlx.json",
        "--records",
        str(args.records),
        "--microsteps",
        str(args.mlx_microsteps),
        "--warmup-microsteps",
        str(args.warmup_microsteps),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--barrier-timeout",
        str(args.timeout),
    ]
    if ready and start:
        command.extend(("--ready-file", str(ready), "--start-file", str(start)))
    return command


def ane_command(args: argparse.Namespace, ready: Path | None = None, start: Path | None = None) -> list[str]:
    command = [
        str(args.repo / "native/ane/training_dynamic/train"),
        "--scratch",
        "--records",
        "--data",
        str(args.records),
        "--steps",
        str(args.ane_microsteps),
        "--accum",
        str(args.ane_microsteps),
        "--warmup",
        "1",
        "--log-interval",
        str(max(1, args.ane_microsteps)),
        "--no-checkpoint",
        "--barrier-timeout",
        str(math.ceil(args.timeout)),
    ]
    if ready and start:
        command.extend(("--ready-file", str(ready), "--start-file", str(start)))
    return command


def concurrent(args: argparse.Namespace) -> tuple[Result, Result, dict[str, float | int | str]]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.repo / "src")
    with tempfile.TemporaryDirectory(prefix="jishui-hybrid-") as directory:
        barrier = Path(directory)
        mlx_ready, ane_ready, start = barrier / "mlx.ready", barrier / "ane.ready", barrier / "start"
        mlx = subprocess.Popen(
            mlx_command(args, mlx_ready, start),
            cwd=args.repo,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        ane = subprocess.Popen(
            ane_command(args, ane_ready, start),
            cwd=args.repo / "native/ane/training_dynamic",
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + args.timeout
        try:
            while not (mlx_ready.exists() and ane_ready.exists()):
                if mlx.poll() is not None or ane.poll() is not None:
                    raise RuntimeError("worker exited before the concurrency barrier")
                if time.monotonic() >= deadline:
                    raise TimeoutError("workers did not reach the concurrency barrier")
                time.sleep(0.02)
            start.write_text("go\n", encoding="ascii")
            wall_started = time.perf_counter()
            mlx_output, _ = mlx.communicate(timeout=max(1.0, deadline - time.monotonic()))
            ane_output, _ = ane.communicate(timeout=max(1.0, deadline - time.monotonic()))
            wall_seconds = time.perf_counter() - wall_started
        except BaseException:
            for process in (mlx, ane):
                if process.poll() is None:
                    process.terminate()
            for process in (mlx, ane):
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
        if mlx.returncode or ane.returncode:
            raise RuntimeError(
                f"concurrent worker failed: mlx={mlx.returncode} ane={ane.returncode}\n"
                f"--- MLX ---\n{mlx_output}\n--- ANE ---\n{ane_output}"
            )
        mlx_result = parse_mlx(mlx_output)
        ane_result = parse_ane(ane_output, args.ane_microsteps)
        total_tokens = mlx_result.tokens + ane_result.tokens
        combined = {
            "worker": "concurrent",
            "tokens": total_tokens,
            "seconds": wall_seconds,
            "tokens_per_second": total_tokens / wall_seconds,
        }
        return mlx_result, ane_result, combined


def main() -> None:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.records = args.records.resolve()
    if min(args.mlx_microsteps, args.ane_microsteps, args.warmup_microsteps, args.grad_accum_steps) < 1:
        raise ValueError("all microstep counts must be positive")
    if args.mlx_microsteps % args.grad_accum_steps:
        raise ValueError("MLX microsteps must be divisible by grad accumulation")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.repo / "src")
    results: dict[str, object] = {}
    if args.mode in {"mlx", "all"}:
        output = run_checked(
            mlx_command(args), cwd=args.repo, timeout=args.timeout, environment=environment
        )
        results["mlx_alone"] = parse_mlx(output).as_dict()
    if args.mode in {"ane", "all"}:
        output = run_checked(
            ane_command(args), cwd=args.repo / "native/ane/training_dynamic",
            timeout=args.timeout, environment=environment
        )
        results["ane_alone"] = parse_ane(output, args.ane_microsteps).as_dict()
    if args.mode in {"concurrent", "all"}:
        mlx_result, ane_result, combined = concurrent(args)
        results["mlx_concurrent"] = mlx_result.as_dict()
        results["ane_concurrent"] = ane_result.as_dict()
        results["combined"] = combined
    if "mlx_alone" in results and "combined" in results:
        mlx_rate = float(results["mlx_alone"]["tokens_per_second"])
        combined_rate = float(results["combined"]["tokens_per_second"])
        results["speedup_over_mlx"] = combined_rate / mlx_rate
    encoded = json.dumps(results, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output:
        args.json_output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
