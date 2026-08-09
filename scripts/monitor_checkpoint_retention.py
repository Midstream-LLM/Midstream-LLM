#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jishui.checkpoints import prune_checkpoints


LOG = logging.getLogger("jishui.checkpoint_retention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce checkpoint retention for a running trainer")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--optimizer-checkpoints", type=int, default=3)
    return parser.parse_args()


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    while process_exists(args.pid):
        try:
            prune_checkpoints(
                args.run_dir,
                max_checkpoints=args.max_checkpoints,
                optimizer_checkpoints=args.optimizer_checkpoints,
            )
        except OSError:
            LOG.exception("checkpoint retention pass failed; retrying")
        time.sleep(args.interval)
    prune_checkpoints(
        args.run_dir,
        max_checkpoints=args.max_checkpoints,
        optimizer_checkpoints=args.optimizer_checkpoints,
    )
    LOG.info("trainer pid %d exited; final retention pass complete", args.pid)


if __name__ == "__main__":
    main()
