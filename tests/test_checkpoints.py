from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jishui.checkpoints import latest_checkpoint, prune_checkpoints


class CheckpointRetentionTest(unittest.TestCase):
    def test_keeps_newest_complete_checkpoints_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint_root = run_dir / "checkpoints"
            checkpoint_root.mkdir()
            for step in (200, 400, 600, 800):
                checkpoint = checkpoint_root / f"step_{step:08d}"
                checkpoint.mkdir()
                (checkpoint / "optimizer.npz").write_bytes(b"adam")
            (checkpoint_root / ".step_00001000.tmp").mkdir()
            (checkpoint_root / "step_notes").mkdir()

            prune_checkpoints(run_dir, max_checkpoints=3)

            self.assertEqual(
                [path.name for path in sorted(checkpoint_root.iterdir())],
                [".step_00001000.tmp", "step_00000400", "step_00000600", "step_00000800", "step_notes"],
            )
            self.assertEqual(latest_checkpoint(run_dir).name, "step_00000800")

    def test_can_trim_adam_from_older_retained_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint_root = run_dir / "checkpoints"
            checkpoint_root.mkdir()
            for step in (100, 200, 300):
                checkpoint = checkpoint_root / f"step_{step:08d}"
                checkpoint.mkdir()
                (checkpoint / "optimizer.npz").write_bytes(b"adam")

            prune_checkpoints(run_dir, max_checkpoints=3, optimizer_checkpoints=1)

            self.assertFalse((checkpoint_root / "step_00000100" / "optimizer.npz").exists())
            self.assertFalse((checkpoint_root / "step_00000200" / "optimizer.npz").exists())
            self.assertTrue((checkpoint_root / "step_00000300" / "optimizer.npz").exists())


if __name__ == "__main__":
    unittest.main()
