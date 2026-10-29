import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from jishui.tracking import ExperimentTracker


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.sdk = MagicMock()
        self.sdk.init.return_value.url = "https://wandb.ai/team/project/runs/test"
        self.settings = {"entity": "team", "project": "project", "mode": "online"}

    def start(self, mode=None, step=100):
        tracker = ExperimentTracker(self.path, self.settings, mode)
        tracker.start({"training": {"max_steps": 1000}}, step, step * 16384)
        return tracker

    def test_online_resume_keeps_identity_and_accepts_checkpoint_replay(self):
        with patch.dict("sys.modules", {"wandb": self.sdk}):
            first = self.start()
            first.log({"train/loss": 4}, 110, 110 * 16384)
            first.finish("stopped")
            run_id = self.sdk.init.call_args.kwargs["id"]
            second = self.start(step=105)
            self.assertEqual(self.sdk.init.call_args.kwargs["id"], run_id)
            self.assertEqual(self.sdk.init.call_args.kwargs["resume"], "allow")
            self.assertEqual(second.attempt, 2)
            # Use auto-incrementing W&B _step; global optimizer step is a custom
            # axis. Replayed updates after checkpoint recovery must not be lost.
            second.log({"train/loss": 3.9}, 106, 106 * 16384)
            args, kwargs = self.sdk.init.return_value.log.call_args
            self.assertNotIn("step", kwargs)
            self.assertEqual(args[0]["progress/optimizer_step"], 106)
            self.assertEqual(args[0]["resume/attempt"], 2)

    def test_offline_sessions_are_distinct_in_same_group(self):
        with patch.dict("sys.modules", {"wandb": self.sdk}):
            first = self.start(mode="offline")
            first.finish("stopped")
            first_args = dict(self.sdk.init.call_args.kwargs)
            self.start(mode="offline")
            second_args = self.sdk.init.call_args.kwargs
            self.assertNotEqual(first_args["id"], second_args["id"])
            self.assertEqual(first_args["group"], second_args["group"])
            self.assertIsNone(second_args["resume"])

    def test_validation_and_checkpoint_metrics(self):
        with patch.dict("sys.modules", {"wandb": self.sdk}):
            tracker = self.start()
            tracker.validation({"step": 101, "loss": 4, "per_category_loss": {1: 3, 6: 5}}, 2000000)
            record = self.sdk.init.return_value.log.call_args.args[0]
            self.assertEqual(record["val/c6_loss"], 5)
            self.assertIn("val/c1_perplexity", record)
            tracker.checkpoint(Path("checkpoints/step_00000101"), 101, 2000000)
            self.sdk.init.return_value.log_artifact.assert_not_called()
            self.sdk.init.return_value.watch.assert_not_called()
            self.assertEqual(tracker.identity["project"], "project")

    def test_changed_destination_rejected(self):
        with patch.dict("sys.modules", {"wandb": self.sdk}):
            self.start().finish("stopped")
            changed = ExperimentTracker(self.path, dict(self.settings, project="wrong"))
            with self.assertRaisesRegex(ValueError, "destination changed"):
                changed.start({}, 100, 1000)

    def test_online_failure_does_not_silently_disable_tracking(self):
        with patch.dict("sys.modules", {"wandb": self.sdk}):
            self.sdk.init.side_effect = RuntimeError("not authenticated")
            with self.assertRaisesRegex(RuntimeError, "not authenticated"):
                self.start()
            self.assertFalse((self.path / "wandb_run.json").exists())

    def test_disabled_needs_no_sdk_or_files(self):
        tracker = ExperimentTracker(self.path, self.settings, "disabled")
        with patch.dict("sys.modules", {"wandb": None}):
            tracker.start({}, 100, 1000)
            tracker.validation({"step": 100, "loss": 4, "per_category_loss": {}}, 1000)
            tracker.checkpoint(self.path, 100, 1000)
            tracker.finish("completed")
        self.assertIsNone(tracker.identity)
        self.assertEqual(list(self.path.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
