import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "plot_trainlog", Path(__file__).parents[1] / "scripts/plot_trainlog.py"
)
plot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plot
spec.loader.exec_module(plot)


class PlotTrainlogTests(unittest.TestCase):
    def parse(self, lines):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainlog.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return plot.parse_log(path)

    def test_cuda_metrics_and_resume_discard_abandoned_history(self):
        prefix = "2026-10-02 20:00:00,000 INFO jishui.pretrain_torch: "
        def train(step):
            return prefix + (f"step={step} loss=4.1 ppl=60.3 lr=1.000e-04 grad_norm=1.2 "
                             "tokens=10000 tok/s=7300 peak_vram_gib=6.30 scale=256 eta_hours=1.0")
        points, validation, _, malformed = self.parse([
            train(100), train(101),
            prefix + "validation step=101 loss=4.0 ppl=54.6 per_category=c1=55.0",
            prefix + "resumed=runs/checkpoints/step_00000100 step=100 tokens=10000 final_step=200",
        ])
        self.assertEqual([point.step for point in points], [100])
        self.assertEqual(points[0].tok_s, 7300)
        self.assertEqual(validation, [])
        self.assertEqual(malformed, [])

    def test_mlx_format_and_resume_remain_supported(self):
        prefix = "2026-08-07 20:00:00,000 INFO jishui.train_mlx: "
        def train(step):
            return prefix + (f"step={step} loss=4.1 ppl=60.3 lr=1e-4 grad_norm=1.2 "
                             "tokens=10000 tok/s=1200")
        points, _, _, _ = self.parse([
            train(100), train(101), prefix + "resumed runs/checkpoints at step=100 tokens=10000",
        ])
        self.assertEqual([point.step for point in points], [100])


if __name__ == "__main__":
    unittest.main()
