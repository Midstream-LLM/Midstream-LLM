"""train_watch.py 的回归测试。

这两个缺陷都是**在真实 run 上误报过**才被发现的（2026-10-07）：

1. **陈旧规则不认识「已完成」**——训练正常收官后 trainlog 停止增长，
   `STALE_WARN_S`/`STALE_ALERT_S` 把「跑完了」报成「日志 8.0 分钟无更新」并推了飞书。
2. **回升规则不认识「余弦重启」**——LR 被重置回峰值，验证 loss 短期必然抬升，
   `VAL_REGRESS` 把预期抬升报成退化。

两者的共同形状是「**规则不识别已知的预期状态**」，所以测试的重点不只是「误报没了」，
还要证明**反向场景仍然会告警**——否则修法就变成了把告警能力一起关掉。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import train_watch  # noqa: E402

_NOW = datetime.now()


def _ts(minutes_ago: float) -> str:
    return (_NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S") + ",000"


# 与 pretrain_torch 实际写出的行同形（RUN_START_RE / STEP_RE / VAL_RE / TARGET_RE 都必须匹配）。
# eta_hours 固定为 0.0：它在末尾为假值，能关掉「自算 ETA vs trainer ETA」这条无关规则，
# 使本文件的断言只针对被测的两条规则（该规则本次未被改动）。
def _start(minutes_ago: float) -> str:
    return f"{_ts(minutes_ago)} INFO jishui.pretrain_torch: device=cuda:0 torch=2.9.0 parameters=200224640"


def _step(n: int, minutes_ago: float) -> str:
    return (f"{_ts(minutes_ago)} INFO jishui.pretrain_torch: step={n} loss=3.5 ppl=33.1 lr=1e-4 "
            f"grad_norm=1.5 tokens={n * 16384} tok/s=7400 peak_vram_gib=6.30 scale=65536 eta_hours=0.0")


def _val(n: int, loss: float, minutes_ago: float) -> str:
    return f"{_ts(minutes_ago)} INFO jishui.pretrain_torch: validation step={n} loss={loss} ppl=6.0"


def _save(n: int, minutes_ago: float) -> str:
    return (f"{_ts(minutes_ago)} INFO jishui.pretrain_torch: "
            f"saved checkpoint runs/x/checkpoints/step_{n:08d}")


def _target(n: int, minutes_ago: float) -> str:
    return f"{_ts(minutes_ago)} INFO jishui.pretrain_torch: target reached step={n} tokens={n * 16384}"


# 与本 run 的 resolved_config.json 同形；cycle_boundaries 的真实取值就是这四个。
_BOUNDARIES = [55200, 118181, 181161, 244141]


def _config(max_steps: int = 118400, save_interval: int = 0) -> dict:
    return {
        "training": {"seq_len": 2048, "batch_size": 1, "grad_accum_steps": 8,
                     "max_steps": max_steps, "save_interval": save_interval},
        "continuation": {"target_tokens": 2_000_000_000},
        "schedule": {"mode": "cosine_warm_restarts", "cycle_boundaries": list(_BOUNDARIES),
                     "peak_lr": 1e-4, "end_lr": 1e-5, "warmup_steps": 1000},
    }


class TrainWatchTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def make_dir(self, lines: list[str], name: str = "run", **cfg_kwargs) -> Path:
        run_dir = Path(self._tmp.name) / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trainlog.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (run_dir / "resolved_config.json").write_text(json.dumps(_config(**cfg_kwargs)), encoding="utf-8")
        return run_dir

    def build(self, lines: list[str], name: str = "run", **cfg_kwargs):
        return train_watch.build(self.make_dir(lines, name, **cfg_kwargs))


class CompletedRunTest(TrainWatchTestCase):
    """缺陷 1：收官后日志停止增长被误报为陈旧。"""

    def _finished(self, save_minutes_ago: float = 60.0, target_minutes_ago: float = 40.0):
        # 最后一步发生在 60 分钟前，收尾行 40 分钟前 —— 远早于 900s 的 ALERT 阈值。
        return self.build(
            [_start(70), _step(100, 60), _val(100, 1.90, 60),
             _target(100, target_minutes_ago), _save(100, save_minutes_ago)],
            name="finished", max_steps=100, save_interval=400,
        )

    def test_completed_run_is_flagged(self):
        rep = self._finished()
        self.assertTrue(rep.infos["completed"])
        self.assertIn("completed_ts", rep.infos)

    def test_completed_run_does_not_warn_about_stale_log(self):
        rep = self._finished()
        self.assertTrue(rep.ok, f"不应判异常：{rep.alerts}")
        self.assertEqual([w for w in rep.warns if "无更新" in w], [])

    def test_completed_run_does_not_warn_about_old_checkpoint(self):
        rep = self._finished(save_minutes_ago=60.0)
        self.assertEqual([w for w in rep.warns if "存点" in w], [],
                         "已收官时「距上次存点过久」也是误报")

    def test_render_says_completed(self):
        self.assertIn("✅ 已完成", self._finished().text)
        self.assertIn("训练已完成", self._finished().text)

    def test_stale_log_still_alerts_while_running(self):
        """反例：没有收尾行（进程还在跑但日志 25 分钟没动）→ 必须仍然告警。"""
        rep = self.build([_start(30), _step(100, 25), _val(100, 1.90, 25)], name="stale_running")
        self.assertFalse(rep.infos["completed"])
        self.assertFalse(rep.ok)
        self.assertTrue(any("无更新" in a for a in rep.alerts), rep.alerts)

    def test_old_checkpoint_still_warns_while_running(self):
        """反例：同夹具去掉收尾行 → 存点过久必须仍然告警。"""
        rep = self.build([_start(70), _step(100, 60), _step(1200, 1), _save(100, 60)],
                         name="ck_stale", max_steps=5000, save_interval=400)
        self.assertFalse(rep.infos["completed"])
        self.assertTrue(any("存点" in w for w in rep.warns), rep.warns)

    def test_completion_does_not_immunize_a_later_run(self):
        """收官后又启动了新进程：收尾行属于上一段，陈旧规则必须恢复生效。"""
        rep = self.build([_start(90), _step(100, 60), _target(100, 45),
                          _start(30), _step(101, 25), _val(100, 1.90, 25)],
                         name="restarted", max_steps=100)
        self.assertFalse(rep.infos["completed"])
        self.assertFalse(rep.ok)
        self.assertTrue(any("无更新" in a for a in rep.alerts), rep.alerts)

    def test_completion_requires_reaching_max_steps(self):
        """收尾行的 step 未达 max_steps（例如配置被改大后续跑）时不算完成。"""
        rep = self.build([_start(70), _step(100, 60), _target(100, 40)], name="partial", max_steps=200)
        self.assertFalse(rep.infos["completed"])


    def test_brief_says_completed_instead_of_a_zero_eta(self):
        """--brief 是每小时巡检的实际调用形态，不能和完整报告给出互相矛盾的状态。

        踩过：完整报告已改成「训练已完成」，brief 却仍打印 `ETA=0.0h`。
        """
        import contextlib
        import io

        run_dir = self.make_dir([_start(70), _step(100, 60), _target(100, 40)],
                                name="brief_done", max_steps=100)
        buf = io.StringIO()
        argv = sys.argv
        sys.argv = ["train_watch.py", "--run-dir", str(run_dir), "--brief"]
        try:
            with contextlib.redirect_stdout(buf):
                rc = train_watch.main()
        finally:
            sys.argv = argv
        out = buf.getvalue()
        self.assertEqual(rc, 0, out)
        self.assertIn("✅已完成", out)
        self.assertNotIn("ETA=", out)

    def test_brief_still_shows_eta_while_running(self):
        run_dir = self.make_dir([_start(5), _step(100, 1)], name="brief_running", max_steps=100000)
        rep = train_watch.build(run_dir)
        self.assertFalse(rep.infos["completed"])
        self.assertIsNotNone(rep.infos["eta_hours"])


class RestartBumpTest(TrainWatchTestCase):
    """缺陷 2：余弦重启的预期抬升被误报为验证 loss 退化。"""

    def test_rise_inside_restart_window_is_not_a_warning(self):
        rep = self.build([_start(5), _step(118200, 1), _val(118000, 1.80, 3), _val(118200, 1.90, 1)],
                         name="restart")
        self.assertEqual([w for w in rep.warns if "验证 loss" in w], [], rep.warns)

    def test_rise_inside_restart_window_is_still_reported(self):
        """不告警 ≠ 不让你看见：仍须落进信息区。"""
        rep = self.build([_start(5), _step(118200, 1), _val(118000, 1.80, 3), _val(118200, 1.90, 1)],
                         name="restart_info")
        self.assertIn("val_rise_expected", rep.infos)
        self.assertIn("+0.1000", rep.infos["val_rise_expected"])
        self.assertIn("118,181", rep.infos["val_rise_expected"])

    def test_rise_inside_warmup_of_first_boundary_is_exempt(self):
        """cycle 0 的 warmup 同样会把 LR 抬上去——第一个边界也要豁免。"""
        rep = self.build([_start(5), _step(56000, 1), _val(55800, 1.80, 3), _val(56000, 1.90, 1)],
                         name="warmup")
        self.assertEqual([w for w in rep.warns if "验证 loss" in w], [], rep.warns)
        self.assertIn("val_rise_expected", rep.infos)

    def test_rise_outside_restart_window_still_warns(self):
        """反例：远离任何边界的 >0.05 回升必须仍然告警。"""
        rep = self.build([_start(5), _step(150000, 1), _val(149800, 1.80, 3), _val(150000, 1.90, 1)],
                         name="genuine")
        self.assertTrue(any("验证 loss 回升" in w for w in rep.warns), rep.warns)

    def test_small_rise_is_never_a_warning(self):
        rep = self.build([_start(5), _step(150000, 1), _val(149800, 1.80, 3), _val(150000, 1.84, 1)],
                         name="small")
        self.assertEqual([w for w in rep.warns if "验证 loss" in w], [], rep.warns)

    def test_final_boundary_is_not_treated_as_a_restart(self):
        """244141 是终末步、不是重启点：它之后没有「预期抬升」这回事。"""
        rep = self.build([_start(5), _step(244110, 1), _val(244000, 1.80, 3), _val(244100, 1.90, 1)],
                         name="final_boundary")
        self.assertTrue(any("验证 loss 回升" in w for w in rep.warns), rep.warns)


class RestartWithinHelperTest(unittest.TestCase):
    """窗口边界本身：既要覆盖实测到的最大距离，又不能无限宽。"""

    def setUp(self):
        self.sched = {"cycle_boundaries": list(_BOUNDARIES)}

    def test_hit_just_after_boundary(self):
        self.assertEqual(train_watch.restart_within(118181, self.sched), 118181)
        self.assertEqual(train_watch.restart_within(118181 + train_watch.RESTART_BUMP_STEPS, self.sched), 118181)

    def test_miss_before_boundary_and_past_window(self):
        self.assertIsNone(train_watch.restart_within(118180, self.sched))
        self.assertIsNone(train_watch.restart_within(118181 + train_watch.RESTART_BUMP_STEPS + 1, self.sched))
        self.assertIsNone(train_watch.restart_within(150000, self.sched))

    def test_final_boundary_excluded(self):
        self.assertIsNone(train_watch.restart_within(_BOUNDARIES[-1], self.sched))

    def test_window_covers_every_rise_observed_in_the_real_run(self):
        """窗口尺寸是实测定的，不是拍的。

        2026-10-07 统计整段 run 的 946 个验证点：相邻回升 >0.05 共 5 次，
        全部落在边界后 <=800 步（56000/118200/118400/181200/181400）。
        窗口若被改小到覆盖不住它们，这个测试会失败。
        """
        observed = [56000, 118200, 118400, 181200, 181400]
        for step in observed:
            with self.subTest(step=step):
                self.assertIsNotNone(train_watch.restart_within(step, self.sched))

    def test_missing_or_empty_boundaries_is_safe(self):
        self.assertIsNone(train_watch.restart_within(150000, {}))
        self.assertIsNone(train_watch.restart_within(150000, {"cycle_boundaries": []}))


if __name__ == "__main__":
    unittest.main()
