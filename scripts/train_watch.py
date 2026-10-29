#!/usr/bin/env python3
"""Jishui 预训练进度巡检：读日志与 checkpoint，输出简报，可按需推送到飞书。

只读 runs/<run>/ 下的产物，不接触 GPU 与训练进程。设计为可在 DSH 沙箱内运行
（沙箱内看不到主机进程、没有 /dev/nvidia*，所以健康判断一律基于日志新鲜度与
数值本身，而不是 nvidia-smi）。

用法：
    python scripts/train_watch.py                     # 人类可读简报
    python scripts/train_watch.py --json              # 机器可读
    python scripts/train_watch.py --feishu            # 推送到飞书（需先配 targetId）
    python scripts/train_watch.py --feishu --force    # 无视异常才推的抑制策略
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

TIMESTAMP = r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}"
STEP_RE = re.compile(
    rf"^{TIMESTAMP} .*: step=(?P<step>\d+) loss=(?P<loss>[-+0-9.eE]+) ppl=(?P<ppl>[-+0-9.eE]+) "
    r"lr=(?P<lr>[-+0-9.eE]+) grad_norm=(?P<grad>[-+0-9.eE]+) tokens=(?P<tokens>\d+) "
    r"tok/s=(?P<tok_s>[-+0-9.eE]+) peak_vram_gib=(?P<vram>[-+0-9.eE]+) "
    r"scale=(?P<scale>[-+0-9.eE]+) eta_hours=(?P<eta>[-+0-9.eE]+)$"
)
VAL_RE = re.compile(
    rf"^{TIMESTAMP} .*: validation step=(?P<step>\d+) loss=(?P<loss>[-+0-9.eE]+) "
    r"ppl=(?P<ppl>[-+0-9.eE]+)"
)
SAVE_RE = re.compile(rf"^{TIMESTAMP} .*: saved checkpoint (?P<path>.+)$")
# 正常收尾行：trainer 到达 max_steps 后写下它，然后进程结束、日志停止增长。
TARGET_RE = re.compile(rf"^{TIMESTAMP} .*: target reached step=(?P<step>\d+) tokens=(?P<tokens>\d+)$")
# 进程启动行：trainlog.txt 会跨多次启动累加，ETA/均速必须以当前运行段为界，
# 否则重启间隙（演练/重启各数十秒~数分钟）会被算成训练时间。
RUN_START_RE = re.compile(rf"^{TIMESTAMP} .*: device=.*torch=.*parameters=\d+$")
# 真·致命：训练进程会终止或即将终止。
FATAL_RE = re.compile(r"(Traceback \(most recent call last\)|non-finite loss|FloatingPointError|CUDA out of memory|training failed)")
# 非致命但需记：GradScaler 检测到非有限梯度后回退 scale 并重试同一样本
# （pretrain_torch 最多重试 8 次，是设计内的处理，不是崩溃）。
OVERFLOW_RE = re.compile(rf"^{TIMESTAMP} .*: nonfinite gradient; retry=(?P<attempt>\d+) scale=(?P<scale>[0-9.eE+-]+)$")

# 巡检阈值
STALE_WARN_S = 300      # 日志 5 分钟无新行：可疑（存 checkpoint 约 20s，正常不会这么久）
STALE_ALERT_S = 900     # 15 分钟：按 ~2.3s/step 已丢约 400 步，判为异常
DISK_WARN_GIB = 15.0
VAL_REGRESS = 0.05      # 验证 loss 回升幅度告警阈值
# 溢出/scale 退回的「时近性」窗口，必须**略大于巡检间隔**。
# 取舍方向经实测定为「宁重复、勿漏报」：
#   · 窗口取 3500s（< 3600s 间隔）看似能避免重复，但会在每次巡检后留下约 100s 盲区——
#     2026-10-03 19:31:05 那次溢出一头栽进盲区，20:30 巡检 age=3535s 直接漏掉。
#   · 改取 3700s：事件在首次巡检后必被捕获（漏报=0）；代价是「发生在巡检前 100s 内」的
#     事件会被连续两次巡检各报一次（约 2.8% 概率）。因为单次单轮溢出本就不推送，
#     重复只可能落在少数真正该报的簇发/深重试上，可接受。
OVERFLOW_LOOKBACK_S = 3700
# 余弦重启（含 cycle 0 的 warmup）后，LR 被重置回峰值，验证 loss 短期**必然**抬升。
# 这段回升是预期行为，不该按「退化」告警。窗口取多大，由实测定：
#   2026-10-07 用整段 run 的 946 个验证点统计，相邻验证点回升 >0.05 仅 5 次，
#   且 5 次全部落在 cycle 边界后 ≤800 步内（56000/118200/118400/181200/181400），
#   无一例外。取 2000 步（≈46 分钟 @1560 步/h）即有 2.5 倍余量。
# 取舍：窗口内的真·退化会被一并豁免——但重启后那段本就噪声大、读不出趋势，
# 豁免它不损失有效信号；窗口**之外**的 >0.05 回升照旧告警。
RESTART_BUMP_STEPS = 2000
DEFAULT_FEISHU = REPO / "configs" / "train_watch.json"


@dataclass
class Report:
    run_dir: Path
    ok: bool = True
    alerts: list[str] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)
    infos: dict = field(default_factory=dict)
    text: str = ""


def parse_logs(run_dir: Path):
    trainlog = run_dir / "trainlog.txt"
    if not trainlog.exists():
        return None, None, None, None, None, None, None
    steps, vals, saves, bad, starts, overflows, completions = [], [], [], [], [], [], []
    with trainlog.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.match(line)
            if m:
                steps.append({
                    "ts": m["ts"], "step": int(m["step"]), "loss": float(m["loss"]),
                    "lr": float(m["lr"]), "grad": float(m["grad"]), "tokens": int(m["tokens"]),
                    "tok_s": float(m["tok_s"]), "vram": float(m["vram"]),
                    "scale": float(m["scale"]), "eta": float(m["eta"]),
                })
                continue
            m = VAL_RE.match(line)
            if m:
                vals.append({"ts": m["ts"], "step": int(m["step"]), "loss": float(m["loss"]), "ppl": float(m["ppl"])})
                continue
            m = SAVE_RE.match(line)
            if m:
                saves.append({"ts": m["ts"], "path": m["path"].strip()})
                continue
            m = RUN_START_RE.match(line)
            if m:
                starts.append(m["ts"])
                continue
            m = OVERFLOW_RE.match(line)
            if m:
                overflows.append({"ts": m["ts"], "attempt": int(m["attempt"]), "scale": float(m["scale"])})
                continue
            m = TARGET_RE.match(line)
            if m:
                completions.append({"ts": m["ts"], "step": int(m["step"]), "tokens": int(m["tokens"])})
                continue
            if FATAL_RE.search(line):
                bad.append(line.rstrip())
    return steps, vals, saves, bad, starts, overflows, completions


def schedule_phase(schedule: dict, step: int) -> str:
    """复述 schedules_torch.continuation_lr 的相位，用于人话描述。"""
    bounds = schedule.get("cycle_boundaries")
    peak = schedule.get("peak_lr")
    if not bounds:
        return f"cosine {schedule.get('start_lr'):.3e}→{schedule.get('end_lr'):.3e}"
    if step >= bounds[-1]:
        return "已到终末 lr"
    for i, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        if begin <= step < end:
            warm = schedule.get("warmup_steps", 0) if i == 0 else 0
            offset = step - begin
            if offset < warm:
                return f"cycle {i+1}/{len(bounds)-1} 线性 warmup，{warm-offset} 步后达峰值 {peak:.0e}"
            pct = 100.0 * (offset - warm) / max(1, end - begin - warm - 1)
            tail = f"；{end-step} 步后本 cycle 收于 {schedule.get('end_lr'):.0e}"
            if i + 2 < len(bounds):
                tail += "，随后重启回峰值"
            return f"cycle {i+1}/{len(bounds)-1} 余弦衰减 {pct:.0f}%{tail}"
    return "相位未知"


def restart_within(step: int, schedule: dict, window: int = RESTART_BUMP_STEPS) -> int | None:
    """step 是否落在某个「刚重启/刚热身」的 cycle 边界之后的 window 步内。

    `cycle_boundaries = [b0, b1, ..., bn]` 中 bn 是终末步、不是重启点，故只看 [:-1]。
    命中时返回该边界；否则 None。
    """
    bounds = schedule.get("cycle_boundaries") or []
    for b in bounds[:-1]:
        if 0 <= step - b <= window:
            return b
    return None


def disk_free_gib(path: Path) -> float | None:
    import shutil
    try:
        return shutil.disk_usage(path).free / 2**30
    except OSError:
        return None


def build(run_dir: Path, window: int = 100) -> Report:
    rep = Report(run_dir=run_dir)
    steps, vals, saves, bad, starts, overflows, completions = parse_logs(run_dir)
    if steps is None or not steps:
        rep.ok = False
        rep.alerts.append(f"找不到可解析的 step 记录：{run_dir}/trainlog.txt"
                          + ("（文件不存在）" if steps is None else "（文件存在但没有匹配的 step 行）"))
        # 必须落成文字：这正是最要命的场景（日志消失/被截断），沉默等于漏报。
        rep.text = render(rep)
        return rep

    resolved = {}
    rp = run_dir / "resolved_config.json"
    if rp.exists():
        resolved = json.loads(rp.read_text())
    training = resolved.get("training", {})
    cont = resolved.get("continuation", {})
    sched = resolved.get("schedule", {})
    step_tokens = int(training.get("seq_len", 2048)) * int(training.get("batch_size", 1)) * int(training.get("grad_accum_steps", 8))
    target_tokens = int(cont.get("target_tokens", 0)) or 0
    max_steps = int(training.get("max_steps", 0)) or 0
    save_interval = int(training.get("save_interval", 0)) or 0

    last = steps[-1]
    tokens = last["tokens"]
    recent = steps[-window:]
    prev = steps[-2 * window:-window] if len(steps) > window else []

    now = datetime.now()
    def parse_ts(ts: str) -> datetime:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    def age_s(ts: str) -> float:
        return (now - parse_ts(ts)).total_seconds()

    log_age = age_s(last["ts"])
    mean_recent = sum(s["loss"] for s in recent) / len(recent)
    mean_prev = sum(s["loss"] for s in prev) / len(prev) if prev else None
    rates = sorted(s["tok_s"] for s in recent)
    tok_s_median = rates[len(rates) // 2]          # 稳健：接近纯计算吞吐
    tok_s_recent_mean = sum(rates) / len(rates)    # 含窗口内的存点/验证停顿
    # grad_norm 是 clip 之前的值（clip_grad_norm_ 的返回值），且分布重尾：
    # 本 run 实测 p50=1.27 / p90=1.71 / p99=2.79 / p99.9=9.66，孤立尖峰到 ~10
    # 属单批离群、被裁到 1.0 后无害。故只看「中位数是否整体抬高」，不看单点峰值。
    win_grads = sorted(s["grad"] for s in recent)
    n_g = len(win_grads)
    grad_median = win_grads[n_g // 2]
    grad_p90 = win_grads[min(n_g - 1, max(0, int(0.9 * n_g) - 1))]
    grad_spikes = sum(1 for g in win_grads if g > 5)

    # ETA/均速只统计「当前运行段」：trainlog.txt 跨多次启动累加，重启间隙
    # 不是训练时间；而步号在重启处可能仍然连续（55210→55211），无法靠步号检测。
    # 起点 = 最后一个进程启动行（device=... torch=...）之后的第一个 step。
    run_start_ts = starts[-1] if starts else None
    # 训练正常收官后 trainlog 不再增长，这是**预期**而非陈旧（2026-10-07 实测踩过：
    # 收官 8 分钟后巡检把「已完成」报成「⚠️ 日志 8 分钟无更新」，并推了飞书）。
    # 只认「当前运行段内」的收尾行：若之后又启动了新进程，收尾行属于上一段，
    # 陈旧规则必须恢复生效——否则一次完成会永久免疫掉「进程真的死了」的检测。
    done = [c for c in completions if run_start_ts is None or c["ts"] >= run_start_ts]
    completed = bool(done) and (max_steps == 0 or done[-1]["step"] >= max_steps)
    rep.infos["completed"] = completed
    if completed:
        rep.infos["completed_ts"] = done[-1]["ts"]
    seg = [s for s in steps if run_start_ts is None or s["ts"] >= run_start_ts] or steps
    span_s = (parse_ts(seg[-1]["ts"]) - parse_ts(seg[0]["ts"])).total_seconds()
    span_tokens = seg[-1]["tokens"] - seg[0]["tokens"]
    full_rate = span_tokens / span_s if span_s > 0 else 0.0
    eta_mine = (target_tokens - tokens) / full_rate / 3600 if full_rate > 0 else None
    # 主用 trainer 自报的 eta_hours：它是进程内 average_rate 算的，用户可在
    # trainlog 里逐行核对；我自算的仅作交叉校验。
    eta_hours = last.get("eta")
    eta_seconds = (eta_hours * 3600) if eta_hours is not None else float("inf")

    rep.infos.update({
        "step": last["step"],
        "max_steps": max_steps,
        "step_pct": 100.0 * last["step"] / max_steps if max_steps else None,
        "tokens": tokens,
        "target_tokens": target_tokens,
        "token_pct": 100.0 * tokens / target_tokens if target_tokens else None,
        "lr": last["lr"],
        "grad_norm": last["grad"],
        "scale": last["scale"],
        "vram_gib": last["vram"],
        "tok_s_instant": last["tok_s"],
        "tok_s_median": tok_s_median,
        "tok_s_recent_mean": tok_s_recent_mean,
        "tok_s_full_avg": full_rate,
        "grad_median": grad_median,
        "grad_p90": grad_p90,
        "grad_spikes_gt5": grad_spikes,
        "loss_last": last["loss"],
        "loss_mean_window": mean_recent,
        "loss_mean_prev_window": mean_prev,
        "window": len(recent),
        "log_age_s": log_age,
        "last_log_ts": last["ts"],
        "eta_hours": eta_hours,
        "eta_hours_trainer": last.get("eta"),
        "eta_hours_mine": eta_mine,
        "run_start_ts": run_start_ts,
        "run_count": len(starts),
        "eta_wall": (now + timedelta(seconds=eta_seconds)).strftime("%Y-%m-%d %H:%M") if math.isfinite(eta_seconds) else None,
        "phase": schedule_phase(sched, last["step"]) if sched else None,
        "val_last": vals[-1] if vals else None,
        "val_prev": vals[-2] if len(vals) >= 2 else None,
        "saves_last": saves[-1] if saves else None,
        "disk_free_gib": disk_free_gib(REPO),
    })

    # ---- 健康判定 ----
    # 陈旧规则的前提是「进程还在跑」。已收官时日志必然停止增长，此时报陈旧是纯粹误报。
    if completed:
        pass
    elif log_age > STALE_ALERT_S:
        rep.ok = False
        rep.alerts.append(f"日志 {log_age/60:.0f} 分钟无更新（阈值 {STALE_ALERT_S/60:.0f} 分钟）——训练可能已停或卡死")
    elif log_age > STALE_WARN_S:
        rep.warns.append(f"日志 {log_age/60:.1f} 分钟无更新（存 checkpoint 约 20s，正常不会这么久）")

    bad_tail = [b for b in bad[-5:]]
    if bad_tail:
        rep.ok = False
        rep.alerts.append("日志出现异常行：" + " | ".join(bad_tail[-2:])[:300])

    # --- GradScaler / fp16 溢出判定 ---
    # 语义（torch.amp.GradScaler.update，已对 .venv-torch 源码核实）：
    #   · 走无参 update() 时：溢出则 scale×backoff、tracker 归零；连续 growth_interval
    #     （默认 2000）步无溢出则 scale×2、tracker 归零 —— 实测本 run 每 2000 步精确翻倍。
    #   · **但** trainer 的溢出路径调用的是 update(new_scale=scale/2)，该分支只写 scale、
    #     **不碰 _growth_tracker**。于是计数器保留，scale 可能在远少于 2000 步后翻回原位
    #     （2026-10-03 实测：73200 翻倍 → 74567 溢出(tracker=1367) → 再攒 633 步 → 75200 翻倍，
    #     1367+633=2000 精确吻合）。
    # ⇒ 判断溢出必须看 retry 日志行，不能靠 scale 变化方向或步数推断。
    # 每次溢出必然同时产生「一条 retry 行」与「一次 scale 下调」，属同一事件，只报一次。
    # 归组为「事件」：同一 step 的多条 retry 行（attempt=1,2,…）属于同一次溢出。
    # 计数必须用**事件数**，否则一次需要多次重试的溢出会被误当成簇发
    # （2026-10-03 15:20 实测：retry=1 与 retry=2 同属一步，被旧逻辑算成 2 次）。
    # 严重度用该事件内的最大重试轮次。
    seg_overflows = [o for o in overflows if run_start_ts is None or o["ts"] >= run_start_ts]
    ts_list = [s["ts"] for s in steps]
    def step_for(ts: str) -> int:
        i = bisect.bisect_left(ts_list, ts)
        return steps[i]["step"] if i < len(steps) else steps[-1]["step"]
    events: dict = {}
    for o in seg_overflows:
        k = step_for(o["ts"])
        ev = events.setdefault(k, {"step": k, "ts": o["ts"], "attempts": []})
        ev["attempts"].append(o["attempt"])
    recent_events = [e for e in events.values() if age_s(e["ts"]) <= OVERFLOW_LOOKBACK_S]
    # 无条件落进 infos：改为「单次溢出不告警」后，warn 计数已不再等价于「有无溢出」，
    # 读简报的人（包括我）不能再从 warn 反推溢出情况，必须直接给出次数。
    rep.infos["overflow_recent_n"] = len(recent_events)
    rep.infos["overflow_total_segment"] = len(events)
    if events:
        last_ev = max(events.values(), key=lambda e: e["ts"])
        rep.infos["overflow_last"] = {"ts": last_ev["ts"], "step": last_ev["step"],
                                      "attempts": max(last_ev["attempts"])}
    rises = [(b["step"], a["scale"], b["scale"], b["ts"]) for a, b in zip(steps, steps[1:]) if b["scale"] > a["scale"]]
    drops = [(b["step"], a["scale"], b["scale"], b["ts"]) for a, b in zip(steps, steps[1:]) if b["scale"] < a["scale"]]
    # 「过去 1 小时发生了什么」——全部从日志推算，无需状态文件。
    # 目的：让每小时心跳的输出天然是增量而不是重复快照；也便于人工核对趋势。
    cutoff_1h = now - timedelta(seconds=3600)
    past_steps = [s for s in steps if parse_ts(s["ts"]) <= cutoff_1h]
    past_vals = [v for v in vals if parse_ts(v["ts"]) <= cutoff_1h]
    d1h = {"ovr": sum(1 for e in events.values() if age_s(e["ts"]) <= 3600)}
    if past_steps:
        d1h["steps"] = last["step"] - past_steps[-1]["step"]
        d1h["tokens"] = tokens - past_steps[-1]["tokens"]
    if vals and past_vals:
        d1h["val"] = vals[-1]["loss"] - past_vals[-1]["loss"]
        d1h["val_from_to"] = (past_vals[-1]["step"], vals[-1]["step"])
    rep.infos["d1h"] = d1h
    recent_drops = [d for d in drops if age_s(d[3]) <= OVERFLOW_LOOKBACK_S]
    if recent_events:
        worst = max(max(e["attempts"]) for e in recent_events)
        last_ev = max(recent_events, key=lambda e: e["ts"])
        drop_txt = f"，scale {recent_drops[-1][1]:g}→{recent_drops[-1][2]:g} @ step {recent_drops[-1][0]}" if recent_drops else ""
        msg = (f"fp16 溢出 {len(recent_events)} 次事件（最近 @ {last_ev['ts'][11:]} step {last_ev['step']}"
               f"{drop_txt}，最高重试轮次 {worst}/8；{OVERFLOW_LOOKBACK_S/60:.0f} 分钟窗口内）")
        if worst >= 4 or len(recent_events) >= 3:
            rep.ok = False
            rep.alerts.append(msg + f"——溢出频率或重试深度异常（本运行段累计 {len(events)} 次事件），"
                                    "虽在自动重试但已偏离健康区间")
        elif worst >= 2 or len(recent_events) >= 2:
            # attempt≥2 = 一次减半不够用（torch 允许最多 8 次），值得知道；单纯单次单轮不推。
            rep.warns.append(msg + "——torch 已自动处理（同批样本降 scale 后重试成功）")
        else:
            rep.infos["overflow_recent_single"] = msg + "——单次且单轮重试，属稳态（详见累计行），未推送"
    if len(recent_drops) > len(recent_events):
        st, a0, b0, _ = recent_drops[-1]
        rep.warns.append(f"GradScaler 下调 {len(recent_drops)} 次，但只有 {len(recent_events)} 次溢出事件"
                         f"（最近 {a0:g}→{b0:g} @ step {st}）——存在无法归因的 scale 退回")
    if rises:
        st, a0, b0, _ = rises[-1]
        ov = rep.infos.get("overflow_total_segment", 0)
        tail = "无溢出" if not ov else f"本段溢出 {ov} 次"
        rep.infos["scale_note"] = (f"GradScaler：{last['scale']:g}（{len(rises)} 次翻倍，最近 {a0:g}→{b0:g} @ step {st}；{tail}）")
    if grad_median > 2.5:
        rep.warns.append(f"近 {n_g} 步 grad_norm 中位数 {grad_median:.2f}（基线 1.27）——梯度整体抬高，非单点尖峰")
    elif grad_spikes >= 3:
        rep.warns.append(f"近 {n_g} 步有 {grad_spikes} 步 grad_norm >5（基线仅 0.7%）——离群变密，需留意")

    if len(vals) >= 2:
        d = vals[-1]["loss"] - vals[-2]["loss"]
        if d > VAL_REGRESS:
            restart = restart_within(vals[-1]["step"], sched)
            if restart is None:
                rep.warns.append(f"验证 loss 回升 {d:+.4f}（step {vals[-2]['step']}→{vals[-1]['step']}）")
            else:
                # 落在重启窗口内：LR 被重置回峰值导致的预期抬升，不升级为告警。
                # 但**照旧落进信息区**——压制告警 ≠ 隐瞒数据，读简报的人仍应看见它。
                rep.infos["val_rise_expected"] = (
                    f"验证 loss 回升 {d:+.4f}（step {vals[-2]['step']}→{vals[-1]['step']}）"
                    f"——在重启点 {restart:,} 后 {RESTART_BUMP_STEPS} 步内，属预期抬升，未告警")
        rep.infos["val_delta"] = d

    if save_interval and steps and not completed:
        sec_per_step = 1.0 / max(1e-9, full_rate / step_tokens)
        if saves:
            ck_age = age_s(saves[-1]["ts"])
            if ck_age > 2.5 * save_interval * sec_per_step:
                rep.warns.append(f"最近一次存点已 {ck_age/60:.0f} 分钟前，超过 {save_interval} 步的预期间隔")
        else:
            rep.warns.append("本次启动后尚未见 saved checkpoint 行")

    if len(starts) > 1:
        rep.infos["note_runs"] = f"trainlog 含 {len(starts)} 次进程启动，均速/ETA 只按当前运行段（起始 {run_start_ts}）统计"
    if eta_mine and eta_hours and abs(eta_mine - eta_hours) / eta_hours > 0.15:
        rep.warns.append(f"ETA 自算 {eta_mine:.0f}h 与 trainer 自报 {eta_hours:.0f}h 相差 >15%（刚重启或存在停顿）")

    free = rep.infos["disk_free_gib"]
    if free is not None and free < DISK_WARN_GIB:
        rep.warns.append(f"磁盘仅剩 {free:.1f} GiB（每个 checkpoint 约 2.3 GiB）")

    rep.text = render(rep)
    return rep


def render(rep: Report) -> str:
    i = rep.infos
    stamp = datetime.now().strftime("%m-%d %H:%M")
    if not i:
        head = "⛔ 异常" if rep.alerts else ("⚠️ 注意" if rep.warns else "ℹ️ 无数据")
        lines = [f"【击水 Jishui-200M】训练巡检 · {stamp} · {head}"]
        lines += [f"异常：{a}" for a in rep.alerts]
        lines += [f"注意：{w}" for w in rep.warns]
        return "\n".join(lines)
    if i.get("completed"):
        head = "✅ 已完成" if rep.ok else "⛔ 异常"
    else:
        head = "✅ 正常" if rep.ok and not rep.warns else ("⛔ 异常" if not rep.ok else "⚠️ 注意")
    lines = [f"【击水 Jishui-200M】训练进度 · {stamp} · {head}"]
    step_s = f"step {i['step']:,}"
    if i.get("max_steps"):
        step_s += f" / {i['max_steps']:,}（{i['step_pct']:.1f}%）"
    lines.append(f"进度：{step_s}")
    if i.get("target_tokens"):
        lines.append(f"Token：{i['tokens']/1e9:.3f}B / {i['target_tokens']/1e9:.2f}B（{i['token_pct']:.1f}%）")
    loss_s = f"训练 loss：近{i['window']}步均值 {i['loss_mean_window']:.4f}"
    if i.get("loss_mean_prev_window"):
        loss_s += f"（更早 {i['window']} 步 {i['loss_mean_prev_window']:.4f}）"
    lines.append(loss_s)
    v = i.get("val_last")
    if v:
        vs = f"验证 loss：{v['loss']:.4f}（ppl {v['ppl']:.1f}）@ step {v['step']:,}"
        if i.get("val_delta") is not None:
            vs += f" · 较上次 {i['val_delta']:+.4f}"
        lines.append(vs)
    lines.append(f"速度：{i['tok_s_median']:.0f} tok/s（近{i['window']}步中位；含停顿均值 {i['tok_s_recent_mean']:.0f}）· 显存峰值 {i['vram_gib']:.2f} GiB · grad_norm 中位 {i['grad_median']:.2f}")
    if i.get("scale_note"):
        lines.append(i["scale_note"])
    if i.get("overflow_recent_single"):
        lines.append(f"近况：{i['overflow_recent_single']}")
    if i.get("overflow_total_segment"):
        lines.append(f"fp16 溢出：本运行段累计 {i['overflow_total_segment']} 次事件"
                     f"（最近 {i['overflow_last']['ts'][11:]}；均已被 GradScaler 降 scale 重试恢复，loss 轨迹无影响）")
    lines.append(f"学习率：{i['lr']:.3e}" + (f" · {i['phase']}" if i.get("phase") else ""))
    if i.get("saves_last"):
        ck = i["saves_last"]
        lines.append(f"最近存点：{Path(ck['path']).name}（{ck['ts'][11:]}）")
    if i.get("val_rise_expected"):
        lines.append(f"近况：{i['val_rise_expected']}")
    if i.get("completed"):
        lines.append(f"状态：训练已完成（target reached step {i['step']:,} @ {i['completed_ts'][11:]}）"
                     "——日志停止增长属预期，非故障")
    elif i.get("eta_hours") is not None:
        lines.append(f"ETA：约 {i['eta_hours']:.1f} 小时（trainer 均速 {i['tok_s_full_avg']:.0f} tok/s，推算 {i['eta_wall']}）")
    if i.get("note_runs"):
        lines.append(i["note_runs"])
    lines.append(f"日志新鲜度：{i['log_age_s']:.0f} 秒前更新")
    if rep.warns:
        lines.append("注意：" + "；".join(rep.warns))
    if rep.alerts:
        lines.append("异常：" + "；".join(rep.alerts))
    return "\n".join(lines)


def send_feishu(cfg: dict, text: str) -> tuple[bool, str]:
    url = cfg.get("endpoint", "http://127.0.0.1:3080/api/dsh-im/delivery/messages")
    body = json.dumps({
        "botId": cfg["botId"], "targetId": cfg["targetId"],
        "text": text, "format": "plain",
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = resp.read().decode()
        return json.loads(payload).get("sent") is True, payload
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode()[:300]}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=REPO / "runs" / "jishui-200m-cuda-4b")
    ap.add_argument("--window", type=int, default=100, help="算滚动均值的步数窗口")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--feishu", action="store_true", help="推送到飞书")
    ap.add_argument("--feishu-config", type=Path, default=DEFAULT_FEISHU)
    ap.add_argument("--force", action="store_true", help="即使没有告警也推送")
    ap.add_argument("--quiet", action="store_true", help="不打印简报（只推送）")
    ap.add_argument("--brief", action="store_true",
                    help="只打印一行摘要；与 --feishu 同用即为每小时巡检的单次调用形态")
    args = ap.parse_args()

    rep = build(args.run_dir, args.window)
    if args.json:
        print(json.dumps({"ok": rep.ok, "alerts": rep.alerts, "warns": rep.warns,
                          "infos": rep.infos, "text": rep.text}, ensure_ascii=False, indent=2, default=str))
    elif args.brief:
        i = rep.infos
        if i:
            v = i.get("val_last") or {}
            d = i.get("d1h") or {}
            d_txt = ""
            if d.get("steps") is not None:
                d_txt = f" · Δ1h: +{d['steps']:,}步"
                if d.get("val") is not None:
                    d_txt += f" val{d['val']:+.4f}"
                d_txt += f" ovr+{d.get('ovr', 0)}"
            if i.get("completed"):
                eta_txt = " · ✅已完成"
            elif i.get("eta_hours") is not None:
                eta_txt = f" · ETA={i['eta_hours']:.1f}h"
            else:
                eta_txt = ""
            print(f"击水 step={i['step']:,}/{i['max_steps']:,} ({i['step_pct']:.1f}%) · "
                  f"val={v.get('loss', float('nan')):.4f} ppl={v.get('ppl', float('nan')):.1f} · "
                  f"loss{i['window']}={i['loss_mean_window']:.4f} · grad_med={i['grad_median']:.2f} · "
                  f"tok/s={i['tok_s_median']:.0f} · scale={i['scale']:g} · "
                  f"ovr={i.get('overflow_recent_n', 0)}(win)/{i.get('overflow_total_segment', 0)}(all) · "
                  f"log_age={i['log_age_s']:.0f}s · ok={rep.ok} warn={len(rep.warns)} alert={len(rep.alerts)}"
                  f"{d_txt}{eta_txt}")
        else:
            print("击水 " + ("；".join(rep.alerts) or "无数据"))
    elif not args.quiet:
        print(rep.text)

    if args.feishu:
        if not args.force and rep.ok and not rep.warns:
            print("[feishu] 无异常，按策略不推送（--force 可强制）", file=sys.stderr)
            return 0
        if not args.feishu_config.exists():
            print(f"[feishu] 缺少投递配置 {args.feishu_config}；需先保存 botId + targetId", file=sys.stderr)
            return 2
        cfg = json.loads(args.feishu_config.read_text())
        ok, detail = send_feishu(cfg, rep.text)
        print(f"[feishu] {'已接受' if ok else '失败'}：{detail}", file=sys.stderr)
        return 0 if ok else 1
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
