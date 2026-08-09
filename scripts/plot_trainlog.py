#!/usr/bin/env python3
"""Parse an MLX training log and render a dependency-free SVG dashboard."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


TIMESTAMP = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
TRAIN_RE = re.compile(
    rf"^{TIMESTAMP} .*: step=(?P<step>\d+) "
    r"loss=(?P<loss>[-+0-9.eE]+) ppl=(?P<ppl>[-+0-9.eE]+) "
    r"lr=(?P<lr>[-+0-9.eE]+) grad_norm=(?P<grad>[-+0-9.eE]+) "
    r"tokens=(?P<tokens>\d+) tok/s=(?P<tok_s>[-+0-9.eE]+)$"
)
VALIDATION_RE = re.compile(
    rf"^{TIMESTAMP} .*: validation step=(?P<step>\d+) "
    r"loss=(?P<loss>[-+0-9.eE]+) ppl=(?P<ppl>[-+0-9.eE]+)$"
)
ANY_TIMESTAMP_RE = re.compile(rf"^{TIMESTAMP}")


@dataclass(frozen=True)
class TrainPoint:
    timestamp: datetime
    step: int
    loss: float
    ppl: float
    learning_rate: float
    grad_norm: float
    tokens: int
    tok_s: float


@dataclass(frozen=True)
class ValidationPoint:
    timestamp: datetime
    step: int
    loss: float
    ppl: float


@dataclass(frozen=True)
class RunMetadata:
    target_steps: int
    target_tokens: int
    checkpoint_step: int
    checkpoint_tokens: int
    checkpoint_timestamp: datetime | None


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f")


def parse_log(path: Path) -> tuple[list[TrainPoint], list[ValidationPoint], datetime, list[str]]:
    train_by_step: dict[int, TrainPoint] = {}
    validation_by_step: dict[int, ValidationPoint] = {}
    first_timestamp: datetime | None = None
    malformed: list[str] = []

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            timestamp_match = ANY_TIMESTAMP_RE.match(line)
            if timestamp_match:
                timestamp = parse_timestamp(timestamp_match.group("timestamp"))
                first_timestamp = min(first_timestamp, timestamp) if first_timestamp else timestamp

            match = TRAIN_RE.match(line)
            if match:
                point = TrainPoint(
                    timestamp=parse_timestamp(match.group("timestamp")),
                    step=int(match.group("step")),
                    loss=float(match.group("loss")),
                    ppl=float(match.group("ppl")),
                    learning_rate=float(match.group("lr")),
                    grad_norm=float(match.group("grad")),
                    tokens=int(match.group("tokens")),
                    tok_s=float(match.group("tok_s")),
                )
                train_by_step[point.step] = point
                continue

            match = VALIDATION_RE.match(line)
            if match:
                point = ValidationPoint(
                    timestamp=parse_timestamp(match.group("timestamp")),
                    step=int(match.group("step")),
                    loss=float(match.group("loss")),
                    ppl=float(match.group("ppl")),
                )
                validation_by_step[point.step] = point
                continue

            if line and not timestamp_match:
                malformed.append(f"line {line_number}: {line[:120]}")

    if not train_by_step or first_timestamp is None:
        raise ValueError(f"no training records found in {path}")
    return (
        [train_by_step[step] for step in sorted(train_by_step)],
        [validation_by_step[step] for step in sorted(validation_by_step)],
        first_timestamp,
        malformed,
    )


def load_run_metadata(run_dir: Path, latest_train: TrainPoint) -> RunMetadata:
    target_steps = latest_train.step
    target_tokens = latest_train.tokens
    config_path = run_dir / "resolved_config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)["training"]
        target_steps = int(config["max_steps"])
        tokens_per_step = (
            int(config["seq_len"])
            * int(config["batch_size"])
            * int(config["grad_accum_steps"])
        )
        target_tokens = target_steps * tokens_per_step

    checkpoint_step = latest_train.step
    checkpoint_tokens = latest_train.tokens
    checkpoint_timestamp: datetime | None = latest_train.timestamp
    latest_path = run_dir / "latest.json"
    if latest_path.exists():
        with latest_path.open(encoding="utf-8") as handle:
            latest = json.load(handle)
        checkpoint = Path(latest["checkpoint"])
        if not checkpoint.is_absolute():
            repo_relative = run_dir.parents[1] / checkpoint
            run_relative = run_dir / checkpoint
            checkpoint = repo_relative if repo_relative.exists() else run_relative
        state_path = checkpoint / "train_state.json"
        if state_path.exists():
            with state_path.open(encoding="utf-8") as handle:
                state = json.load(handle)
            checkpoint_step = int(state["step"])
            checkpoint_tokens = int(state["tokens_seen"])
            checkpoint_timestamp = datetime.fromtimestamp(state_path.stat().st_mtime)
    return RunMetadata(
        target_steps=target_steps,
        target_tokens=target_tokens,
        checkpoint_step=checkpoint_step,
        checkpoint_tokens=checkpoint_tokens,
        checkpoint_timestamp=checkpoint_timestamp,
    )


def rolling_mean(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def format_count(value: float) -> str:
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= divisor:
            return f"{value / divisor:.2f}{suffix}"
    return f"{value:.0f}"


def format_duration(seconds: float) -> str:
    days = seconds / 86400
    if days >= 1:
        return f"{days:.1f} days"
    hours = seconds / 3600
    if hours >= 1:
        return f"{hours:.1f} hours"
    return f"{seconds / 60:.0f} min"


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            "<style>",
            "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172033}",
            ".title{font-size:31px;font-weight:700}.subtitle{font-size:16px;fill:#526075}",
            ".metric{font-size:21px;font-weight:650}.label{font-size:14px;fill:#667085}",
            ".panel-title{font-size:19px;font-weight:650}.tick{font-size:12px;fill:#667085}",
            ".legend{font-size:13px;fill:#3d485c}.note{font-size:13px;fill:#8a3d22}",
            "</style>",
            '<rect width="100%" height="100%" fill="#f5f7fa"/>',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def rect(self, x: float, y: float, width: float, height: float, **attrs: object) -> None:
        values = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" {values}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, **attrs: object) -> None:
        values = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {values}/>')

    def text(self, x: float, y: float, value: str, css_class: str = "", **attrs: object) -> None:
        values = " ".join(f'{key.replace("_", "-")}="{item}"' for key, item in attrs.items())
        class_attr = f' class="{css_class}"' if css_class else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}"{class_attr} {values}>{html.escape(value)}</text>')

    def polyline(self, points: list[tuple[float, float]], **attrs: object) -> None:
        if not points:
            return
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        values = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
        self.add(f'<polyline points="{coords}" {values}/>')

    def circle(self, x: float, y: float, radius: float, **attrs: object) -> None:
        values = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" {values}/>')

    def finish(self) -> str:
        return "\n".join([*self.parts, "</svg>"])


@dataclass(frozen=True)
class Panel:
    x: float
    y: float
    width: float
    height: float

    @property
    def plot_x(self) -> float:
        return self.x + 66

    @property
    def plot_y(self) -> float:
        return self.y + 54

    @property
    def plot_width(self) -> float:
        return self.width - 96

    @property
    def plot_height(self) -> float:
        return self.height - 100


def draw_panel(svg: Svg, panel: Panel, title: str) -> None:
    svg.rect(panel.x, panel.y, panel.width, panel.height, rx=7, fill="#ffffff", stroke="#d9dee8")
    svg.text(panel.x + 22, panel.y + 32, title, "panel-title")


def draw_axes(
    svg: Svg,
    panel: Panel,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    y_formatter=lambda value: f"{value:.1f}",
) -> tuple[object, object]:
    def sx(value: float) -> float:
        return panel.plot_x + (value - x_min) / max(x_max - x_min, 1e-12) * panel.plot_width

    def sy(value: float) -> float:
        return panel.plot_y + panel.plot_height - (value - y_min) / max(y_max - y_min, 1e-12) * panel.plot_height

    for index in range(6):
        fraction = index / 5
        x = panel.plot_x + fraction * panel.plot_width
        step = x_min + fraction * (x_max - x_min)
        svg.line(x, panel.plot_y, x, panel.plot_y + panel.plot_height, stroke="#edf0f5", stroke_width=1)
        svg.text(x, panel.plot_y + panel.plot_height + 25, f"{step:,.0f}", "tick", text_anchor="middle")
    for index in range(5):
        fraction = index / 4
        y = panel.plot_y + fraction * panel.plot_height
        value = y_max - fraction * (y_max - y_min)
        svg.line(panel.plot_x, y, panel.plot_x + panel.plot_width, y, stroke="#edf0f5", stroke_width=1)
        svg.text(panel.plot_x - 10, y + 4, y_formatter(value), "tick", text_anchor="end")
    svg.line(panel.plot_x, panel.plot_y + panel.plot_height, panel.plot_x + panel.plot_width, panel.plot_y + panel.plot_height, stroke="#aab3c2")
    return sx, sy


def downsample(points: list[tuple[float, float]], limit: int = 5000) -> list[tuple[float, float]]:
    if len(points) <= limit:
        return points
    stride = math.ceil(len(points) / limit)
    return points[::stride]


def draw_dashboard(
    output_path: Path,
    log_path: Path,
    train: list[TrainPoint],
    validation: list[ValidationPoint],
    run_start: datetime,
    malformed: list[str],
    metadata: RunMetadata,
    baseline_tok_s: float | None,
) -> dict[str, float | int | str]:
    steps = [point.step for point in train]
    losses = [point.loss for point in train]
    throughputs = [point.tok_s for point in train]
    gradients = [point.grad_norm for point in train]
    learning_rates = [point.learning_rate for point in train]
    loss_ma = rolling_mean(losses, 50)
    loss_ma100 = rolling_mean(losses, 100)
    throughput_ma = rolling_mean(throughputs, 50)
    grad_ma = rolling_mean(gradients, 50)

    latest_train = train[-1]
    progress_tokens = max(latest_train.tokens, metadata.checkpoint_tokens)
    progress_step = max(latest_train.step, metadata.checkpoint_step)
    progress = progress_step / metadata.target_steps if metadata.target_steps else 0.0
    effective_end = metadata.checkpoint_timestamp or latest_train.timestamp
    wall_seconds = max((effective_end - run_start).total_seconds(), 1.0)
    wall_tok_s = progress_tokens / wall_seconds
    remaining_seconds = max(metadata.target_tokens - progress_tokens, 0) / wall_tok_s
    latest_validation = validation[-1] if validation else None

    summary: dict[str, float | int | str] = {
        "train_points": len(train),
        "latest_log_step": latest_train.step,
        "latest_checkpoint_step": metadata.checkpoint_step,
        "progress_percent": 100 * progress,
        "tokens_seen": progress_tokens,
        "train_loss_ma100": loss_ma100[-1],
        "reported_tok_s_mean": statistics.mean(throughputs),
        "reported_tok_s_median": statistics.median(throughputs),
        "reported_tok_s_recent100": statistics.mean(throughputs[-100:]),
        "wall_tok_s": wall_tok_s,
        "eta_seconds": remaining_seconds,
        "grad_clip_fraction": sum(value > 1.0 for value in gradients) / len(gradients),
        "grad_clip_fraction_recent500": sum(value > 1.0 for value in gradients[-500:]) / min(500, len(gradients)),
        "grad_norm_max": max(gradients),
        "malformed_lines": len(malformed),
    }
    if latest_validation:
        summary["validation_loss"] = latest_validation.loss
        summary["validation_ppl"] = latest_validation.ppl

    svg = Svg(1800, 1080)
    svg.text(72, 53, "Jishui 200M Stage 0 - Training Snapshot", "title")
    svg.text(72, 82, f"Source: {log_path}", "subtitle")

    metric_x = [72, 390, 720, 1050, 1390]
    metric_values = [
        (f"{progress_step:,} / {metadata.target_steps:,}", "optimizer steps"),
        (f"{format_count(progress_tokens)} / {format_count(metadata.target_tokens)}", "tokens"),
        (f"{loss_ma100[-1]:.3f}", "train loss, trailing 100"),
        (
            f"{latest_validation.loss:.3f}" if latest_validation else "n/a",
            "latest validation loss",
        ),
        (f"{wall_tok_s:,.0f} tok/s", "wall-clock throughput"),
    ]
    for x, (value, label) in zip(metric_x, metric_values):
        svg.text(x, 122, value, "metric")
        svg.text(x, 147, label, "label")

    if metadata.checkpoint_step > latest_train.step or malformed:
        note = (
            f"Log ends at step {latest_train.step:,}; checkpoint is step {metadata.checkpoint_step:,}. "
            f"Ignored malformed lines: {len(malformed)}."
        )
        svg.text(72, 174, note, "note")

    loss_panel = Panel(60, 195, 1680, 405)
    draw_panel(svg, loss_panel, "Cross-entropy loss")
    loss_min = max(0.0, min([*loss_ma, *[point.loss for point in validation]]) - 0.35)
    loss_max = max(losses) + 0.25
    sx, sy = draw_axes(svg, loss_panel, 0, max(steps), loss_min, loss_max)
    raw_loss_points = downsample([(sx(point.step), sy(point.loss)) for point in train])
    smooth_loss_points = [(sx(step), sy(value)) for step, value in zip(steps, loss_ma)]
    svg.polyline(raw_loss_points, fill="none", stroke="#8fb7e8", stroke_width=1, opacity=0.34)
    svg.polyline(smooth_loss_points, fill="none", stroke="#1769aa", stroke_width=3)
    if validation:
        validation_points = [(sx(point.step), sy(point.loss)) for point in validation]
        svg.polyline(validation_points, fill="none", stroke="#d45b27", stroke_width=2.5)
        for x, y in validation_points:
            svg.circle(x, y, 4.5, fill="#d45b27", stroke="#ffffff", stroke_width=1.5)
    legend_y = loss_panel.y + 31
    svg.line(1130, legend_y - 5, 1162, legend_y - 5, stroke="#8fb7e8", stroke_width=2)
    svg.text(1170, legend_y, "train raw", "legend")
    svg.line(1270, legend_y - 5, 1302, legend_y - 5, stroke="#1769aa", stroke_width=3)
    svg.text(1310, legend_y, "train MA50", "legend")
    svg.line(1430, legend_y - 5, 1462, legend_y - 5, stroke="#d45b27", stroke_width=3)
    svg.text(1470, legend_y, "validation", "legend")

    throughput_panel = Panel(60, 630, 820, 390)
    draw_panel(svg, throughput_panel, "Throughput and thermal behavior")
    throughput_max = max(max(throughputs), baseline_tok_s or 0, percentile(throughputs, 0.99)) * 1.08
    throughput_min = max(0.0, min(throughputs) * 0.85)
    tx, ty = draw_axes(
        svg,
        throughput_panel,
        0,
        max(steps),
        throughput_min,
        throughput_max,
        y_formatter=lambda value: f"{value:.0f}",
    )
    svg.polyline(
        downsample([(tx(point.step), ty(point.tok_s)) for point in train]),
        fill="none",
        stroke="#8bc8ad",
        stroke_width=1,
        opacity=0.42,
    )
    svg.polyline(
        [(tx(step), ty(value)) for step, value in zip(steps, throughput_ma)],
        fill="none",
        stroke="#147d64",
        stroke_width=3,
    )
    svg.line(
        throughput_panel.plot_x,
        ty(wall_tok_s),
        throughput_panel.plot_x + throughput_panel.plot_width,
        ty(wall_tok_s),
        stroke="#4b5565",
        stroke_width=1.5,
        stroke_dasharray="7 6",
    )
    if baseline_tok_s:
        svg.line(
            throughput_panel.plot_x,
            ty(baseline_tok_s),
            throughput_panel.plot_x + throughput_panel.plot_width,
            ty(baseline_tok_s),
            stroke="#b45309",
            stroke_width=1.5,
            stroke_dasharray="5 6",
        )
        svg.text(
            throughput_panel.plot_x + throughput_panel.plot_width - 6,
            ty(baseline_tok_s) - 7,
            f"short benchmark {baseline_tok_s:.0f}",
            "tick",
            text_anchor="end",
        )
    svg.text(
        throughput_panel.plot_x + throughput_panel.plot_width - 6,
        ty(wall_tok_s) + 17,
        f"wall average {wall_tok_s:.0f}",
        "tick",
        text_anchor="end",
    )

    gradient_panel = Panel(920, 630, 820, 390)
    draw_panel(svg, gradient_panel, "Gradient norm and learning rate")
    grad_max = max(3.0, percentile(gradients, 0.98) * 1.15)
    gx, gy = draw_axes(svg, gradient_panel, 0, max(steps), 0, grad_max)
    clipped_gradients = [min(value, grad_max) for value in gradients]
    svg.polyline(
        downsample([(gx(step), gy(value)) for step, value in zip(steps, clipped_gradients)]),
        fill="none",
        stroke="#bba3d8",
        stroke_width=1,
        opacity=0.36,
    )
    svg.polyline(
        [(gx(step), gy(min(value, grad_max))) for step, value in zip(steps, grad_ma)],
        fill="none",
        stroke="#6941a5",
        stroke_width=3,
    )
    svg.line(
        gradient_panel.plot_x,
        gy(1.0),
        gradient_panel.plot_x + gradient_panel.plot_width,
        gy(1.0),
        stroke="#c2410c",
        stroke_width=1.5,
        stroke_dasharray="7 6",
    )
    max_lr = max(learning_rates)
    lr_points = [
        (
            gx(step),
            gradient_panel.plot_y + gradient_panel.plot_height
            - value / max_lr * gradient_panel.plot_height,
        )
        for step, value in zip(steps, learning_rates)
    ]
    svg.polyline(lr_points, fill="none", stroke="#e5a11a", stroke_width=2.5, opacity=0.9)
    svg.text(gradient_panel.x + 478, gradient_panel.y + 31, "grad MA50", "legend")
    svg.line(gradient_panel.x + 442, gradient_panel.y + 26, gradient_panel.x + 470, gradient_panel.y + 26, stroke="#6941a5", stroke_width=3)
    svg.text(gradient_panel.x + 620, gradient_panel.y + 31, "learning rate", "legend")
    svg.line(gradient_panel.x + 584, gradient_panel.y + 26, gradient_panel.x + 612, gradient_panel.y + 26, stroke="#e5a11a", stroke_width=3)
    svg.text(
        gradient_panel.plot_x + gradient_panel.plot_width - 5,
        gy(1.0) - 7,
        "clip threshold 1.0",
        "tick",
        text_anchor="end",
    )

    svg.text(
        60,
        1060,
        f"Progress {100 * progress:.2f}% | ETA at observed wall rate: {format_duration(remaining_seconds)} | "
        f"reported throughput median: {statistics.median(throughputs):.0f} tok/s",
        "subtitle",
    )
    output_path.write_text(svg.finish(), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-tok-s", type=float)
    args = parser.parse_args()

    train, validation, run_start, malformed = parse_log(args.log)
    metadata = load_run_metadata(args.log.parent, train[-1])
    output = args.output or args.log.with_name("trainlog-analysis.svg")
    summary = draw_dashboard(
        output,
        args.log,
        train,
        validation,
        run_start,
        malformed,
        metadata,
        args.baseline_tok_s,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"wrote {output}")
    if malformed:
        print("ignored malformed input:")
        for line in malformed:
            print(f"  {line}")


if __name__ == "__main__":
    main()
