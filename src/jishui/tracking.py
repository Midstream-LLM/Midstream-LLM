"""W&B experiment metrics; model/data payloads remain in local checkpoints."""
from __future__ import annotations

import json
import logging
import math
import uuid
from pathlib import Path

LOG = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, run_dir: Path, settings: dict | None = None, mode: str | None = None):
        self.run_dir = Path(run_dir)
        self.settings = dict(settings or {})
        self.mode = mode or self.settings.get("mode", "disabled")
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("tracking mode must be online, offline or disabled")
        self.run = None
        self.identity = None
        self.attempt = 0

    def start(self, config: dict, step: int, tokens: int, parent: dict | None = None) -> None:
        if self.mode == "disabled":
            return
        import wandb

        entity, project = self.settings.get("entity"), self.settings.get("project")
        if not entity or not project:
            raise ValueError("W&B tracking requires explicit entity and project")
        identity_file = self.run_dir / "wandb_run.json"
        if identity_file.exists():
            saved = json.loads(identity_file.read_text())
            if saved["entity"] != entity or saved["project"] != project:
                raise ValueError("W&B destination changed for this run directory")
        else:
            saved = {"entity": entity, "project": project, "online_id": uuid.uuid4().hex[:12],
                     "group": self.run_dir.name + "-" + uuid.uuid4().hex[:8], "attempt": 0}
        self.attempt = saved["attempt"] + 1
        # W&B cannot resume offline runs. Give each offline session a fresh ID
        # within the same experiment group, so a later sync cannot overwrite it.
        run_id = saved["online_id"] if self.mode == "online" else uuid.uuid4().hex[:12]
        self.run = wandb.init(
            entity=entity, project=project, id=run_id,
            name=self.settings.get("name", self.run_dir.name), group=saved["group"],
            job_type="pretraining", tags=self.settings.get("tags", ["200m", "cuda", "4b", "cosine-restarts"]),
            dir=str(self.run_dir.resolve()), mode=self.mode,
            resume="allow" if self.mode == "online" else None,
            config=config, force=self.mode == "online", save_code=False,
            settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                    init_timeout=45, x_stats_sampling_interval=30),
        )
        saved["attempt"] = self.attempt
        saved["last_run_id"] = run_id
        saved["last_mode"] = self.mode
        saved["url"] = self.run.url if self.mode == "online" else None
        temporary = identity_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, indent=2))
        temporary.replace(identity_file)
        self.identity = {"entity": entity, "project": project, "run_id": run_id,
                         "group": saved["group"], "mode": self.mode, "url": saved["url"]}
        self.run.define_metric("progress/optimizer_step")
        self.run.define_metric("progress/tokens_seen")
        for prefix in ("train/*", "val/*", "perf/*"):
            self.run.define_metric(prefix, step_metric="progress/tokens_seen")
        for prefix in ("optimizer/*", "checkpoint/*", "resume/*"):
            self.run.define_metric(prefix, step_metric="progress/optimizer_step")
        self.run.summary.update({"status": "running", "resume_step": step,
                                 "resume_tokens": tokens, "resume_attempt": self.attempt})
        if parent and parent.get("run_id") != run_id:
            self.run.summary["parent_tracking"] = parent
        self.log({"resume/from_step": step, "resume/attempt": self.attempt}, step, tokens)
        LOG.info("W&B mode=%s run=%s", self.mode, saved["url"] or run_id)

    def log(self, metrics: dict, step: int, tokens: int) -> None:
        if self.run is None:
            return
        # W&B's own _step stays monotonic across log calls and resumes. The
        # checkpoint's optimizer step may rewind after a crash; preserve those
        # replayed samples with an attempt marker rather than silently drop them.
        self.run.log(dict(metrics, **{"progress/optimizer_step": step,
            "progress/tokens_seen": tokens, "resume/attempt": self.attempt}))

    def validation(self, report: dict, tokens: int) -> None:
        metrics = {"val/loss": report["loss"], "val/perplexity": math.exp(min(report["loss"], 20))}
        for category, loss in report["per_category_loss"].items():
            metrics[f"val/c{category}_loss"] = loss
            metrics[f"val/c{category}_perplexity"] = math.exp(min(loss, 20))
        for source, loss in report.get("per_source_loss", {}).items():
            metrics[f"val/source/{source}_loss"] = loss
        self.log(metrics, report["step"], tokens)

    def checkpoint(self, path: Path, step: int, tokens: int) -> None:
        if self.run is None:
            return
        self.run.summary.update({"last_checkpoint": str(path), "checkpoint_step": step,
                                 "checkpoint_tokens": tokens})
        self.log({"checkpoint/saved_step": step}, step, tokens)

    def finish(self, status: str, exit_code: int = 0) -> None:
        if self.run is None:
            return
        self.run.summary["status"] = status
        self.run.finish(exit_code=exit_code)
        self.run = None
