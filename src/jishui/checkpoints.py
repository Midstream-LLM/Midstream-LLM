from __future__ import annotations

from pathlib import Path


def _checkpoint_directories(run_dir: Path) -> list[Path]:
    """Return complete checkpoint directories, excluding temporary saves."""
    return sorted(
        path
        for path in (run_dir / "checkpoints").glob("step_*")
        if (
            path.is_dir()
            and path.name.startswith("step_")
            and path.name[5:].isdigit()
        )
    )


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = _checkpoint_directories(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {run_dir / 'checkpoints'}")
    return checkpoints[-1]


def _remove_checkpoint_tree(path: Path) -> None:
    # Python 3.12's fd-relative shutil.rmtree can fail on this ExFAT volume.
    try:
        children = list(path.iterdir())
    except FileNotFoundError:
        return
    for child in children:
        if child.is_dir() and not child.is_symlink():
            _remove_checkpoint_tree(child)
        else:
            child.unlink(missing_ok=True)
    try:
        path.rmdir()
    except FileNotFoundError:
        pass


def prune_checkpoints(
    run_dir: Path,
    max_checkpoints: int,
    optimizer_checkpoints: int | None = None,
) -> None:
    """Keep recent model snapshots and optionally trim old Adam state files.

    ``optimizer_checkpoints=None`` preserves the legacy behavior. When set,
    only that many of the retained newest checkpoints keep ``optimizer.npz``;
    the latest checkpoint is therefore an exact-resume point when the value is
    at least one, while older snapshots remain loadable with a fresh optimizer.
    """
    if max_checkpoints < 1:
        raise ValueError("max_checkpoints must be positive")
    if optimizer_checkpoints is not None and not 0 <= optimizer_checkpoints <= max_checkpoints:
        raise ValueError("optimizer_checkpoints must be between 0 and max_checkpoints")
    checkpoints = _checkpoint_directories(run_dir)
    for checkpoint in checkpoints[:-max_checkpoints]:
        _remove_checkpoint_tree(checkpoint)
    if optimizer_checkpoints is not None:
        retained = checkpoints[-max_checkpoints:]
        without_optimizer = retained[:-optimizer_checkpoints] if optimizer_checkpoints else retained
        for checkpoint in without_optimizer:
            (checkpoint / "optimizer.npz").unlink(missing_ok=True)
