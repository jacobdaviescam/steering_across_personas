"""Optional Weights & Biases integration for experiment tracking and artifact storage.

All public functions are no-ops if wandb is not installed or WANDB_DISABLED=true.
Import this module freely -- it never raises ImportError.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_wandb = None  # lazy import sentinel


def _get_wandb():
    """Lazy-import wandb, return None if disabled or no API key set."""
    global _wandb
    if _wandb is not None:
        return _wandb if _wandb is not False else None
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes"):
        _wandb = False
        return None
    if not os.environ.get("WANDB_API_KEY"):
        _wandb = False
        return None
    import wandb
    _wandb = wandb
    return wandb


def is_available() -> bool:
    """Check if wandb is installed and not disabled."""
    return _get_wandb() is not None


def infer_method(path: Path | str) -> str:
    """Infer extraction method ('iv' or 'caa') from a directory path.

    Any path component containing 'caa' → 'caa', otherwise → 'iv'.
    """
    return "caa" if "caa" in str(path) else "iv"


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def init_run(
    step_name: str,
    model_short: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    method: str = "iv",
) -> Any | None:
    """Initialize a W&B run for a pipeline step. Returns the run or None.

    Tags use structured ``key:value`` format for easy filtering:
      - ``model:<model_short>``
      - ``step:<step_name>``
      - ``method:<method>``  (``iv`` or ``caa``)
    """
    wandb = _get_wandb()
    if wandb is None:
        return None
    default_tags = [
        f"model:{model_short}",
        f"step:{step_name}",
        f"method:{method}",
    ]
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "persona-steering"),
        group=model_short,
        job_type=step_name,
        name=f"{model_short}/{step_name}",
        config=config or {},
        tags=tags or default_tags,
        reinit=True,
    )


def finish_run() -> None:
    """Finish the current W&B run."""
    wandb = _get_wandb()
    if wandb is None or wandb.run is None:
        return
    wandb.run.finish()


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log scalar metrics to the current W&B run."""
    wandb = _get_wandb()
    if wandb is None or wandb.run is None:
        return
    wandb.run.log(metrics, step=step)


def log_summary(metrics: dict[str, Any]) -> None:
    """Set summary metrics on the current W&B run."""
    wandb = _get_wandb()
    if wandb is None or wandb.run is None:
        return
    for k, v in metrics.items():
        wandb.run.summary[k] = v


# ---------------------------------------------------------------------------
# Image logging
# ---------------------------------------------------------------------------

def log_images(image_dir: Path, prefix: str = "") -> None:
    """Log all .png files in a directory as W&B Images."""
    wandb = _get_wandb()
    if wandb is None or wandb.run is None:
        return
    for png in sorted(image_dir.glob("*.png")):
        key = f"{prefix}/{png.stem}" if prefix else png.stem
        wandb.run.log({key: wandb.Image(str(png))})


# ---------------------------------------------------------------------------
# Artifact upload
# ---------------------------------------------------------------------------

def log_artifact(
    name: str,
    artifact_type: str,
    local_dir: Path,
    metadata: dict[str, Any] | None = None,
    glob_pattern: str = "*",
) -> None:
    """Upload a local directory as a W&B artifact.

    Disabled by default to avoid large storage costs.  Enable with
    ``WANDB_UPLOAD_ARTIFACTS=true`` in the environment or ``.env``.
    """
    if not os.environ.get("WANDB_UPLOAD_ARTIFACTS", "").lower() in ("true", "1", "yes"):
        return
    wandb = _get_wandb()
    if wandb is None or wandb.run is None:
        return
    art = wandb.Artifact(name, type=artifact_type, metadata=metadata or {})
    for f in sorted(local_dir.glob(glob_pattern)):
        if f.is_file():
            art.add_file(str(f), name=f.name)
    wandb.run.log_artifact(art)


# ---------------------------------------------------------------------------
# Artifact download (for consuming steps)
# ---------------------------------------------------------------------------

def ensure_dir(
    artifact_name: str,
    local_path: Path,
    expected_glob: str = "*",
) -> Path:
    """Return local_path if it has files, otherwise download from W&B.

    Preserves backward compatibility: if the directory already exists locally
    with matching files, W&B is never contacted. If the directory is
    empty/missing and W&B is available, the artifact is downloaded.

    Returns the path to use (either local_path or the downloaded cache dir).
    """
    if local_path.exists() and any(local_path.glob(expected_glob)):
        return local_path

    wandb = _get_wandb()
    if wandb is None:
        return local_path

    try:
        api = wandb.Api()
        project = os.environ.get("WANDB_PROJECT", "persona-steering")
        entity = os.environ.get("WANDB_ENTITY", "")
        full_name = (
            f"{entity}/{project}/{artifact_name}:latest"
            if entity
            else f"{project}/{artifact_name}:latest"
        )
        artifact = api.artifact(full_name)
        return Path(artifact.download())
    except Exception:
        return local_path
