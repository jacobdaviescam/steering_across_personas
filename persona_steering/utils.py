"""Shared utilities: device selection, caching, logging."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path

import torch

from persona_steering.config import OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger("persona_steering")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def cache_key(*args: str) -> str:
    """Generate a short hash from string arguments."""
    blob = "|".join(args).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def save_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info("Saved %s", path)


def load_pickle(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> object:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def model_short_name(model: str) -> str:
    """Extract short model name from HuggingFace model ID (e.g. 'google/gemma-2-9b-it' -> 'gemma-2-9b-it')."""
    return model.split("/")[-1]


def ensure_output_dirs() -> None:
    """Create all output subdirectories."""
    for d in [OUTPUTS_DIR, OUTPUTS_DIR / "vectors", OUTPUTS_DIR / "activations",
              OUTPUTS_DIR / "evaluations", OUTPUTS_DIR / "figures"]:
        d.mkdir(parents=True, exist_ok=True)
