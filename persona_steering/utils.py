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


def save_fig(fig, path: Path, dpi: int = 200) -> None:
    """Save a matplotlib figure and close it."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", path)


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


def parse_persona_trait_from_stem(stem: str) -> tuple[str | None, str | None]:
    """Parse a '{persona}_{trait}' stem into (persona_slug, trait_value).

    Handles multi-word persona slugs (e.g. 'con_artist') by matching
    known trait suffixes from the Trait enum.

    Returns (None, None) if no known trait suffix is found.
    """
    from persona_steering.config import Trait
    trait_values = {t.value for t in Trait}
    for tv in trait_values:
        if stem.endswith(f"_{tv}"):
            persona = stem[: -(len(tv) + 1)]
            return persona, tv
    return None, None


def discover_activation_pairs(activations_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """Find matching pos/neg activation file pairs in a directory.

    Returns list of (persona, trait, pos_path, neg_path).
    """
    files: dict[tuple[str, str], dict[str, Path]] = {}

    for pt_file in sorted(activations_dir.glob("*.pt")):
        stem = pt_file.stem
        if stem.endswith("_pos"):
            direction, rest = "pos", stem[:-4]
        elif stem.endswith("_neg"):
            direction, rest = "neg", stem[:-4]
        else:
            continue

        persona, trait = parse_persona_trait_from_stem(rest)
        if persona is None:
            continue
        files.setdefault((persona, trait), {})[direction] = pt_file

    pairs = []
    for (persona, trait), directions in sorted(files.items()):
        if "pos" in directions and "neg" in directions:
            pairs.append((persona, trait, directions["pos"], directions["neg"]))
        else:
            missing = "neg" if "pos" in directions else "pos"
            log.warning("Missing %s file for %s/%s, skipping", missing, persona, trait)
    return pairs


def ensure_output_dirs() -> None:
    """Create all output subdirectories."""
    for d in [OUTPUTS_DIR, OUTPUTS_DIR / "vectors", OUTPUTS_DIR / "activations",
              OUTPUTS_DIR / "evaluations", OUTPUTS_DIR / "figures"]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Vector shim
# ---------------------------------------------------------------------------

class VectorShim:
    """Minimal stand-in for SteeringVector used by analysis functions.

    Wraps a single-layer steering vector with metadata needed by
    analysis functions like build_transfer_matrix and decompose_shared_specific.
    """

    def __init__(self, vector: torch.Tensor, persona: str, trait, layer: int):
        self.vector = vector
        self.persona = persona
        self.trait = trait
        self.layer = layer

    @property
    def magnitude(self) -> float:
        return self.vector.norm().item()
