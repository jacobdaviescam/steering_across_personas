"""Load pre-computed assistant axis vectors from HuggingFace.

Reference vectors from lu-christina/assistant-axis-vectors (Lu et al., 2026)
trained on Gemma 2 27B-it. These are mean activations (not contrastive steering
vectors), but useful for validating directions and projecting onto the axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from huggingface_hub import hf_hub_download

REPO_ID = "lu-christina/assistant-axis-vectors"

# Map our trait names to the filenames used in the HF repo.
# Not all traits have direct equivalents in the reference dataset.
TRAIT_NAME_MAP: dict[str, str] = {
    "assertiveness": "assertive",
    "empathy": "empathetic",
    "confidence": "confident",
    "warmth": "warm",
    "impulsivity": "impulsive",
    "deference": "deferential",
    "honesty": "transparent",
    # risk_taking has no direct reference equivalent
}


def _download(filename: str, subfolder: str | None = None) -> Path:
    """Download a file from the HF repo, returning the cached local path."""
    return Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        subfolder=subfolder,
    ))


def load_assistant_axis(device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """Load the assistant axis direction. Shape: ``(num_layers, hidden_dim)``."""
    path = _download("assistant_axis.pt")
    return torch.load(path, map_location=device, weights_only=True)


def load_default_vector(device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """Load the mean assistant activation. Shape: ``(num_layers, hidden_dim)``."""
    path = _download("default_vector.pt")
    return torch.load(path, map_location=device, weights_only=True)


def load_role_vector(
    name: str, device: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    """Load a per-role mean activation vector.

    Parameters
    ----------
    name:
        Role name (without .pt extension), e.g. ``"hamlet"``.
    """
    path = _download(f"{name}.pt", subfolder="role_vectors")
    return torch.load(path, map_location=device, weights_only=True)


def load_trait_vector(
    name: str, device: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    """Load a per-trait mean activation vector.

    Parameters
    ----------
    name:
        Trait name as used in the HF repo (e.g. ``"sycophantic"``),
        or one of our trait names (e.g. ``"sycophancy"``), which will
        be mapped automatically via :data:`TRAIT_NAME_MAP`.
    """
    mapped = TRAIT_NAME_MAP.get(name, name)
    path = _download(f"{mapped}.pt", subfolder="trait_vectors")
    return torch.load(path, map_location=device, weights_only=True)


def project_onto_axis(
    activation: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
) -> float:
    """Project an activation vector onto the assistant axis at a given layer.

    Parameters
    ----------
    activation:
        Vector to project. Shape ``(hidden_dim,)``.
    axis:
        Full axis tensor. Shape ``(num_layers, hidden_dim)``.
    layer:
        Layer index to select from ``axis``.

    Returns
    -------
    float
        Scalar projection (dot product with unit axis direction).
    """
    axis_dir = axis[layer]
    axis_unit = axis_dir / axis_dir.norm()
    return float(activation @ axis_unit)
