"""Comparison metrics: cosine similarity, transfer matrices, decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from persona_steering.config import Trait
from persona_steering.extraction import SteeringVector
from persona_steering.utils import cosine_similarity, log


# ---------------------------------------------------------------------------
# Pairwise vector comparison
# ---------------------------------------------------------------------------

@dataclass
class VectorComparison:
    """Result of comparing two steering vectors."""
    cosine_sim: float
    magnitude_ratio: float  # |v2| / |v1|
    orthogonal_residual: float  # magnitude of component of v2 orthogonal to v1
    shared_component: torch.Tensor | None = None
    persona_a: str = ""
    persona_b: str = ""
    trait: str = ""
    layer: int = -1


def compare_vectors(v1: SteeringVector, v2: SteeringVector) -> VectorComparison:
    """Compare two steering vectors.

    Computes cosine similarity, magnitude ratio, and the size of the
    component of v2 that is orthogonal to v1.
    """
    a = v1.vector.float()
    b = v2.vector.float()

    cos_sim = cosine_similarity(a, b)

    mag_ratio = b.norm().item() / (a.norm().item() + 1e-10)

    # Project b onto a, compute residual
    a_unit = a / a.norm()
    proj = torch.dot(b, a_unit) * a_unit
    residual = b - proj
    orth_mag = residual.norm().item()

    return VectorComparison(
        cosine_sim=cos_sim,
        magnitude_ratio=mag_ratio,
        orthogonal_residual=orth_mag,
        shared_component=proj,
        persona_a=v1.persona,
        persona_b=v2.persona,
        trait=v1.trait.value,
        layer=v1.layer,
    )


# ---------------------------------------------------------------------------
# Transfer matrix
# ---------------------------------------------------------------------------

def build_transfer_matrix(
    vectors: dict[str, dict[Trait, dict[int, SteeringVector]]],
    personas: list[str],
    traits: list[Trait],
    layer: int,
) -> np.ndarray:
    """Build a cosine similarity transfer matrix across personas.

    For each pair of personas, computes the average cosine similarity
    of their steering vectors across traits.

    Args:
        vectors: Nested dict from extract_all() (persona -> trait -> layer -> vec).
        personas: List of persona slugs.
        traits: List of traits to include.
        layer: Which layer to compare.

    Returns:
        np.ndarray of shape (n_personas, n_personas) with pairwise sims.
    """
    n = len(personas)
    matrix = np.zeros((n, n))

    for i, pa in enumerate(personas):
        for j, pb in enumerate(personas):
            sims = []
            for trait in traits:
                va = vectors.get(pa, {}).get(trait, {}).get(layer)
                vb = vectors.get(pb, {}).get(trait, {}).get(layer)
                if va is not None and vb is not None:
                    sims.append(cosine_similarity(va.vector, vb.vector))
            matrix[i, j] = np.mean(sims) if sims else 0.0

    log.info("Transfer matrix (%d personas, %d traits, layer %d)", n, len(traits), layer)
    return matrix


def build_per_trait_transfer(
    vectors: dict[str, dict[Trait, dict[int, SteeringVector]]],
    personas: list[str],
    trait: Trait,
    layer: int,
) -> np.ndarray:
    """Build a per-trait cosine similarity matrix across personas.

    Returns:
        np.ndarray of shape (n_personas, n_personas).
    """
    n = len(personas)
    matrix = np.zeros((n, n))

    for i, pa in enumerate(personas):
        for j, pb in enumerate(personas):
            va = vectors.get(pa, {}).get(trait, {}).get(layer)
            vb = vectors.get(pb, {}).get(trait, {}).get(layer)
            if va is not None and vb is not None:
                matrix[i, j] = cosine_similarity(va.vector, vb.vector)

    return matrix


# ---------------------------------------------------------------------------
# Curvature / decay characterisation
# ---------------------------------------------------------------------------

def compute_curvature(
    transfer_matrix: np.ndarray,
    persona_distances: np.ndarray | None = None,
) -> dict:
    """Characterise how steering transfer decays with persona distance.

    Args:
        transfer_matrix: Cosine similarity matrix (n x n).
        persona_distances: Optional pairwise distances between personas.
            If None, uses 1 - transfer_matrix as proxy.

    Returns:
        Dict with decay statistics.
    """
    n = transfer_matrix.shape[0]
    if persona_distances is None:
        persona_distances = 1.0 - transfer_matrix

    # Off-diagonal entries
    mask = ~np.eye(n, dtype=bool)
    off_diag_sims = transfer_matrix[mask]
    off_diag_dists = persona_distances[mask]

    # Simple linear fit: sim = a * dist + b
    if len(off_diag_dists) >= 2:
        coeffs = np.polyfit(off_diag_dists, off_diag_sims, 1)
        slope, intercept = coeffs
    else:
        slope, intercept = 0.0, 0.0

    return {
        "mean_off_diagonal_sim": float(np.mean(off_diag_sims)) if len(off_diag_sims) > 0 else 0.0,
        "std_off_diagonal_sim": float(np.std(off_diag_sims)) if len(off_diag_sims) > 0 else 0.0,
        "min_sim": float(np.min(off_diag_sims)) if len(off_diag_sims) > 0 else 0.0,
        "max_sim": float(np.max(off_diag_sims)) if len(off_diag_sims) > 0 else 0.0,
        "decay_slope": float(slope),
        "decay_intercept": float(intercept),
        "diagonal_mean": float(np.mean(np.diag(transfer_matrix))),
    }


# ---------------------------------------------------------------------------
# Shared vs persona-specific decomposition
# ---------------------------------------------------------------------------

@dataclass
class SharedSpecificDecomposition:
    """Decomposition of steering vectors into shared and persona-specific components."""
    shared_direction: torch.Tensor  # unit vector for shared component
    shared_magnitudes: dict[str, float]  # persona -> magnitude along shared
    specific_vectors: dict[str, torch.Tensor]  # persona -> residual vector
    specific_magnitudes: dict[str, float]
    variance_explained: float  # fraction of total variance in shared direction


def decompose_shared_specific(
    vectors: dict[str, SteeringVector],
) -> SharedSpecificDecomposition:
    """Decompose steering vectors into shared and persona-specific components.

    Takes vectors for the same trait across personas. Computes the mean
    direction as the shared component, and persona-specific residuals.

    Args:
        vectors: Dict mapping persona slug to SteeringVector.

    Returns:
        SharedSpecificDecomposition.
    """
    slugs = list(vectors.keys())
    vecs = torch.stack([vectors[s].vector.float() for s in slugs])

    # Shared direction: mean of unit vectors
    unit_vecs = vecs / vecs.norm(dim=1, keepdim=True)
    mean_dir = unit_vecs.mean(dim=0)
    shared_unit = mean_dir / mean_dir.norm()

    shared_mags = {}
    specific_vecs = {}
    specific_mags = {}

    for i, slug in enumerate(slugs):
        v = vecs[i]
        proj_mag = torch.dot(v, shared_unit).item()
        shared_mags[slug] = proj_mag
        residual = v - proj_mag * shared_unit
        specific_vecs[slug] = residual
        specific_mags[slug] = residual.norm().item()

    # Variance explained by shared direction
    total_var = vecs.var(dim=0).sum().item()
    proj_vals = torch.tensor([shared_mags[s] for s in slugs])
    shared_var = proj_vals.var().item() if len(slugs) > 1 else 0.0
    # Fraction of vector magnitude in shared direction
    total_mag_sq = sum(vecs[i].norm().item() ** 2 for i in range(len(slugs)))
    shared_mag_sq = sum(shared_mags[s] ** 2 for s in slugs)
    variance_explained = shared_mag_sq / (total_mag_sq + 1e-10)

    return SharedSpecificDecomposition(
        shared_direction=shared_unit,
        shared_magnitudes=shared_mags,
        specific_vectors=specific_vecs,
        specific_magnitudes=specific_mags,
        variance_explained=variance_explained,
    )


# ---------------------------------------------------------------------------
# Steering vs inter-persona direction (Step 4)
# ---------------------------------------------------------------------------

def compare_steering_vs_interpersona(
    steering_vec: SteeringVector,
    persona_axis: torch.Tensor,
) -> dict:
    """Compare a steering direction to the inter-persona axis.

    The inter-persona axis is the direction between two persona's mean
    activations. This tests whether steering is aligned with, orthogonal
    to, or opposed to the persona difference.

    Args:
        steering_vec: The steering vector for a trait.
        persona_axis: Direction between personas in activation space.

    Returns:
        Dict with alignment metrics.
    """
    s = steering_vec.vector.float()
    p = persona_axis.float()

    cos = cosine_similarity(s, p)

    s_unit = s / s.norm()
    p_unit = p / p.norm()

    # Project steering onto persona axis
    proj_mag = torch.dot(s, p_unit).item()
    # Orthogonal component
    orth = s - proj_mag * p_unit
    orth_mag = orth.norm().item()

    return {
        "cosine_similarity": cos,
        "projection_onto_persona_axis": proj_mag,
        "orthogonal_magnitude": orth_mag,
        "steering_magnitude": s.norm().item(),
        "persona_axis_magnitude": p.norm().item(),
        "alignment_ratio": abs(proj_mag) / (s.norm().item() + 1e-10),
    }
