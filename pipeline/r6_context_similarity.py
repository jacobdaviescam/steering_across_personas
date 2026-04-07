#!/usr/bin/env python3
"""Pairwise context (persona) similarity analysis for steering vectors.

For each trait, computes the N x N cosine similarity matrix between all persona
steering vectors, then analyses semantic coherence of similar pairs, cross-trait
persona profiles, and hierarchical clustering.

Usage:
    python pipeline/r6_context_similarity.py \
        --vectors-dir outputs/gemma-2-27b-it/vectors --layer 22
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from persona_steering.config import Trait, OUTPUTS_DIR, TARGET_LAYER
from persona_steering.utils import (
    log, save_json, save_fig, cosine_similarity,
    parse_persona_trait_from_stem, load_vectors,
)
from persona_steering.wandb_utils import init_run, finish_run, log_summary, log_images

# Human-labeled semantically similar persona pairs
SEMANTIC_PAIRS = [
    ("therapist", "kindergarten_teacher"),   # caring/nurturing roles
    ("con_artist", "street_hustler"),        # street-smart, deceptive
    ("drill_sergeant", "surgeon"),           # high-authority, decisive
    ("professor", "tech_ceo"),              # intellectual authority
    ("politician", "con_artist"),           # strategic/manipulative
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise context (persona) similarity analysis")
    p.add_argument("--vectors-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_similarity_matrix(
    personas: list[str],
    traits: list[str],
    vectors: dict[tuple[str, str], torch.Tensor],
    trait: str | None = None,
) -> np.ndarray:
    """Build N x N cosine similarity matrix for given trait (or mean across all traits)."""
    n = len(personas)
    if trait is not None:
        mat = np.full((n, n), np.nan)
        for i, pi in enumerate(personas):
            for j, pj in enumerate(personas):
                if (pi, trait) in vectors and (pj, trait) in vectors:
                    mat[i, j] = cosine_similarity(vectors[(pi, trait)], vectors[(pj, trait)])
        return mat
    else:
        # Mean across all traits
        per_trait = []
        for t in traits:
            m = _build_similarity_matrix(personas, traits, vectors, trait=t)
            if not np.all(np.isnan(m)):
                per_trait.append(m)
        if not per_trait:
            return np.full((n, n), np.nan)
        stacked = np.stack(per_trait)
        return np.nanmean(stacked, axis=0)


def _plot_heatmap(
    matrix: np.ndarray,
    personas: list[str],
    title: str,
    path: Path,
) -> None:
    """Plot and save a persona x persona similarity heatmap."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="equal")
    labels = [p.replace("_", " ").title() for p in personas]
    ax.set_xticks(range(len(personas)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(personas)):
        for j in range(len(personas)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if matrix[i, j] < 0.3 else "black")
    plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, path)


def main() -> None:
    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    short = vectors_dir.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "robustness" / "context_similarity"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    # Load all vectors
    vectors = load_vectors(vectors_dir, layer)

    if not vectors:
        log.error("No vectors loaded")
        return

    personas = sorted({p for p, _ in vectors})
    traits = sorted({t for _, t in vectors})
    log.info("Loaded %d vectors: %d personas, %d traits", len(vectors), len(personas), len(traits))

    init_run("r6_context_similarity", short, config=vars(args))

    # ------------------------------------------------------------------
    # 1. Per-trait similarity matrices + heatmaps
    # ------------------------------------------------------------------
    similarity_matrices: dict[str, list[list[float]]] = {}
    similarity_summary: dict[str, dict] = {}

    for trait in traits:
        mat = _build_similarity_matrix(personas, traits, vectors, trait=trait)
        similarity_matrices[trait] = [[float(x) if not np.isnan(x) else None for x in row] for row in mat]

        # Off-diagonal statistics
        n = len(personas)
        off_diag = [mat[i, j] for i in range(n) for j in range(n) if i != j and not np.isnan(mat[i, j])]
        if off_diag:
            similarity_summary[trait] = {
                "mean_off_diagonal": float(np.mean(off_diag)),
                "std_off_diagonal": float(np.std(off_diag)),
                "min_off_diagonal": float(np.min(off_diag)),
                "max_off_diagonal": float(np.max(off_diag)),
            }

        _plot_heatmap(
            mat, personas,
            title=f"Persona Similarity: {trait.replace('_', ' ').title()}",
            path=output_dir / f"similarity_heatmap_{trait}.png",
        )

    save_json(similarity_matrices, output_dir / "similarity_matrices.json")
    save_json(similarity_summary, output_dir / "similarity_summary.json")

    # ------------------------------------------------------------------
    # 2. Mean similarity heatmap (averaged across traits)
    # ------------------------------------------------------------------
    mean_mat = _build_similarity_matrix(personas, traits, vectors, trait=None)
    _plot_heatmap(
        mean_mat, personas,
        title="Mean Persona Similarity (Averaged Across Traits)",
        path=output_dir / "similarity_heatmap_mean.png",
    )

    # ------------------------------------------------------------------
    # 3. Semantic coherence test (permutation test)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    persona_idx = {p: i for i, p in enumerate(personas)}

    # Compute mean similarity for labeled pairs
    labeled_sims = []
    valid_pairs = []
    for p1, p2 in SEMANTIC_PAIRS:
        if p1 in persona_idx and p2 in persona_idx:
            i, j = persona_idx[p1], persona_idx[p2]
            val = mean_mat[i, j]
            if not np.isnan(val):
                labeled_sims.append(val)
                valid_pairs.append((p1, p2))

    all_pairs = [(personas[i], personas[j]) for i, j in combinations(range(len(personas)), 2)]
    all_pair_sims = []
    for p1, p2 in all_pairs:
        i, j = persona_idx[p1], persona_idx[p2]
        val = mean_mat[i, j]
        if not np.isnan(val):
            all_pair_sims.append(val)

    n_perms = 10000
    labeled_mean = float(np.mean(labeled_sims)) if labeled_sims else 0.0
    n_labeled = len(labeled_sims)

    count_ge = 0
    perm_means = []
    for _ in range(n_perms):
        sample = rng.choice(all_pair_sims, size=n_labeled, replace=False)
        m = float(np.mean(sample))
        perm_means.append(m)
        if m >= labeled_mean:
            count_ge += 1

    p_value = count_ge / n_perms
    random_mean = float(np.mean(all_pair_sims)) if all_pair_sims else 0.0

    semantic_coherence = {
        "labeled_pairs": [{"pair": list(pair), "similarity": float(sim)} for pair, sim in zip(valid_pairs, labeled_sims)],
        "labeled_mean_similarity": labeled_mean,
        "random_mean_similarity": random_mean,
        "n_permutations": n_perms,
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
    }
    save_json(semantic_coherence, output_dir / "semantic_coherence.json")

    # Semantic coherence figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_means, bins=50, alpha=0.7, color="#4C72B0", label="Random pair means")
    ax.axvline(labeled_mean, color="#C44E52", lw=2, ls="--", label=f"Labeled pairs (mean={labeled_mean:.3f})")
    ax.axvline(random_mean, color="gray", lw=1, ls=":", label=f"Overall mean={random_mean:.3f}")
    ax.set_xlabel("Mean Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Semantic Coherence Test (p={p_value:.4f}, n_perm={n_perms})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, output_dir / "semantic_coherence.png")

    # ------------------------------------------------------------------
    # 4. Cross-trait persona profiles
    # ------------------------------------------------------------------
    cross_trait_profiles: dict[str, dict[str, float]] = {}
    for p1, p2 in combinations(personas, 2):
        pair_key = f"{p1}_vs_{p2}"
        profile: dict[str, float] = {}
        for trait in traits:
            if (p1, trait) in vectors and (p2, trait) in vectors:
                profile[trait] = cosine_similarity(vectors[(p1, trait)], vectors[(p2, trait)])
        if profile:
            cross_trait_profiles[pair_key] = profile
    save_json(cross_trait_profiles, output_dir / "cross_trait_profiles.json")

    # ------------------------------------------------------------------
    # 5. Hierarchical clustering dendrogram
    # ------------------------------------------------------------------
    # Convert mean similarity to distance: d = 1 - sim
    dist_mat = 1.0 - mean_mat
    np.fill_diagonal(dist_mat, 0.0)
    # Ensure symmetry and clip
    dist_mat = (dist_mat + dist_mat.T) / 2.0
    dist_mat = np.clip(dist_mat, 0, None)
    # Replace any remaining NaN with max distance
    dist_mat = np.nan_to_num(dist_mat, nan=1.0)

    condensed = squareform(dist_mat)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [p.replace("_", " ").title() for p in personas]
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=10)
    ax.set_ylabel("Distance (1 - cosine similarity)")
    ax.set_title("Hierarchical Clustering of Personas (Average Linkage, Mean Across Traits)")
    fig.tight_layout()
    save_fig(fig, output_dir / "persona_dendrogram.png")

    # ------------------------------------------------------------------
    # W&B logging and summary
    # ------------------------------------------------------------------
    summary_metrics = {
        "r6/mean_off_diagonal": float(np.mean([s["mean_off_diagonal"] for s in similarity_summary.values()])),
        "r6/semantic_p_value": p_value,
        "r6/labeled_mean_sim": labeled_mean,
        "r6/random_mean_sim": random_mean,
    }
    for trait, s in similarity_summary.items():
        summary_metrics[f"r6/{trait}/mean_sim"] = s["mean_off_diagonal"]
    log_summary(summary_metrics)
    log_images(output_dir, prefix="r6_context_sim")
    finish_run()

    log.info("=== Context Similarity Summary ===")
    for trait in sorted(similarity_summary, key=lambda t: similarity_summary[t]["mean_off_diagonal"]):
        s = similarity_summary[trait]
        log.info("  %-15s: mean=%.4f +/- %.4f  [%.4f, %.4f]",
                 trait, s["mean_off_diagonal"], s["std_off_diagonal"],
                 s["min_off_diagonal"], s["max_off_diagonal"])
    log.info("Semantic coherence: labeled=%.4f vs random=%.4f (p=%.4f)",
             labeled_mean, random_mean, p_value)
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
