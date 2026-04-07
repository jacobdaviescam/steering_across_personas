#!/usr/bin/env python3
"""Map personas in a geometric landscape derived from their steering vectors.

Four visualisations:
  1. PCA of all 80 raw vectors (coloured by persona and by trait)
  2. Persona embeddings from 8-d trait profile (projection onto shared direction)
  3. Persona-specific residual landscape (shared component removed)
  4. Assistant axis projection per persona

Optionally compares two extraction methods (IV vs CAA) side-by-side.

Usage:
    # Single method:
    python pipeline/5b_persona_landscape.py \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --analysis-dir outputs/gemma-2-27b-it/analysis_instruction_variant \
        --output-dir outputs/gemma-2-27b-it/figures_landscape

    # Compare two methods:
    python pipeline/5b_persona_landscape.py \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --analysis-dir outputs/gemma-2-27b-it/analysis_instruction_variant \
        --vectors-dir-2 outputs/gemma-2-27b-it/caa_vectors \
        --analysis-dir-2 outputs/gemma-2-27b-it/caa_analysis \
        --output-dir outputs/gemma-2-27b-it/figures_landscape

    # With assistant axis:
    python pipeline/5b_persona_landscape.py \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --analysis-dir outputs/gemma-2-27b-it/analysis_instruction_variant \
        --axis outputs/gemma-2-27b-it/axis.pt \
        --output-dir outputs/gemma-2-27b-it/figures_landscape
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from persona_steering.config import Trait, PERSONA_SLUGS
from persona_steering.utils import log, load_json, save_fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRETTY_PERSONAS = {
    "con_artist": "Con Artist",
    "drill_sergeant": "Drill Sgt",
    "farmer": "Farmer",
    "kindergarten_teacher": "K. Teacher",
    "politician": "Politician",
    "professor": "Professor",
    "street_hustler": "St. Hustler",
    "surgeon": "Surgeon",
    "tech_ceo": "Tech CEO",
    "therapist": "Therapist",
}

PERSONA_COLORS = {
    "con_artist": "#e41a1c",
    "drill_sergeant": "#ff7f00",
    "farmer": "#4daf4a",
    "kindergarten_teacher": "#f781bf",
    "politician": "#984ea3",
    "professor": "#377eb8",
    "street_hustler": "#a65628",
    "surgeon": "#999999",
    "tech_ceo": "#e6ab02",
    "therapist": "#66c2a5",
}

TRAIT_MARKERS = {
    "assertiveness": "o",
    "confidence": "s",
    "deference": "^",
    "empathy": "D",
    "honesty": "v",
    "impulsivity": "P",
    "risk_taking": "*",
    "warmth": "X",
}

def pretty(slug: str) -> str:
    return PRETTY_PERSONAS.get(slug, slug.replace("_", " ").title())



def load_vectors(vectors_dir: Path, layer: int) -> dict[tuple[str, str], np.ndarray]:
    """Load all steering vectors at a given layer.

    Returns dict mapping (persona, trait) -> 1-d numpy array.
    """
    trait_values = {t.value for t in Trait}
    vectors = {}
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        vec_full = data["vector"]  # (n_layers, hidden_dim)
        persona = data.get("persona", "")
        trait = data.get("trait", "")
        if not persona or not trait:
            # Parse from filename
            stem = pt_file.stem
            for tv in trait_values:
                if stem.endswith(f"_{tv}"):
                    persona = stem[: -(len(tv) + 1)]
                    trait = tv
                    break
        if layer >= vec_full.shape[0]:
            log.warning("Layer %d out of range for %s (max %d), skipping",
                        layer, pt_file.name, vec_full.shape[0] - 1)
            continue
        vectors[(persona, trait)] = vec_full[layer].float().numpy()
    return vectors


def vectors_to_matrix(vectors: dict[tuple[str, str], np.ndarray],
                       personas: list[str], traits: list[str]) -> np.ndarray:
    """Stack vectors into (n_personas * n_traits, hidden_dim) matrix.

    Returns matrix, persona_labels, trait_labels (parallel lists).
    """
    rows, p_labels, t_labels = [], [], []
    for persona in personas:
        for trait in traits:
            key = (persona, trait)
            if key in vectors:
                rows.append(vectors[key])
                p_labels.append(persona)
                t_labels.append(trait)
    return np.stack(rows), p_labels, t_labels


# ---------------------------------------------------------------------------
# Figure 1: PCA of all raw vectors (colour by persona + colour by trait)
# ---------------------------------------------------------------------------

def fig_raw_vector_pca(vectors: dict, personas: list[str], traits: list[str],
                       output_dir: Path, label: str = "") -> None:
    matrix, p_labels, t_labels = vectors_to_matrix(vectors, personas, traits)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)

    suffix = f"_{label}" if label else ""
    title_suffix = f" ({label.upper()})" if label else ""

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: colour by persona
    ax = axes[0]
    for persona in personas:
        mask = [p == persona for p in p_labels]
        idx = [i for i, m in enumerate(mask) if m]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=PERSONA_COLORS.get(persona, "gray"),
                   label=pretty(persona), s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Steering Vectors by Persona{title_suffix}")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.2)

    # Right: colour by trait
    ax = axes[1]
    trait_colors = plt.cm.Set1(np.linspace(0, 1, len(traits)))
    trait_cmap = {t: trait_colors[i] for i, t in enumerate(traits)}
    for trait in traits:
        mask = [t == trait for t in t_labels]
        idx = [i for i, m in enumerate(mask) if m]
        marker = TRAIT_MARKERS.get(trait, "o")
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=[trait_cmap[trait]], marker=marker,
                   label=trait.replace("_", " ").title(), s=80, alpha=0.8,
                   edgecolors="white", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Steering Vectors by Trait{title_suffix}")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.2)

    fig.suptitle(f"PCA of Raw Steering Vectors (Layer 22){title_suffix}", fontsize=13, y=1.02)
    save_fig(fig, output_dir / f"raw_vector_pca{suffix}.png")


# ---------------------------------------------------------------------------
# Figure 2: Persona landscape from trait profile
# ---------------------------------------------------------------------------

def fig_persona_landscape(analysis_dir: Path, output_dir: Path, label: str = "",
                          axis_path: Path | None = None) -> None:
    decomp = load_json(analysis_dir / "decomposition.json")
    meta = load_json(analysis_dir / "transfer_meta.json")
    personas = meta["personas"]
    traits = meta["traits"]

    # Build 8-d trait profile per persona: shared_magnitude per trait
    profiles = np.zeros((len(personas), len(traits)))
    for ti, trait in enumerate(traits):
        for pi, persona in enumerate(personas):
            profiles[pi, ti] = decomp[trait]["shared_magnitudes"].get(persona, 0)

    suffix = f"_{label}" if label else ""
    title_suffix = f" ({label.upper()})" if label else ""

    pca = PCA(n_components=2)
    coords = pca.fit_transform(profiles)

    fig, ax = plt.subplots(figsize=(10, 8))
    for pi, persona in enumerate(personas):
        ax.scatter(coords[pi, 0], coords[pi, 1],
                   c=PERSONA_COLORS.get(persona, "gray"),
                   s=200, edgecolors="black", linewidth=1.5, zorder=3)
        ax.annotate(pretty(persona), (coords[pi, 0], coords[pi, 1]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=10, fontweight="bold")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Persona Landscape (Trait Profile Projection){title_suffix}", fontsize=13)
    ax.grid(alpha=0.3)

    # Add trait loading arrows (capped to fit within plot)
    loadings = pca.components_.T  # (n_traits, 2)
    scale = np.abs(coords).max() / np.abs(loadings).max() * 0.5
    for ti, trait in enumerate(traits):
        arrow_end = loadings[ti] * scale
        ax.annotate("", xy=(arrow_end[0], arrow_end[1]),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.6))
        ax.text(arrow_end[0] * 1.12, arrow_end[1] * 1.12,
                trait.replace("_", " ").title(), fontsize=8, color="gray",
                ha="center", va="center", alpha=0.8)

    # Expand axes to fit arrows and labels
    margin = 1.3
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] * margin, xlim[1] * margin)
    ax.set_ylim(ylim[0] * margin, ylim[1] * margin)

    save_fig(fig, output_dir / f"persona_landscape{suffix}.png")


# ---------------------------------------------------------------------------
# Figure 3: Persona-specific residual landscape
# ---------------------------------------------------------------------------

def fig_residual_landscape(vectors: dict, analysis_dir: Path,
                           personas: list[str], traits: list[str],
                           output_dir: Path, label: str = "") -> None:
    decomp = load_json(analysis_dir / "decomposition.json")

    # For each trait, compute the shared direction (PC1 across personas)
    # then subtract it from each vector to get the residual
    residuals = {}
    for trait in traits:
        trait_vecs = []
        trait_personas = []
        for persona in personas:
            key = (persona, trait)
            if key in vectors:
                trait_vecs.append(vectors[key])
                trait_personas.append(persona)
        if not trait_vecs:
            continue
        mat = np.stack(trait_vecs)  # (n_personas, hidden_dim)
        mean_vec = mat.mean(axis=0)
        centered = mat - mean_vec
        # PC1 = shared direction
        pca = PCA(n_components=1)
        pca.fit(centered)
        shared_dir = pca.components_[0]  # (hidden_dim,)
        shared_dir = shared_dir / (np.linalg.norm(shared_dir) + 1e-10)
        for i, persona in enumerate(trait_personas):
            proj = np.dot(mat[i], shared_dir) * shared_dir
            residuals[(persona, trait)] = mat[i] - proj

    # Build residual matrix and do PCA
    rows, p_labels, t_labels = [], [], []
    for persona in personas:
        for trait in traits:
            key = (persona, trait)
            if key in residuals:
                rows.append(residuals[key])
                p_labels.append(persona)
                t_labels.append(trait)

    if not rows:
        log.warning("No residual vectors to plot")
        return

    matrix = np.stack(rows)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)

    suffix = f"_{label}" if label else ""
    title_suffix = f" ({label.upper()})" if label else ""

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: by persona
    ax = axes[0]
    for persona in personas:
        mask = [p == persona for p in p_labels]
        idx = [i for i, m in enumerate(mask) if m]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=PERSONA_COLORS.get(persona, "gray"),
                   label=pretty(persona), s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Residual Vectors by Persona{title_suffix}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)

    # Right: by trait
    ax = axes[1]
    trait_colors = plt.cm.Set1(np.linspace(0, 1, len(traits)))
    trait_cmap = {t: trait_colors[i] for i, t in enumerate(traits)}
    for trait in traits:
        mask = [t == trait for t in t_labels]
        idx = [i for i, m in enumerate(mask) if m]
        marker = TRAIT_MARKERS.get(trait, "o")
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=[trait_cmap[trait]], marker=marker,
                   label=trait.replace("_", " ").title(), s=80, alpha=0.8,
                   edgecolors="white", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Residual Vectors by Trait{title_suffix}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)

    fig.suptitle(f"Persona-Specific Residuals (Shared Direction Removed){title_suffix}",
                 fontsize=13, y=1.02)
    save_fig(fig, output_dir / f"residual_landscape{suffix}.png")


# ---------------------------------------------------------------------------
# Figure 4: Assistant axis projection per persona
# ---------------------------------------------------------------------------

def fig_axis_projection(vectors: dict, axis_path: Path,
                        personas: list[str], traits: list[str],
                        layer: int, output_dir: Path, label: str = "") -> None:
    axis_full = torch.load(axis_path, map_location="cpu", weights_only=False)
    if isinstance(axis_full, dict):
        axis_full = axis_full["vector"]
    if layer >= axis_full.shape[0]:
        log.warning("Layer %d out of range for axis (max %d), skipping axis projection",
                    layer, axis_full.shape[0] - 1)
        return
    axis_vec = axis_full[layer].float().numpy()
    axis_unit = axis_vec / (np.linalg.norm(axis_vec) + 1e-10)

    suffix = f"_{label}" if label else ""
    title_suffix = f" ({label.upper()})" if label else ""

    # Compute mean axis projection per persona (across all traits)
    persona_projs = {}
    persona_per_trait = {}
    for persona in personas:
        projs = []
        per_trait = {}
        for trait in traits:
            key = (persona, trait)
            if key in vectors:
                proj = np.dot(vectors[key], axis_unit) / (np.linalg.norm(vectors[key]) + 1e-10)
                projs.append(proj)
                per_trait[trait] = proj
        if projs:
            persona_projs[persona] = np.mean(projs)
            persona_per_trait[persona] = per_trait

    # Sort by mean projection
    sorted_personas = sorted(persona_projs.keys(), key=lambda p: persona_projs[p])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: mean projection bar chart
    ax = axes[0]
    y = range(len(sorted_personas))
    values = [persona_projs[p] for p in sorted_personas]
    colors = [PERSONA_COLORS.get(p, "gray") for p in sorted_personas]
    bars = ax.barh(y, values, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(p) for p in sorted_personas], fontsize=10)
    ax.set_xlabel("Mean Cosine with Assistant Axis")
    ax.set_title(f"Mean Axis Alignment per Persona{title_suffix}")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005 * np.sign(bar.get_width()),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    # Right: per-trait heatmap
    ax = axes[1]
    matrix = np.zeros((len(sorted_personas), len(traits)))
    for pi, persona in enumerate(sorted_personas):
        for ti, trait in enumerate(traits):
            matrix[pi, ti] = persona_per_trait.get(persona, {}).get(trait, 0)

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([t.replace("_", " ").title() for t in traits],
                        rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(sorted_personas)))
    ax.set_yticklabels([pretty(p) for p in sorted_personas], fontsize=9)
    ax.set_title(f"Per-Trait Axis Alignment{title_suffix}")
    fig.colorbar(im, ax=ax, label="Cosine with Axis", shrink=0.8)

    for i in range(len(sorted_personas)):
        for j in range(len(traits)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(matrix[i, j]) > 0.4 else "black")

    fig.suptitle(f"Assistant Axis Projection{title_suffix}", fontsize=13, y=1.02)
    save_fig(fig, output_dir / f"axis_projection{suffix}.png")


# ---------------------------------------------------------------------------
# Figure 5: Side-by-side method comparison
# ---------------------------------------------------------------------------

def fig_method_comparison(vectors_iv: dict, vectors_caa: dict,
                          personas: list[str], traits: list[str],
                          output_dir: Path) -> None:
    """PCA of both methods' vectors, projected into the same space."""
    matrix_iv, p_iv, t_iv = vectors_to_matrix(vectors_iv, personas, traits)
    matrix_caa, p_caa, t_caa = vectors_to_matrix(vectors_caa, personas, traits)

    # Fit PCA on combined data
    combined = np.vstack([matrix_iv, matrix_caa])
    pca = PCA(n_components=2)
    all_coords = pca.fit_transform(combined)
    n_iv = len(matrix_iv)
    coords_iv = all_coords[:n_iv]
    coords_caa = all_coords[n_iv:]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, coords, p_labels, method in [
        (axes[0], coords_iv, p_iv, "Instruction-Variant"),
        (axes[1], coords_caa, p_caa, "CAA"),
    ]:
        for persona in personas:
            mask = [p == persona for p in p_labels]
            idx = [i for i, m in enumerate(mask) if m]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       c=PERSONA_COLORS.get(persona, "gray"),
                       label=pretty(persona), s=60, alpha=0.8,
                       edgecolors="white", linewidth=0.5)
            # Draw convex hull for each persona
            if len(idx) >= 3:
                from scipy.spatial import ConvexHull
                pts = coords[idx]
                try:
                    hull = ConvexHull(pts)
                    for simplex in hull.simplices:
                        ax.plot(pts[simplex, 0], pts[simplex, 1],
                                c=PERSONA_COLORS.get(persona, "gray"), alpha=0.3, linewidth=1)
                except Exception:
                    pass

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(method, fontsize=12)
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Method Comparison: Persona Clustering in Shared PCA Space",
                 fontsize=13, y=1.02)
    save_fig(fig, output_dir / "method_comparison.png")


# ---------------------------------------------------------------------------
# Figure 6: Persona distance matrix (from trait profiles)
# ---------------------------------------------------------------------------

def fig_persona_distance(vectors: dict, personas: list[str], traits: list[str],
                         output_dir: Path, label: str = "") -> None:
    """Compute pairwise cosine distance between persona trait profiles."""
    # Mean vector per persona (average across traits)
    persona_means = {}
    for persona in personas:
        vecs = [vectors[(persona, t)] for t in traits if (persona, t) in vectors]
        if vecs:
            persona_means[persona] = np.stack(vecs).mean(axis=0)

    n = len(personas)
    dist_matrix = np.zeros((n, n))
    for i, pa in enumerate(personas):
        for j, pb in enumerate(personas):
            if pa in persona_means and pb in persona_means:
                va = persona_means[pa]
                vb = persona_means[pb]
                cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10)
                dist_matrix[i, j] = cos

    suffix = f"_{label}" if label else ""
    title_suffix = f" ({label.upper()})" if label else ""

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dist_matrix, cmap="RdYlBu_r", vmin=-0.5, vmax=1.0)
    labels = [pretty(p) for p in personas]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if dist_matrix[i, j] < 0.3 else "black")
    fig.colorbar(im, ax=ax, label="Cosine Similarity (Mean Vector)", shrink=0.8)
    ax.set_title(f"Persona Similarity (Mean Across Traits){title_suffix}", fontsize=12)
    save_fig(fig, output_dir / f"persona_distance{suffix}.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate persona landscape figures")
    parser.add_argument("--vectors-dir", type=str, required=True)
    parser.add_argument("--analysis-dir", type=str, required=True)
    parser.add_argument("--vectors-dir-2", type=str, default=None,
                        help="Second method vectors dir (for comparison)")
    parser.add_argument("--analysis-dir-2", type=str, default=None,
                        help="Second method analysis dir (for comparison)")
    parser.add_argument("--axis", type=str, default=None,
                        help="Path to assistant axis .pt file")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label", type=str, default="iv",
                        help="Label for first method (default: iv)")
    parser.add_argument("--label-2", type=str, default="caa",
                        help="Label for second method (default: caa)")
    return parser.parse_args()


def main() -> None:
    from persona_steering.wandb_utils import init_run, finish_run, log_images, log_artifact, ensure_dir, infer_method

    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    analysis_dir = Path(args.analysis_dir)
    short = vectors_dir.parent.name
    vectors_dir = ensure_dir(f"{short}-vectors", vectors_dir, "*.pt")
    analysis_dir = ensure_dir(f"{short}-analysis", analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else vectors_dir.parent / "figures_landscape"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_json(analysis_dir / "transfer_meta.json")
    personas = meta["personas"]
    traits = meta["traits"]

    method = infer_method(vectors_dir)
    init_run("step5b_landscape", short, method=method)

    log.info("Loading vectors from %s...", vectors_dir)
    vectors = load_vectors(vectors_dir, args.layer)
    log.info("Loaded %d vectors (%d personas, %d traits)", len(vectors), len(personas), len(traits))

    # Figure 1: PCA of raw vectors
    fig_raw_vector_pca(vectors, personas, traits, output_dir, label=args.label)

    # Figure 2: Persona landscape from trait profiles
    fig_persona_landscape(analysis_dir, output_dir, label=args.label, axis_path=args.axis)

    # Figure 3: Residual landscape
    fig_residual_landscape(vectors, analysis_dir, personas, traits, output_dir, label=args.label)

    # Figure 4: Axis projection (if axis provided)
    if args.axis:
        fig_axis_projection(vectors, Path(args.axis), personas, traits,
                            args.layer, output_dir, label=args.label)

    # Figure 6: Persona distance matrix
    fig_persona_distance(vectors, personas, traits, output_dir, label=args.label)

    # Second method (if provided)
    if args.vectors_dir_2 and args.analysis_dir_2:
        vectors_dir_2 = Path(args.vectors_dir_2)
        analysis_dir_2 = Path(args.analysis_dir_2)

        log.info("Loading second method vectors from %s...", vectors_dir_2)
        vectors_2 = load_vectors(vectors_dir_2, args.layer)

        fig_raw_vector_pca(vectors_2, personas, traits, output_dir, label=args.label_2)
        fig_persona_landscape(analysis_dir_2, output_dir, label=args.label_2, axis_path=args.axis)
        fig_residual_landscape(vectors_2, analysis_dir_2, personas, traits, output_dir, label=args.label_2)
        if args.axis:
            fig_axis_projection(vectors_2, Path(args.axis), personas, traits,
                                args.layer, output_dir, label=args.label_2)
        fig_persona_distance(vectors_2, personas, traits, output_dir, label=args.label_2)

        # Figure 5: Side-by-side comparison
        fig_method_comparison(vectors, vectors_2, personas, traits, output_dir)

    log.info("All figures saved to %s", output_dir)
    for f in sorted(output_dir.glob("*.png")):
        log.info("  %s", f.name)

    log_images(output_dir, prefix="landscape")
    log_artifact(f"{short}-figures-landscape", "figures", output_dir, glob_pattern="*.png")
    finish_run()


if __name__ == "__main__":
    main()
