#!/usr/bin/env python3
"""Generate publication-ready figures from geometric analysis outputs.

Reads transfer matrices, decomposition results, axis alignment data from
pipeline step 4 and produces 7 figure types. No GPU needed.

Usage:
    python pipeline/5_visualize.py \
      --analysis-dir outputs/gemma-2-27b-it/analysis \
      --vectors-dir outputs/gemma-2-27b-it/vectors \
      --output-dir outputs/gemma-2-27b-it/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from persona_steering.config import Trait, PERSONA_SLUGS
from persona_steering.utils import log


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

PRETTY_TRAITS = {
    "assertiveness": "Assertiveness",
    "confidence": "Confidence",
    "deference": "Deference",
    "empathy": "Empathy",
    "honesty": "Honesty",
    "impulsivity": "Impulsivity",
    "risk_taking": "Risk-Taking",
    "warmth": "Warmth",
}


def pretty_persona(slug: str) -> str:
    return PRETTY_PERSONAS.get(slug, slug.replace("_", " ").title())


def pretty_trait(name: str) -> str:
    return PRETTY_TRAITS.get(name, name.replace("_", " ").title())


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_fig(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Figure 1: Transfer matrix heatmap (mean across traits)
# ---------------------------------------------------------------------------

def fig_transfer_heatmap(analysis_dir: Path, output_dir: Path) -> None:
    matrix = np.load(analysis_dir / "transfer_matrix.npy")
    meta = load_json(analysis_dir / "transfer_meta.json")
    personas = meta["personas"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0.5, vmax=1.0)

    labels = [pretty_persona(p) for p in personas]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(personas)):
        for j in range(len(personas)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if matrix[i, j] < 0.75 else "black")

    fig.colorbar(im, ax=ax, label="Mean Cosine Similarity", shrink=0.8)
    ax.set_title("Cross-Persona Transfer Matrix (All Traits, Layer 22)", fontsize=12)
    save_fig(fig, output_dir / "transfer_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 2: Per-trait transfer heatmaps (2x4 grid)
# ---------------------------------------------------------------------------

def fig_per_trait_heatmaps(analysis_dir: Path, output_dir: Path) -> None:
    meta = load_json(analysis_dir / "transfer_meta.json")
    personas = meta["personas"]
    traits = meta["traits"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Find global range for shared color scale
    all_matrices = []
    for trait in traits:
        path = analysis_dir / f"transfer_{trait}.npy"
        if path.exists():
            all_matrices.append(np.load(path))

    vmin = min(m.min() for m in all_matrices) if all_matrices else 0.0
    vmax = 1.0

    labels = [pretty_persona(p) for p in personas]
    for idx, trait in enumerate(traits):
        ax = axes[idx]
        path = analysis_dir / f"transfer_{trait}.npy"
        if not path.exists():
            ax.set_visible(False)
            continue

        matrix = np.load(path)
        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(pretty_trait(trait), fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)

    # Hide unused axes
    for idx in range(len(traits), len(axes)):
        axes[idx].set_visible(False)

    fig.colorbar(im, ax=axes, label="Cosine Similarity", shrink=0.6)
    fig.suptitle("Per-Trait Cross-Persona Transfer (Layer 22)", fontsize=14, y=1.02)
    save_fig(fig, output_dir / "per_trait_heatmaps.png")


# ---------------------------------------------------------------------------
# Figure 3: Shared variance bar chart
# ---------------------------------------------------------------------------

def fig_shared_variance(analysis_dir: Path, output_dir: Path) -> None:
    decomp = load_json(analysis_dir / "decomposition.json")

    traits = sorted(decomp.keys(), key=lambda t: decomp[t]["variance_explained"])
    values = [decomp[t]["variance_explained"] * 100 for t in traits]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh([pretty_trait(t) for t in traits], values, color="#4C72B0", edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Shared Variance Explained (%)")
    ax.set_title("Fraction of Steering Vector Variance in Shared Direction", fontsize=12)
    ax.axvline(x=80, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
    ax.legend(fontsize=8)
    save_fig(fig, output_dir / "shared_variance.png")


# ---------------------------------------------------------------------------
# Figure 4: Persona specificity chart
# ---------------------------------------------------------------------------

def fig_persona_specificity(analysis_dir: Path, output_dir: Path) -> None:
    decomp = load_json(analysis_dir / "decomposition.json")
    meta = load_json(analysis_dir / "transfer_meta.json")
    personas = meta["personas"]
    traits = meta["traits"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(traits))
    width = 0.8 / len(personas)
    colors = plt.cm.tab10(np.linspace(0, 1, len(personas)))

    for pi, persona in enumerate(personas):
        ratios = []
        for trait in traits:
            d = decomp.get(trait, {})
            shared = abs(d.get("shared_magnitudes", {}).get(persona, 0))
            specific = d.get("specific_magnitudes", {}).get(persona, 0)
            total = shared + specific
            ratios.append(specific / total if total > 0 else 0)

        offset = (pi - len(personas) / 2 + 0.5) * width
        ax.bar(x + offset, ratios, width, label=pretty_persona(persona),
               color=colors[pi], edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_trait(t) for t in traits], fontsize=9)
    ax.set_ylabel("Persona-Specific Ratio")
    ax.set_title("Persona Specificity: Specific / (Shared + Specific) per Trait", fontsize=12)
    ax.legend(fontsize=7, ncol=5, loc="upper right")
    ax.set_ylim(0, 0.7)
    save_fig(fig, output_dir / "persona_specificity.png")


# ---------------------------------------------------------------------------
# Figure 5: Layer sweep lines
# ---------------------------------------------------------------------------

def fig_layer_sweep(analysis_dir: Path, output_dir: Path) -> None:
    parent = analysis_dir.parent

    # Discover analysis_layer_* directories
    layer_dirs = sorted(parent.glob("analysis_layer_*"))
    if not layer_dirs:
        log.warning("No analysis_layer_* dirs found, skipping layer sweep figure")
        return

    # Also include the main analysis dir (extract its layer from transfer_meta)
    layer_data: dict[int, dict[str, float]] = {}

    for ld in layer_dirs:
        match = re.search(r"analysis_layer_(\d+)", ld.name)
        if not match:
            continue
        layer_num = int(match.group(1))
        decomp_path = ld / "decomposition.json"
        if not decomp_path.exists():
            continue
        decomp = load_json(decomp_path)
        layer_data[layer_num] = {t: d["variance_explained"] for t, d in decomp.items()}

    # Add main analysis dir
    main_meta = analysis_dir / "transfer_meta.json"
    main_decomp = analysis_dir / "decomposition.json"
    if main_meta.exists() and main_decomp.exists():
        meta = load_json(main_meta)
        layer_num = meta.get("layer", 22)
        decomp = load_json(main_decomp)
        layer_data[layer_num] = {t: d["variance_explained"] for t, d in decomp.items()}

    if not layer_data:
        log.warning("No layer data found for sweep figure")
        return

    layers = sorted(layer_data.keys())
    all_traits = set()
    for ld in layer_data.values():
        all_traits.update(ld.keys())
    traits = sorted(all_traits)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(traits)))

    for ti, trait in enumerate(traits):
        values = [layer_data[l].get(trait, float("nan")) * 100 for l in layers]
        ax.plot(layers, values, marker="o", markersize=4, label=pretty_trait(trait),
                color=colors[ti], linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Shared Variance Explained (%)")
    ax.set_title("Shared Variance by Layer", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(50, 100)
    ax.grid(alpha=0.3)
    save_fig(fig, output_dir / "layer_sweep.png")


# ---------------------------------------------------------------------------
# Figure 6: Axis alignment comparison
# ---------------------------------------------------------------------------

def fig_axis_alignment(analysis_dir: Path, output_dir: Path) -> None:
    summary_path = analysis_dir / "axis_alignment_summary.json"
    residual_path = analysis_dir / "residual_axis_alignment.json"

    if not summary_path.exists() or not residual_path.exists():
        log.warning("Missing axis alignment files, skipping figure")
        return

    summary = load_json(summary_path)
    residual = load_json(residual_path)

    traits = sorted(summary.keys())

    full_cos = [summary[t]["mean_abs_cosine"] for t in traits]
    resid_cos = [residual[t]["mean_abs_cosine"] for t in traits]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(traits))
    width = 0.35

    ax.bar(x - width / 2, full_cos, width, label="Full Vector |cos|", color="#4C72B0")
    ax.bar(x + width / 2, resid_cos, width, label="Residual |cos|", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_trait(t) for t in traits], fontsize=9)
    ax.set_ylabel("|Cosine| with Assistant Axis")
    ax.set_title("Full Vector vs Persona-Specific Residual: Alignment with Assistant Axis", fontsize=11)
    ax.legend()
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, output_dir / "axis_alignment_comparison.png")


# ---------------------------------------------------------------------------
# Figure 7: Extreme pair comparison (drill_sergeant vs therapist)
# ---------------------------------------------------------------------------

def fig_extreme_pair(analysis_dir: Path, vectors_dir: Path, output_dir: Path) -> None:
    import torch

    meta = load_json(analysis_dir / "transfer_meta.json")
    personas = meta["personas"]
    traits = meta["traits"]
    layer = meta.get("layer", 22)

    pa, pb = "drill_sergeant", "therapist"
    if pa not in personas or pb not in personas:
        log.warning("Extreme pair personas not found, skipping")
        return

    # Load per-trait cosines from per-trait transfer matrices
    cosines = []
    for trait in traits:
        path = analysis_dir / f"transfer_{trait}.npy"
        if not path.exists():
            cosines.append(0)
            continue
        matrix = np.load(path)
        ia = personas.index(pa)
        ib = personas.index(pb)
        cosines.append(matrix[ia, ib])

    # Load axis alignment for both personas
    alignment_path = analysis_dir / "axis_alignment.json"
    axis_data = load_json(alignment_path) if alignment_path.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-trait cosine similarity
    ax = axes[0]
    trait_labels = [pretty_trait(t) for t in traits]
    bars = ax.bar(trait_labels, cosines, color="#4C72B0", edgecolor="white")
    for bar, val in zip(bars, cosines):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"{pretty_persona(pa)} vs {pretty_persona(pb)}: Per-Trait Similarity", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # Right: axis alignment ratio vs orthogonal ratio for both personas
    ax = axes[1]
    if axis_data and pa in axis_data and pb in axis_data:
        x = np.arange(len(traits))
        width = 0.35

        pa_along = []
        pb_along = []
        for trait in traits:
            pa_entry = axis_data.get(pa, {}).get(trait, {})
            pb_entry = axis_data.get(pb, {}).get(trait, {})
            pa_ratio = abs(pa_entry.get("alignment_ratio", 0))
            pb_ratio = abs(pb_entry.get("alignment_ratio", 0))
            pa_along.append(pa_ratio * 100)
            pb_along.append(pb_ratio * 100)

        ax.bar(x - width / 2, pa_along, width, label=pretty_persona(pa), color="#C44E52")
        ax.bar(x + width / 2, pb_along, width, label=pretty_persona(pb), color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(trait_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% Along Assistant Axis")
        ax.set_title("Fraction of Vector Along Assistant Axis", fontsize=11)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No axis data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Axis Alignment (unavailable)")

    fig.suptitle(f"Extreme Pair: {pretty_persona(pa)} vs {pretty_persona(pb)}", fontsize=13, y=1.02)
    save_fig(fig, output_dir / "extreme_pair.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis figures")
    parser.add_argument("--analysis-dir", type=str, required=True,
                        help="Directory with analysis results from step 4")
    parser.add_argument("--vectors-dir", type=str, default=None,
                        help="Directory with vector .pt files (for extreme pair)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: sibling 'figures' dir)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    vectors_dir = Path(args.vectors_dir) if args.vectors_dir else analysis_dir.parent / "vectors"
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating figures from %s -> %s", analysis_dir, output_dir)

    fig_transfer_heatmap(analysis_dir, output_dir)
    fig_per_trait_heatmaps(analysis_dir, output_dir)
    fig_shared_variance(analysis_dir, output_dir)
    fig_persona_specificity(analysis_dir, output_dir)
    fig_layer_sweep(analysis_dir, output_dir)
    fig_axis_alignment(analysis_dir, output_dir)
    fig_extreme_pair(analysis_dir, vectors_dir, output_dir)

    log.info("All figures saved to %s", output_dir)
    for f in sorted(output_dir.glob("*.png")):
        log.info("  %s", f.name)


if __name__ == "__main__":
    main()
