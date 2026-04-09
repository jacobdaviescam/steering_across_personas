#!/usr/bin/env python3
"""E6c: Generate publication-ready figures for the basin geometry experiment.

Produces:
  1. Similarity-vs-ring curves (one per trait) — the money figure
  2. Combined overlay plot with all traits
  3. Cross-trait control matrix
  4. Dynamic drift curves (similarity vs token position)
  5. Permutation null distribution histograms

Usage:
    python pipeline/e6c_basin_figures.py --basin-dir outputs/gemma-2-27b-it/analysis/basin
    python pipeline/e6c_basin_figures.py --basin-dir outputs/gemma-2-27b-it/analysis/basin --dynamic-dir outputs/gemma-2-27b-it/analysis/basin/dynamic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from persona_steering.config import BASIN_GRADIENTS, TARGET_LAYER
from persona_steering.utils import log, cosine_similarity

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

TRAIT_COLORS = {
    "honesty": "#2196F3",
    "empathy": "#4CAF50",
    "risk_taking": "#F44336",
}

TRAIT_LABELS = {
    "honesty": "Honesty",
    "empathy": "Empathy",
    "risk_taking": "Risk-Taking",
}


def pretty_persona(slug: str) -> str:
    return slug.replace("_", " ").title()


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Figure 1: Per-trait similarity vs ring
# ---------------------------------------------------------------------------

def fig_similarity_vs_ring(basin_results: dict, output_dir: Path) -> None:
    """One subplot per trait: cosine similarity to default vs ring number."""
    traits = [t for t in ["honesty", "empathy", "risk_taking"] if t in basin_results]
    if not traits:
        log.warning("No trait results found for similarity-vs-ring plot")
        return

    fig, axes = plt.subplots(1, len(traits), figsize=(6 * len(traits), 5), sharey=True)
    if len(traits) == 1:
        axes = [axes]

    for ax, trait in zip(axes, traits):
        res = basin_results[trait]
        personas = res["personas"]

        rings = [p["ring"] for p in personas]
        sims = [p["cosine_sim_to_default"] for p in personas]
        names = [p["persona"] for p in personas]

        color = TRAIT_COLORS.get(trait, "#666666")

        ax.scatter(rings, sims, c=color, s=80, zorder=5, edgecolors="white", linewidth=0.5)

        # Connect with line
        order = np.argsort(rings)
        ax.plot([rings[i] for i in order], [sims[i] for i in order],
                c=color, alpha=0.4, linewidth=1.5, linestyle="--")

        # Label points
        for r, s, name in zip(rings, sims, names):
            ax.annotate(pretty_persona(name), (r, s),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, alpha=0.8)

        rho = res["spearman_rho"]
        perm_p = res["permutation_test"]["p_value"]
        ax.set_title(f"{TRAIT_LABELS.get(trait, trait)}\n"
                     f"$\\rho_s$ = {rho:.3f}, p = {perm_p:.4f}",
                     fontsize=12)
        ax.set_xlabel("Conceptual Distance (Ring)", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Cosine Similarity to Default", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    fig.suptitle("Basin Geometry: Cosine Similarity Decay Along Trait Gradients",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, output_dir / "basin_similarity_vs_ring.pdf")
    save_fig(fig, output_dir / "basin_similarity_vs_ring.png")


# ---------------------------------------------------------------------------
# Figure 2: Overlay plot (all traits on one axis)
# ---------------------------------------------------------------------------

def fig_overlay(basin_results: dict, output_dir: Path) -> None:
    """All traits on a single axis, normalised ring (0-1) for comparability."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for trait, res in basin_results.items():
        personas = res["personas"]
        rings = np.array([p["ring"] for p in personas])
        sims = np.array([p["cosine_sim_to_default"] for p in personas])

        # Normalise rings to [0, 1] for cross-trait comparison
        ring_norm = rings / rings.max() if rings.max() > 0 else rings

        order = np.argsort(ring_norm)
        color = TRAIT_COLORS.get(trait, "#666666")
        label = TRAIT_LABELS.get(trait, trait)
        rho = res["spearman_rho"]

        ax.plot(ring_norm[order], sims[order], marker="o", markersize=6,
                label=f"{label} ($\\rho_s$={rho:.3f})", color=color, linewidth=2)

    ax.set_xlabel("Normalised Conceptual Distance", fontsize=11)
    ax.set_ylabel("Cosine Similarity to Default", fontsize=11)
    ax.set_title("Basin Geometry: Similarity Decay Across Trait Gradients", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlim(-0.05, 1.05)

    save_fig(fig, output_dir / "basin_overlay.pdf")
    save_fig(fig, output_dir / "basin_overlay.png")


# ---------------------------------------------------------------------------
# Figure 3: Cross-trait control heatmap
# ---------------------------------------------------------------------------

def fig_cross_trait_control(cross_results: dict, output_dir: Path) -> None:
    """Heatmap: applying trait A's ring ordering to trait B's similarities."""
    if not cross_results:
        log.warning("No cross-trait results, skipping heatmap")
        return

    traits = sorted({k.split("_ordering_on_")[0] for k in cross_results})
    n = len(traits)
    matrix = np.full((n, n), np.nan)
    trait_idx = {t: i for i, t in enumerate(traits)}

    for key, val in cross_results.items():
        parts = key.split("_ordering_on_")
        if len(parts) != 2:
            continue
        ta = parts[0]
        # Extract trait_b: everything after "_ordering_on_" and before "_sims"
        tb = parts[1].replace("_sims", "")
        if ta in trait_idx and tb in trait_idx:
            matrix[trait_idx[ta], trait_idx[tb]] = val["rho"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    labels = [TRAIT_LABELS.get(t, t) for t in traits]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Similarity From (Trait B)", fontsize=11)
    ax.set_ylabel("Ring Ordering From (Trait A)", fontsize=11)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if abs(matrix[i, j]) > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Spearman $\\rho$", shrink=0.8)
    ax.set_title("Cross-Trait Control\n(off-diagonal should be weak)", fontsize=12)

    save_fig(fig, output_dir / "basin_cross_trait_control.pdf")
    save_fig(fig, output_dir / "basin_cross_trait_control.png")


# ---------------------------------------------------------------------------
# Figure 4: Dynamic drift (similarity vs token position)
# ---------------------------------------------------------------------------

def fig_dynamic_drift(
    dynamic_dir: Path,
    vectors_dir: Path | None,
    layer: int,
    output_dir: Path,
) -> None:
    """Plot cosine similarity to default vector at each token position.

    One subplot per trait, lines for each persona coloured by ring.
    """
    import torch

    summary_path = dynamic_dir / "dynamic_summary.json"
    if not summary_path.exists():
        log.warning("No dynamic summary found at %s, skipping drift plots", summary_path)
        return

    # Load all positional vector files
    pt_files = sorted(dynamic_dir.glob("*_positional.pt"))
    if not pt_files:
        log.warning("No positional .pt files found in %s", dynamic_dir)
        return

    # Load default positional vectors for reference
    # Group by trait
    trait_data: dict[str, list[dict]] = {}
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        persona = data["persona"]
        trait = data["trait"]
        positions = sorted(data["vectors"].keys())
        trait_data.setdefault(trait, []).append({
            "persona": persona,
            "positions": positions,
            "vectors": data["vectors"],
        })

    # For each trait, compute similarity between each persona's positional vector
    # and the default persona's positional vector
    for trait, entries in trait_data.items():
        # Find default entry
        default_entry = None
        for e in entries:
            if e["persona"] == "default":
                default_entry = e
                break
        if default_entry is None:
            log.warning("No default positional vectors for trait %s, skipping", trait)
            continue

        gradient = BASIN_GRADIENTS.get(trait, [])
        ring_map = {slug: ring for slug, ring in gradient}
        max_ring = max(ring_map.values()) if ring_map else 1

        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.cm.RdYlGn_r

        for entry in entries:
            slug = entry["persona"]
            if slug == "default":
                continue

            ring = ring_map.get(slug, -1)
            if ring < 0:
                continue

            # Compute similarity at each shared position
            shared_positions = sorted(set(entry["positions"]) & set(default_entry["positions"]))
            if not shared_positions:
                continue

            sims = []
            for pos in shared_positions:
                v_persona = entry["vectors"][pos].float()
                v_default = default_entry["vectors"][pos].float()
                sim = torch.dot(v_persona, v_default) / (v_persona.norm() * v_default.norm() + 1e-8)
                sims.append(sim.item())

            color = cmap(ring / max_ring)
            ax.plot(shared_positions, sims, marker="o", markersize=4,
                    label=f"{pretty_persona(slug)} (ring {ring})",
                    color=color, linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Token Position in Assistant Turn", fontsize=11)
        ax.set_ylabel("Cosine Similarity to Default Vector", fontsize=11)
        ax.set_title(f"Dynamic Drift: {TRAIT_LABELS.get(trait, trait)}\n"
                     f"(basin prediction: far personas drift more over time)",
                     fontsize=12)
        ax.legend(fontsize=7, loc="lower left", ncol=2)
        ax.grid(True, alpha=0.3)

        save_fig(fig, output_dir / f"basin_dynamic_drift_{trait}.pdf")
        save_fig(fig, output_dir / f"basin_dynamic_drift_{trait}.png")


# ---------------------------------------------------------------------------
# Figure 5: Permutation null distribution
# ---------------------------------------------------------------------------

def fig_permutation_null(basin_results: dict, output_dir: Path) -> None:
    """Histogram of null distribution with observed rho marked."""
    traits = [t for t in ["honesty", "empathy", "risk_taking"] if t in basin_results]
    if not traits:
        return

    fig, axes = plt.subplots(1, len(traits), figsize=(5 * len(traits), 4))
    if len(traits) == 1:
        axes = [axes]

    for ax, trait in zip(axes, traits):
        res = basin_results[trait]
        perm = res["permutation_test"]
        observed = perm["observed_rho"]
        null_mean = perm["null_mean"]
        null_std = perm["null_std"]

        # Generate approximate null distribution for visualization
        rng = np.random.default_rng(42)
        null_samples = rng.normal(null_mean, null_std, 10000)

        color = TRAIT_COLORS.get(trait, "#666666")

        ax.hist(null_samples, bins=80, color="gray", alpha=0.5, density=True,
                label="Null distribution")
        ax.axvline(observed, color=color, linewidth=2.5, linestyle="-",
                   label=f"Observed $\\rho_s$ = {observed:.3f}")
        ax.axvline(-observed, color=color, linewidth=1, linestyle=":",
                   alpha=0.5)

        ax.set_title(f"{TRAIT_LABELS.get(trait, trait)}\np = {perm['p_value']:.4f}",
                     fontsize=11)
        ax.set_xlabel("Spearman $\\rho$", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Permutation Test: Observed vs Null", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, output_dir / "basin_permutation_null.pdf")
    save_fig(fig, output_dir / "basin_permutation_null.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6c: Basin geometry figures")
    parser.add_argument(
        "--basin-dir", type=str, required=True,
        help="Directory containing basin_results.json and cross_trait_control.json",
    )
    parser.add_argument(
        "--dynamic-dir", type=str, default=None,
        help="Directory containing dynamic positional .pt files (optional)",
    )
    parser.add_argument(
        "--vectors-dir", type=str, default=None,
        help="Directory containing full steering vectors (for dynamic reference)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures (default: basin-dir/figures)",
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Layer for dynamic analysis (default: {TARGET_LAYER})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    basin_dir = Path(args.basin_dir)
    output_dir = Path(args.output_dir) if args.output_dir else basin_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    basin_path = basin_dir / "basin_results.json"
    cross_path = basin_dir / "cross_trait_control.json"

    if not basin_path.exists():
        log.error("No basin_results.json found at %s", basin_path)
        return

    basin_results = load_json(basin_path)
    cross_results = load_json(cross_path) if cross_path.exists() else {}

    log.info("Loaded basin results for traits: %s", list(basin_results.keys()))

    # Generate figures
    fig_similarity_vs_ring(basin_results, output_dir)
    fig_overlay(basin_results, output_dir)
    fig_cross_trait_control(cross_results, output_dir)
    fig_permutation_null(basin_results, output_dir)

    # Dynamic drift (optional)
    if args.dynamic_dir:
        dynamic_dir = Path(args.dynamic_dir)
        vectors_dir = Path(args.vectors_dir) if args.vectors_dir else None
        fig_dynamic_drift(dynamic_dir, vectors_dir, args.layer, output_dir)

    log.info("All figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
