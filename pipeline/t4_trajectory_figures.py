#!/usr/bin/env python3
"""Generate publication-quality figures for the OLMo training trajectory experiment.

Usage:
    python pipeline/t4_trajectory_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

TRAJECTORY_DIR = Path("outputs/OLMo-2-1124-7B/trajectory")
FIGURES_DIR = Path("outputs/OLMo-2-1124-7B/figures/trajectory")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STAGES = [
    "pretrain_1pct", "pretrain_10pct", "pretrain_50pct",
    "base", "sft", "dpo", "instruct",
]
STAGE_LABELS = ["1%", "10%", "50%", "Base", "SFT", "DPO", "Instruct"]
SFT_BOUNDARY = 3.5  # between index 3 (base) and 4 (sft)

# Colors: blues for pretraining, then distinct for post-training
STAGE_COLORS = {
    "pretrain_1pct": "#a6cee3",
    "pretrain_10pct": "#6baed6",
    "pretrain_50pct": "#2171b5",
    "base": "#636363",
    "sft": "#e6550d",
    "dpo": "#d62728",
    "instruct": "#2ca02c",
}
STAGE_COLOR_LIST = [STAGE_COLORS[s] for s in STAGES]

# Trait colors
TRAIT_CMAP = plt.cm.Set2
TRAITS = [
    "assertiveness", "confidence", "deference", "empathy",
    "honesty", "impulsivity", "risk_taking", "warmth",
]
TRAIT_COLORS = {t: TRAIT_CMAP(i / len(TRAITS)) for i, t in enumerate(TRAITS)}
TRAIT_LABELS = {
    "assertiveness": "Assertiveness", "confidence": "Confidence",
    "deference": "Deference", "empathy": "Empathy",
    "honesty": "Honesty", "impulsivity": "Impulsivity",
    "risk_taking": "Risk-taking", "warmth": "Warmth",
}

PERSONA_LABELS = {
    "con_artist": "Con Artist", "drill_sergeant": "Drill Sgt",
    "farmer": "Farmer", "kindergarten_teacher": "K. Teacher",
    "politician": "Politician", "professor": "Professor",
    "street_hustler": "Hustler", "surgeon": "Surgeon",
    "tech_ceo": "Tech CEO", "therapist": "Therapist",
}


def load_json(name: str) -> dict:
    with open(TRAJECTORY_DIR / name) as f:
        return json.load(f)


def add_sft_boundary(ax, label=True):
    ax.axvline(SFT_BOUNDARY, color="#999", ls="--", lw=1, alpha=0.7)
    if label:
        ax.text(SFT_BOUNDARY + 0.08, ax.get_ylim()[1] * 0.97, "SFT",
                fontsize=7, color="#666", va="top")


# ---------------------------------------------------------------------------
# Figure 1: Transfer matrix heatmaps across stages
# ---------------------------------------------------------------------------

def fig_transfer_matrices():
    meta = load_json("trajectory_meta.json")
    personas = meta["personas"]
    plabels = [PERSONA_LABELS.get(p, p) for p in personas]

    fig, axes = plt.subplots(1, 7, figsize=(18, 3.2), sharey=True)
    fig.suptitle("Persona Transfer Matrices Across Training Stages", fontsize=12, y=1.02)

    vmin, vmax = -0.2, 1.0

    for i, (stage, label) in enumerate(zip(STAGES, STAGE_LABELS)):
        tm = np.load(TRAJECTORY_DIR / f"transfer_{stage}.npy")
        ax = axes[i]
        sns.heatmap(
            tm, ax=ax, vmin=vmin, vmax=vmax, cmap="RdBu_r",
            square=True, cbar=i == 6,
            cbar_kws={"shrink": 0.8, "label": "Cosine sim"} if i == 6 else {},
            xticklabels=plabels if i == 0 else False,
            yticklabels=plabels if i == 0 else False,
            linewidths=0.3,
        )
        ax.set_title(label, fontsize=10, fontweight="bold",
                     color=STAGE_COLORS[stage])
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_transfer_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig_transfer_matrices.png")


# ---------------------------------------------------------------------------
# Figure 2: Transfer matrix distance trajectory
# ---------------------------------------------------------------------------

def fig_transfer_distance():
    distances = load_json("transfer_matrix_distances.json")

    frob = [distances[s]["instruct"]["frobenius"] for s in STAGES]
    spear = [distances[s]["instruct"]["spearman_rho"] for s in STAGES]

    fig, ax1 = plt.subplots(figsize=(6, 3.5))

    ax1.plot(range(len(STAGES)), frob, "o-", color="#d62728", lw=2, ms=6,
             label="Frobenius distance", zorder=3)
    ax1.set_ylabel("Frobenius Distance to Instruct", color="#d62728", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xticks(range(len(STAGES)))
    ax1.set_xticklabels(STAGE_LABELS, fontsize=9)
    ax1.set_xlabel("Training Stage", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(range(len(STAGES)), spear, "s-", color="#2171b5", lw=2, ms=6,
             label="Spearman \u03c1", zorder=3)
    ax2.set_ylabel("Spearman \u03c1 with Instruct", color="#2171b5", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#2171b5")
    ax2.set_ylim(0, 1.05)

    ax1.axvline(SFT_BOUNDARY, color="#999", ls="--", lw=1, alpha=0.7)
    ax1.text(SFT_BOUNDARY + 0.1, max(frob) * 0.95, "SFT boundary",
             fontsize=8, color="#666", va="top")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)

    ax1.set_title("Transfer Matrix Distance to Final (Instruct) Model", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_transfer_distance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig_transfer_distance.png")


# ---------------------------------------------------------------------------
# Figure 3: Vector alignment heatmap (stage × stage)
# ---------------------------------------------------------------------------

def fig_vector_alignment():
    alignment = load_json("vector_alignment.json")

    # Compute mean cosine across all persona-trait pairs for each stage pair
    matrix = np.zeros((len(STAGES), len(STAGES)))
    for i, sa in enumerate(STAGES):
        for j, sb in enumerate(STAGES):
            cosines = []
            for persona in alignment:
                for trait in alignment[persona]:
                    entry = alignment[persona][trait]
                    if sa in entry and sb in entry.get(sa, {}):
                        cosines.append(entry[sa][sb])
            matrix[i, j] = np.mean(cosines) if cosines else 0

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        matrix, ax=ax, vmin=0, vmax=1, cmap="YlOrRd",
        annot=True, fmt=".2f", annot_kws={"size": 8},
        xticklabels=STAGE_LABELS, yticklabels=STAGE_LABELS,
        square=True, linewidths=0.5,
        cbar_kws={"label": "Mean cosine similarity"},
    )
    ax.set_title("Vector Alignment Across Training Stages", fontsize=11)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_vector_alignment.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig_vector_alignment.png")


# ---------------------------------------------------------------------------
# Figure 4: Subspace overlap trajectory (per trait)
# ---------------------------------------------------------------------------

def fig_subspace_overlap():
    overlap = load_json("subspace_overlap.json")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    for trait in TRAITS:
        vals = []
        for stage in STAGES:
            entry = overlap.get(trait, {}).get(stage, {}).get("instruct", {})
            vals.append(entry.get("mean_overlap", 0))
        ax.plot(range(len(STAGES)), vals, "o-", color=TRAIT_COLORS[trait],
                lw=1.5, ms=5, label=TRAIT_LABELS[trait])

    # Mean line
    mean_vals = []
    for si, stage in enumerate(STAGES):
        trait_vals = []
        for trait in TRAITS:
            entry = overlap.get(trait, {}).get(stage, {}).get("instruct", {})
            trait_vals.append(entry.get("mean_overlap", 0))
        mean_vals.append(np.mean(trait_vals))
    ax.plot(range(len(STAGES)), mean_vals, "k-", lw=2.5, alpha=0.5, label="Mean")

    add_sft_boundary(ax, label=False)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=9)
    ax.set_xlabel("Training Stage", fontsize=10)
    ax.set_ylabel("Subspace Overlap with Instruct", fontsize=10)
    ax.set_title("Persona Subspace Overlap Across Training", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_subspace_overlap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig_subspace_overlap.png")


# ---------------------------------------------------------------------------
# Figure 5: Variance trajectory (shared vs specific)
# ---------------------------------------------------------------------------

def fig_variance_trajectory():
    variance = load_json("variance_trajectory.json")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    for trait in TRAITS:
        vals = [variance[stage].get(trait, 0) for stage in STAGES]
        ax.plot(range(len(STAGES)), vals, "o-", color=TRAIT_COLORS[trait],
                lw=1.5, ms=5, label=TRAIT_LABELS[trait])

    # Mean
    mean_vals = [np.mean([variance[s].get(t, 0) for t in TRAITS]) for s in STAGES]
    ax.plot(range(len(STAGES)), mean_vals, "k-", lw=2.5, alpha=0.5, label="Mean")

    # Shade the drop
    ax.axhspan(min(mean_vals) - 0.01, max(mean_vals) + 0.01,
               xmin=0, xmax=1, alpha=0.03, color="gray")

    add_sft_boundary(ax, label=False)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=9)
    ax.set_xlabel("Training Stage", fontsize=10)
    ax.set_ylabel("Shared Variance Ratio", fontsize=10)
    ax.set_title("Shared vs Persona-Specific Variance Across Training", fontsize=11)
    ax.set_ylim(0.70, 1.0)
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_variance_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig_variance_trajectory.png")


# ---------------------------------------------------------------------------
# Figure 6: Summary dashboard (2x2)
# ---------------------------------------------------------------------------

def fig_summary():
    distances = load_json("transfer_matrix_distances.json")
    alignment = load_json("vector_alignment.json")
    overlap = load_json("subspace_overlap.json")
    variance = load_json("variance_trajectory.json")

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle("Training Trajectory Summary: OLMo-2 7B", fontsize=13, y=1.01)

    xs = range(len(STAGES))

    # (a) Frobenius distance
    ax = axes[0, 0]
    frob = [distances[s]["instruct"]["frobenius"] for s in STAGES]
    ax.plot(xs, frob, "o-", color="#d62728", lw=2, ms=6)
    ax.axvline(SFT_BOUNDARY, color="#999", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(STAGE_LABELS, fontsize=8)
    ax.set_ylabel("Frobenius Distance", fontsize=9)
    ax.set_title("(a) Transfer Matrix Distance to Instruct", fontsize=9)

    # (b) Mean vector cosine
    ax = axes[0, 1]
    mean_cosines = []
    for stage in STAGES:
        cosines = []
        for persona in alignment:
            for trait in alignment[persona]:
                entry = alignment[persona][trait]
                if stage in entry and "instruct" in entry.get(stage, {}):
                    cosines.append(entry[stage]["instruct"])
        mean_cosines.append(np.mean(cosines) if cosines else 0)
    ax.plot(xs, mean_cosines, "o-", color="#2171b5", lw=2, ms=6)
    ax.axvline(SFT_BOUNDARY, color="#999", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(STAGE_LABELS, fontsize=8)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=9)
    ax.set_title("(b) Vector Alignment with Instruct", fontsize=9)
    ax.set_ylim(0, 1.05)

    # (c) Subspace overlap
    ax = axes[1, 0]
    mean_overlap = []
    for stage in STAGES:
        trait_vals = []
        for trait in TRAITS:
            entry = overlap.get(trait, {}).get(stage, {}).get("instruct", {})
            trait_vals.append(entry.get("mean_overlap", 0))
        mean_overlap.append(np.mean(trait_vals))
    ax.plot(xs, mean_overlap, "o-", color="#e6550d", lw=2, ms=6)
    ax.axvline(SFT_BOUNDARY, color="#999", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(STAGE_LABELS, fontsize=8)
    ax.set_ylabel("Mean Subspace Overlap", fontsize=9)
    ax.set_title("(c) Persona Subspace Overlap with Instruct", fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    # (d) Shared variance ratio
    ax = axes[1, 1]
    mean_var = [np.mean([variance[s].get(t, 0) for t in TRAITS]) for s in STAGES]
    ax.plot(xs, mean_var, "o-", color="#2ca02c", lw=2, ms=6)
    ax.axvline(SFT_BOUNDARY, color="#999", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(STAGE_LABELS, fontsize=8)
    ax.set_ylabel("Shared Variance Ratio", fontsize=9)
    ax.set_title("(d) Shared vs Persona-Specific Variance", fontsize=9)
    ax.set_ylim(0.70, 1.0)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    fig_transfer_matrices()
    fig_transfer_distance()
    fig_vector_alignment()
    fig_subspace_overlap()
    fig_variance_trajectory()
    fig_summary()

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
