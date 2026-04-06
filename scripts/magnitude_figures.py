#!/usr/bin/env python3
"""Generate magnitude analysis figures for steering vectors.

Produces:
  1. Heatmap: persona x trait magnitude at a target layer
  2. Layer profile: magnitude vs layer (per-trait and per-persona curves)
  3. Bar charts: per-persona and per-trait mean magnitudes with error bars
  4. Scatter: magnitude vs cross-persona cosine similarity
  5. CV (coefficient of variation) bar chart across personas per trait

Usage:
    python scripts/magnitude_figures.py --vectors-dir outputs/gemma-2-27b-it/vectors --layer 25
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TRAIT_NAMES = [
    "assertiveness", "confidence", "deference", "empathy",
    "honesty", "impulsivity", "risk_taking", "warmth",
]

TRAIT_LABELS = {
    "assertiveness": "Assertiveness",
    "confidence": "Confidence",
    "deference": "Deference",
    "empathy": "Empathy",
    "honesty": "Honesty",
    "impulsivity": "Impulsivity",
    "risk_taking": "Risk-taking",
    "warmth": "Warmth",
}

PERSONA_LABELS = {
    "con_artist": "Con Artist",
    "drill_sergeant": "Drill Sergeant",
    "farmer": "Farmer",
    "kindergarten_teacher": "K. Teacher",
    "politician": "Politician",
    "professor": "Professor",
    "street_hustler": "Street Hustler",
    "surgeon": "Surgeon",
    "tech_ceo": "Tech CEO",
    "therapist": "Therapist",
}


def load_all_vectors(vectors_dir: Path) -> dict[tuple[str, str], torch.Tensor]:
    """Load all vectors, returning {(persona, trait): tensor(n_layers, hidden_dim)}."""
    data = {}
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        stem = pt_file.stem
        for t in TRAIT_NAMES:
            if stem.endswith(f"_{t}"):
                persona = stem[: -(len(t) + 1)]
                v = torch.load(pt_file, map_location="cpu", weights_only=False)
                data[(persona, t)] = v["vector"].float()
                break
    return data


def get_personas(data: dict) -> list[str]:
    return sorted(set(p for p, _ in data))


def persona_label(slug: str) -> str:
    return PERSONA_LABELS.get(slug, slug.replace("_", " ").title())


def trait_label(name: str) -> str:
    return TRAIT_LABELS.get(name, name.replace("_", " ").title())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Magnitude analysis figures")
    parser.add_argument("--vectors-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir for figures (default: sibling 'figures' dir)")
    return parser.parse_args()


def fig1_heatmap(data, personas, layer, output_dir):
    """Persona x trait magnitude heatmap."""
    matrix = np.zeros((len(personas), len(TRAIT_NAMES)))
    for i, p in enumerate(personas):
        for j, t in enumerate(TRAIT_NAMES):
            matrix[i, j] = data[(p, t)][layer].norm().item()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(
        matrix, annot=True, fmt=".0f",
        xticklabels=[trait_label(t) for t in TRAIT_NAMES],
        yticklabels=[persona_label(p) for p in personas],
        cmap="YlOrRd", ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Steering Vector Magnitude (Layer {layer})")
    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_heatmap.pdf")
    plt.savefig(output_dir / "magnitude_heatmap.png")
    plt.close()
    print(f"  Saved magnitude_heatmap.{{pdf,png}}")


def fig2_layer_profile(data, personas, output_dir):
    """Magnitude vs layer: per-trait and per-persona curves."""
    n_layers = next(iter(data.values())).shape[0]
    layers = list(range(n_layers))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: per-trait (averaged over personas)
    colors_trait = plt.cm.Set2(np.linspace(0, 0.8, len(TRAIT_NAMES)))
    for t, color in zip(TRAIT_NAMES, colors_trait):
        means = []
        for l in layers:
            norms = [data[(p, t)][l].norm().item() for p in personas]
            means.append(np.mean(norms))
        ax1.plot(layers, means, label=trait_label(t), color=color, linewidth=1.5)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean Magnitude")
    ax1.set_title("A) Magnitude by Trait (mean over personas)")
    ax1.legend(loc="upper left", frameon=True, fancybox=False, fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Panel B: per-persona (averaged over traits)
    colors_persona = plt.cm.tab10(np.linspace(0, 1, len(personas)))
    for p, color in zip(personas, colors_persona):
        means = []
        for l in layers:
            norms = [data[(p, t)][l].norm().item() for t in TRAIT_NAMES]
            means.append(np.mean(norms))
        ax2.plot(layers, means, label=persona_label(p), color=color, linewidth=1.5)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean Magnitude")
    ax2.set_title("B) Magnitude by Persona (mean over traits)")
    ax2.legend(loc="upper left", frameon=True, fancybox=False, fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_layer_profile.pdf")
    plt.savefig(output_dir / "magnitude_layer_profile.png")
    plt.close()
    print(f"  Saved magnitude_layer_profile.{{pdf,png}}")


def fig3_bar_charts(data, personas, layer, output_dir):
    """Per-persona and per-trait mean magnitudes with error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: per-persona
    persona_means, persona_stds = [], []
    for p in personas:
        mags = [data[(p, t)][layer].norm().item() for t in TRAIT_NAMES]
        persona_means.append(np.mean(mags))
        persona_stds.append(np.std(mags))

    # Sort by magnitude
    order = np.argsort(persona_means)[::-1]
    sorted_labels = [persona_label(personas[i]) for i in order]
    sorted_means = [persona_means[i] for i in order]
    sorted_stds = [persona_stds[i] for i in order]

    bars1 = ax1.barh(range(len(personas)), sorted_means, xerr=sorted_stds,
                     color="steelblue", edgecolor="white", capsize=3)
    ax1.set_yticks(range(len(personas)))
    ax1.set_yticklabels(sorted_labels)
    ax1.set_xlabel("Mean Magnitude")
    ax1.set_title(f"A) Per-Persona Magnitude (Layer {layer})")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.2, axis="x")

    # Panel B: per-trait
    trait_means, trait_stds = [], []
    for t in TRAIT_NAMES:
        mags = [data[(p, t)][layer].norm().item() for p in personas]
        trait_means.append(np.mean(mags))
        trait_stds.append(np.std(mags))

    order_t = np.argsort(trait_means)[::-1]
    sorted_t_labels = [trait_label(TRAIT_NAMES[i]) for i in order_t]
    sorted_t_means = [trait_means[i] for i in order_t]
    sorted_t_stds = [trait_stds[i] for i in order_t]

    bars2 = ax2.barh(range(len(TRAIT_NAMES)), sorted_t_means, xerr=sorted_t_stds,
                     color="indianred", edgecolor="white", capsize=3)
    ax2.set_yticks(range(len(TRAIT_NAMES)))
    ax2.set_yticklabels(sorted_t_labels)
    ax2.set_xlabel("Mean Magnitude")
    ax2.set_title(f"B) Per-Trait Magnitude (Layer {layer})")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_bars.pdf")
    plt.savefig(output_dir / "magnitude_bars.png")
    plt.close()
    print(f"  Saved magnitude_bars.{{pdf,png}}")


def fig4_magnitude_vs_cosine(data, personas, layer, output_dir):
    """Scatter: mean magnitude vs mean cross-persona cosine similarity per trait."""
    fig, ax = plt.subplots(figsize=(6, 5))

    trait_mag_means = []
    trait_cos_means = []
    trait_mag_stds = []
    trait_cos_stds = []

    for t in TRAIT_NAMES:
        mags = [data[(p, t)][layer].norm().item() for p in personas]
        trait_mag_means.append(np.mean(mags))
        trait_mag_stds.append(np.std(mags))

        cosines = []
        for p1, p2 in combinations(personas, 2):
            v1 = data[(p1, t)][layer]
            v2 = data[(p2, t)][layer]
            cos = torch.nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)
            ).item()
            cosines.append(cos)
        trait_cos_means.append(np.mean(cosines))
        trait_cos_stds.append(np.std(cosines))

    colors = plt.cm.Set2(np.linspace(0, 0.8, len(TRAIT_NAMES)))
    for i, t in enumerate(TRAIT_NAMES):
        ax.errorbar(
            trait_mag_means[i], trait_cos_means[i],
            xerr=trait_mag_stds[i], yerr=trait_cos_stds[i],
            fmt="o", color=colors[i], markersize=10, capsize=3,
            label=trait_label(t), markeredgecolor="white", markeredgewidth=0.5,
        )

    ax.set_xlabel("Mean Magnitude (across personas)")
    ax.set_ylabel("Mean Pairwise Cosine Similarity")
    ax.set_title(f"Magnitude vs Directional Consistency (Layer {layer})")
    ax.legend(frameon=True, fancybox=False)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_vs_cosine.pdf")
    plt.savefig(output_dir / "magnitude_vs_cosine.png")
    plt.close()
    print(f"  Saved magnitude_vs_cosine.{{pdf,png}}")


def fig5_cv_bars(data, personas, layer, output_dir):
    """Coefficient of variation of magnitude across personas, per trait."""
    fig, ax = plt.subplots(figsize=(7, 4))

    cvs = []
    for t in TRAIT_NAMES:
        norms = [data[(p, t)][layer].norm().item() for p in personas]
        cv = np.std(norms) / np.mean(norms) if np.mean(norms) > 0 else 0
        cvs.append(cv)

    order = np.argsort(cvs)[::-1]
    sorted_labels = [trait_label(TRAIT_NAMES[i]) for i in order]
    sorted_cvs = [cvs[i] for i in order]

    bars = ax.bar(range(len(TRAIT_NAMES)), sorted_cvs,
                  color="mediumpurple", edgecolor="white")
    ax.set_xticks(range(len(TRAIT_NAMES)))
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right")
    ax.set_ylabel("Coefficient of Variation (std/mean)")
    ax.set_title(f"Magnitude Variability Across Personas (Layer {layer})")
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate values
    for i, (bar, cv) in enumerate(zip(bars, sorted_cvs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{cv:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_cv.pdf")
    plt.savefig(output_dir / "magnitude_cv.png")
    plt.close()
    print(f"  Saved magnitude_cv.{{pdf,png}}")


def main():
    args = parse_args()
    vectors_dir = Path(args.vectors_dir)
    layer = args.layer

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = vectors_dir.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading vectors from {vectors_dir}...")
    data = load_all_vectors(vectors_dir)
    personas = get_personas(data)
    n_layers = next(iter(data.values())).shape[0]
    print(f"  {len(data)} vectors, {len(personas)} personas, {n_layers} layers")
    print(f"  Target layer: {layer}")
    print(f"  Output: {output_dir}")
    print()

    print("Generating figures...")
    fig1_heatmap(data, personas, layer, output_dir)
    fig2_layer_profile(data, personas, output_dir)
    fig3_bar_charts(data, personas, layer, output_dir)
    fig4_magnitude_vs_cosine(data, personas, layer, output_dir)
    fig5_cv_bars(data, personas, layer, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
