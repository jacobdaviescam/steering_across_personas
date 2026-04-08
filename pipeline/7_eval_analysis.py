#!/usr/bin/env python3
"""Analyse behavioural evaluation results and correlate with geometric analysis.

Loads behavioral_scores.json from step 6 and geometric decomposition from step 4.
Produces figures showing cross-persona trait expression, effect sizes, and
geometry-behaviour correlation.

Usage:
    python pipeline/7_eval_analysis.py \
      --eval-dir outputs/gemma-2-27b-it/eval \
      --analysis-dir outputs/gemma-2-27b-it/analysis \
      --output-dir outputs/gemma-2-27b-it/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from persona_steering.config import Trait, ANALYSIS_SUBDIR, FIGURES_SUBDIR
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



# ---------------------------------------------------------------------------
# Figure 1: Cross-persona trait expression (grouped bar)
# ---------------------------------------------------------------------------

def fig_cross_persona_expression(scores: dict, output_dir: Path) -> None:
    """For each trait, compare pos_mean scores across personas."""
    personas = sorted(scores.keys())
    all_traits = set()
    for p in personas:
        all_traits.update(scores[p].keys())
    traits = sorted(all_traits)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for ti, trait in enumerate(traits):
        if ti >= len(axes):
            break
        ax = axes[ti]
        slugs = [p for p in personas if trait in scores[p] and "pos_mean" in scores[p][trait]]
        pos_means = [scores[p][trait]["pos_mean"] for p in slugs]
        neg_means = [scores[p][trait].get("neg_mean", 0) for p in slugs]

        x = np.arange(len(slugs))
        width = 0.35
        ax.bar(x - width / 2, pos_means, width, label="Positive", color="#55A868")
        ax.bar(x + width / 2, neg_means, width, label="Negative", color="#C44E52")

        ax.set_xticks(x)
        ax.set_xticklabels([pretty_persona(p) for p in slugs], rotation=90, fontsize=7)
        ax.set_title(pretty_trait(trait), fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Mean Score")
        if ti == 0:
            ax.legend(fontsize=8)

    for ti in range(len(traits), len(axes)):
        axes[ti].set_visible(False)

    fig.suptitle("Cross-Persona Trait Expression (Pos vs Neg Instruction)", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir / "cross_persona_expression.png")


# ---------------------------------------------------------------------------
# Figure 2: Effect size heatmap
# ---------------------------------------------------------------------------

def fig_effect_size_heatmap(scores: dict, output_dir: Path) -> None:
    """Heatmap of effect_size (pos_mean - neg_mean) per persona x trait."""
    personas = sorted(scores.keys())
    all_traits = set()
    for p in personas:
        all_traits.update(scores[p].keys())
    traits = sorted(all_traits)

    matrix = np.full((len(personas), len(traits)), np.nan)
    for pi, persona in enumerate(personas):
        for ti, trait in enumerate(traits):
            entry = scores.get(persona, {}).get(trait, {})
            if "effect_size" in entry:
                matrix[pi, ti] = entry["effect_size"]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([pretty_trait(t) for t in traits], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels([pretty_persona(p) for p in personas], fontsize=9)

    for i in range(len(personas)):
        for j in range(len(traits)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val < 0.3 else "black")

    fig.colorbar(im, ax=ax, label="Effect Size (pos - neg)", shrink=0.8)
    ax.set_title("Behavioural Effect Size: Persona x Trait", fontsize=12)
    save_fig(fig, output_dir / "effect_size_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 3: Geometry vs behaviour correlation
# ---------------------------------------------------------------------------

def fig_geometry_vs_behaviour(
    scores: dict,
    decomp: dict,
    output_dir: Path,
) -> None:
    """Scatter: persona-specific magnitude vs behavioural divergence from mean."""
    personas = sorted(scores.keys())
    all_traits = set()
    for p in personas:
        all_traits.update(scores[p].keys())
    traits = sorted(all_traits)

    # Compute per-trait mean effect size, then each persona's deviation from it
    geo_x = []  # specific_magnitude / (shared + specific)
    behav_y = []  # |persona_effect - mean_effect|
    labels = []

    for trait in traits:
        if trait not in decomp:
            continue

        d = decomp[trait]
        shared_mags = d.get("shared_magnitudes", {})
        specific_mags = d.get("specific_magnitudes", {})

        # Behavioural mean effect across personas for this trait
        effects = []
        for persona in personas:
            entry = scores.get(persona, {}).get(trait, {})
            if "effect_size" in entry:
                effects.append((persona, entry["effect_size"]))

        if len(effects) < 2:
            continue

        mean_effect = np.mean([e for _, e in effects])

        for persona, effect in effects:
            shared = abs(shared_mags.get(persona, 0))
            specific = specific_mags.get(persona, 0)
            total = shared + specific
            if total == 0:
                continue

            geo_ratio = specific / total
            behav_dev = abs(effect - mean_effect)

            geo_x.append(geo_ratio)
            behav_y.append(behav_dev)
            labels.append(f"{pretty_persona(persona)}\n{pretty_trait(trait)}")

    if not geo_x:
        log.warning("No data for geometry-behaviour correlation")
        return

    geo_x = np.array(geo_x)
    behav_y = np.array(behav_y)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(geo_x, behav_y, alpha=0.6, s=40, c="#4C72B0", edgecolors="white", linewidth=0.5)

    # Fit line
    if len(geo_x) > 2:
        coeffs = np.polyfit(geo_x, behav_y, 1)
        fit_x = np.linspace(geo_x.min(), geo_x.max(), 100)
        fit_y = np.polyval(coeffs, fit_x)
        ax.plot(fit_x, fit_y, "r--", alpha=0.7, linewidth=1.5)

        # Correlation
        corr = np.corrcoef(geo_x, behav_y)[0, 1]
        ax.text(0.95, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Geometric Specificity (specific / total magnitude)")
    ax.set_ylabel("Behavioural Divergence (|effect - mean_effect|)")
    ax.set_title("Geometry vs Behaviour: Do Geometric Differences Predict Behavioural Ones?",
                 fontsize=11)
    ax.grid(alpha=0.3)
    save_fig(fig, output_dir / "geometry_vs_behaviour.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse behavioural eval results")
    parser.add_argument("--eval-dir", type=str, required=True,
                        help="Directory with behavioral_scores.json from step 6")
    parser.add_argument("--analysis-dir", type=str, default=None,
                        help="Directory with geometric analysis from step 4 (for correlation)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: sibling 'figures' dir)")
    return parser.parse_args()


def main() -> None:
    from persona_steering.wandb_utils import init_run, finish_run, log_images, log_artifact, ensure_dir

    args = parse_args()
    eval_dir = Path(args.eval_dir)
    short = eval_dir.parent.name
    eval_dir = ensure_dir(f"{short}-eval", eval_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else eval_dir.parent / ANALYSIS_SUBDIR
    analysis_dir = ensure_dir(f"{short}-analysis", analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir.parent / FIGURES_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = eval_dir / "behavioral_scores.json"
    if not scores_path.exists():
        log.error("behavioral_scores.json not found in %s", eval_dir)
        return

    scores = load_json(scores_path)
    log.info("Loaded scores for %d personas", len(scores))

    init_run("step7_eval_analysis", short)

    # Figure 1: Cross-persona trait expression
    fig_cross_persona_expression(scores, output_dir)

    # Figure 2: Effect size heatmap
    fig_effect_size_heatmap(scores, output_dir)

    # Figure 3: Geometry-behaviour correlation (needs decomposition)
    decomp_path = analysis_dir / "decomposition.json"
    if decomp_path.exists():
        decomp = load_json(decomp_path)
        fig_geometry_vs_behaviour(scores, decomp, output_dir)
    else:
        log.warning("No decomposition.json found at %s, skipping geometry-behaviour figure",
                    decomp_path)

    log.info("All eval figures saved to %s", output_dir)
    for f in sorted(output_dir.glob("*.png")):
        log.info("  %s", f.name)

    log_images(output_dir, prefix="eval")
    finish_run()


if __name__ == "__main__":
    main()
