#!/usr/bin/env python3
"""Safety-trait context dependence analysis.

Compares context-dependence of safety traits (refusal, deceptiveness,
power-seeking, sycophancy) vs behavioral traits to assess whether
universal safety steering vectors are viable.

Requires R5 (safety traits) to have been run. Exits gracefully if
safety trait vectors are not found.

Usage:
    python pipeline/r9_safety_context.py \\
        --vectors-dir outputs/gemma-2-27b-it/vectors --layer 22
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu

from persona_steering.config import Trait, OUTPUTS_DIR, TARGET_LAYER
from persona_steering.utils import (
    log, save_json, save_fig, cosine_similarity,
    parse_persona_trait_from_stem, load_vectors,
)
from persona_steering.wandb_utils import init_run, finish_run, log_summary, log_images


SAFETY_TRAITS = {"refusal", "deceptiveness", "power_seeking", "sycophancy"}
BEHAVIORAL_TRAITS = {"assertiveness", "empathy", "risk_taking", "honesty",
                     "confidence", "warmth", "impulsivity"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safety-trait context dependence")
    p.add_argument("--vectors-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    short = vectors_dir.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "robustness" / "safety_context"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    # Load all vectors
    vectors = load_vectors(vectors_dir, layer)

    if not vectors:
        log.error("No vectors loaded")
        return

    all_traits = sorted({t for _, t in vectors})
    safety_found = [t for t in all_traits if t in SAFETY_TRAITS]
    behavioral_found = [t for t in all_traits if t in BEHAVIORAL_TRAITS]

    if not safety_found:
        log.warning("No safety trait vectors found. Run R5 first to add safety traits.")
        log.warning("Available traits: %s", all_traits)
        return

    personas = sorted({p for p, _ in vectors})
    log.info("Loaded %d vectors: %d personas, %d traits (%d safety, %d behavioral)",
             len(vectors), len(personas), len(all_traits), len(safety_found), len(behavioral_found))

    init_run("r9_safety_context", short, config=vars(args))

    # ------------------------------------------------------------------
    # 1. General vector per trait
    # ------------------------------------------------------------------
    general: dict[str, torch.Tensor] = {}
    for trait in all_traits:
        vecs = [vectors[(p, trait)] for p in personas if (p, trait) in vectors]
        if vecs:
            general[trait] = torch.stack(vecs).mean(dim=0)

    # ------------------------------------------------------------------
    # 2. Context dependence per trait (cosine to general)
    # ------------------------------------------------------------------
    context_dep: dict[str, dict[str, float]] = {}
    for trait in all_traits:
        if trait not in general:
            continue
        cosines = {p: cosine_similarity(vectors[(p, trait)], general[trait])
                   for p in personas if (p, trait) in vectors}
        context_dep[trait] = cosines

    # ------------------------------------------------------------------
    # 3. Safety vs behavioral comparison (Mann-Whitney U)
    # ------------------------------------------------------------------
    safety_means = [np.mean(list(context_dep[t].values())) for t in safety_found if t in context_dep]
    behavioral_means = [np.mean(list(context_dep[t].values())) for t in behavioral_found if t in context_dep]

    safety_all_cosines = [c for t in safety_found for c in context_dep.get(t, {}).values()]
    behavioral_all_cosines = [c for t in behavioral_found for c in context_dep.get(t, {}).values()]

    if safety_all_cosines and behavioral_all_cosines:
        stat, p_val = mannwhitneyu(safety_all_cosines, behavioral_all_cosines, alternative="two-sided")
    else:
        stat, p_val = float("nan"), float("nan")

    comparison = {
        "safety_traits": safety_found,
        "behavioral_traits": behavioral_found,
        "safety_mean_cos_to_general": float(np.mean(safety_means)) if safety_means else None,
        "behavioral_mean_cos_to_general": float(np.mean(behavioral_means)) if behavioral_means else None,
        "safety_per_trait": {t: float(np.mean(list(context_dep[t].values()))) for t in safety_found if t in context_dep},
        "behavioral_per_trait": {t: float(np.mean(list(context_dep[t].values()))) for t in behavioral_found if t in context_dep},
        "mann_whitney_U": float(stat),
        "mann_whitney_p": float(p_val),
        "n_safety_cosines": len(safety_all_cosines),
        "n_behavioral_cosines": len(behavioral_all_cosines),
    }
    save_json(comparison, output_dir / "safety_vs_behavioral.json")

    # ------------------------------------------------------------------
    # 4. Most-distorting contexts per safety trait
    # ------------------------------------------------------------------
    most_distorting: dict[str, list[dict]] = {}
    for trait in safety_found:
        if trait not in context_dep:
            continue
        ranked = sorted(context_dep[trait].items(), key=lambda kv: kv[1])
        most_distorting[trait] = [
            {"persona": p, "cosine_to_general": float(c), "deviation": float(1.0 - c)}
            for p, c in ranked
        ]
    save_json(most_distorting, output_dir / "most_distorting_contexts.json")

    # ------------------------------------------------------------------
    # 5. Safety transfer matrix per safety trait
    # ------------------------------------------------------------------
    def _trait_transfer_stats(trait: str) -> dict:
        """Build per-trait persona similarity matrix and compute transfer stats."""
        n = len(personas)
        matrix = np.zeros((n, n))
        for i, pa in enumerate(personas):
            for j, pb in enumerate(personas):
                if (pa, trait) in vectors and (pb, trait) in vectors:
                    matrix[i, j] = cosine_similarity(vectors[(pa, trait)], vectors[(pb, trait)])
        off_diag = [matrix[i, j] for i in range(n) for j in range(n) if i != j]
        diag_vals = [matrix[i, i] for i in range(n)]
        return {
            "matrix": [[float(x) for x in row] for row in matrix],
            "mean_off_diagonal": float(np.mean(off_diag)) if off_diag else None,
            "std_off_diagonal": float(np.std(off_diag)) if off_diag else None,
            "mean_diagonal": float(np.mean(diag_vals)) if diag_vals else None,
            "diagonal_dominance": float(np.mean(diag_vals) - np.mean(off_diag)) if off_diag else None,
            "personas": personas,
        }

    safety_transfer = {t: _trait_transfer_stats(t) for t in safety_found}
    save_json(safety_transfer, output_dir / "safety_transfer.json")

    # Also compute behavioral diagonal dominance for comparison
    behavioral_diag_dom = [
        _trait_transfer_stats(t)["diagonal_dominance"]
        for t in behavioral_found
        if t in context_dep and _trait_transfer_stats(t)["diagonal_dominance"] is not None
    ]

    # ------------------------------------------------------------------
    # 6. Cross-trait safety correlation
    # ------------------------------------------------------------------
    cross_corr: dict = {}
    if len(safety_found) >= 2:
        deviation_matrix = np.zeros((len(personas), len(safety_found)))
        for pi, p in enumerate(personas):
            for ti, t in enumerate(safety_found):
                deviation_matrix[pi, ti] = 1.0 - context_dep.get(t, {}).get(p, 1.0)

        # Correlation matrix across safety traits
        # Handle constant columns gracefully
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(deviation_matrix.T)  # (n_safety, n_safety)
        corr = np.nan_to_num(corr, nan=0.0)

        cross_corr = {
            "safety_traits": safety_found,
            "correlation_matrix": [[float(x) for x in row] for row in corr],
            "mean_off_diagonal_corr": float(np.mean([
                corr[i, j] for i in range(len(safety_found))
                for j in range(len(safety_found)) if i != j
            ])) if len(safety_found) > 1 else None,
            "deviation_matrix": {
                "personas": personas,
                "traits": safety_found,
                "values": [[float(x) for x in row] for row in deviation_matrix],
            },
        }
    save_json(cross_corr, output_dir / "cross_trait_correlation.json")

    # ------------------------------------------------------------------
    # Figure 1: Safety vs behavioral grouped bar chart
    # ------------------------------------------------------------------
    all_plot_traits = safety_found + behavioral_found
    all_plot_means = []
    all_plot_colors = []
    for t in all_plot_traits:
        if t in context_dep:
            all_plot_means.append(float(np.mean(list(context_dep[t].values()))))
        else:
            all_plot_means.append(0.0)
        all_plot_colors.append("#C44E52" if t in SAFETY_TRAITS else "#4C72B0")

    fig, ax = plt.subplots(figsize=(8, max(5, len(all_plot_traits) * 0.5)))
    y_pos = range(len(all_plot_traits))
    ax.barh(y_pos, all_plot_means, color=all_plot_colors, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([t.replace("_", " ").title() for t in all_plot_traits], fontsize=9)
    ax.set_xlabel("Mean Cosine to General Vector")
    ax.set_title("Context-Dependence: Safety (red) vs Behavioral (blue) Traits")
    overall_mean = float(np.mean(all_plot_means)) if all_plot_means else 0.0
    ax.axvline(overall_mean, color="gray", ls=":", alpha=0.7, label=f"Overall mean={overall_mean:.3f}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    # Annotate p-value
    ax.text(0.02, 0.02, f"Mann-Whitney p={p_val:.4f}", transform=ax.transAxes,
            fontsize=8, color="gray")
    fig.tight_layout()
    save_fig(fig, output_dir / "safety_vs_behavioral.png")

    # ------------------------------------------------------------------
    # Figure 2: Safety transfer matrices (one heatmap per safety trait)
    # ------------------------------------------------------------------
    n_safety = len(safety_found)
    if n_safety > 0:
        cols = min(n_safety, 2)
        rows = (n_safety + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
        for idx, trait in enumerate(safety_found):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            mat = np.array(safety_transfer[trait]["matrix"])
            im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="equal")
            labels = [p.replace("_", " ").title() for p in personas]
            ax.set_xticks(range(len(personas)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(personas)))
            ax.set_yticklabels(labels, fontsize=7)
            for i in range(len(personas)):
                for j in range(len(personas)):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if mat[i, j] < 0.3 else "black")
            ax.set_title(trait.replace("_", " ").title(), fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Hide unused subplots
        for idx in range(n_safety, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        fig.suptitle("Safety Trait Transfer Matrices (Persona x Persona)", fontsize=12)
        fig.tight_layout()
        save_fig(fig, output_dir / "safety_transfer_matrices.png")

    # ------------------------------------------------------------------
    # Figure 3: Cross-trait safety correlation heatmap
    # ------------------------------------------------------------------
    if len(safety_found) >= 2 and cross_corr:
        corr_mat = np.array(cross_corr["correlation_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        labels = [t.replace("_", " ").title() for t in safety_found]
        ax.set_xticks(range(len(safety_found)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(safety_found)))
        ax.set_yticklabels(labels, fontsize=9)
        for i in range(len(safety_found)):
            for j in range(len(safety_found)):
                ax.text(j, i, f"{corr_mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(corr_mat[i, j]) > 0.6 else "black")
        plt.colorbar(im, ax=ax, label="Pearson Correlation", shrink=0.8)
        ax.set_title("Cross-Trait Safety Correlation\n(deviation profiles across personas)")
        fig.tight_layout()
        save_fig(fig, output_dir / "cross_trait_correlation.png")

    # ------------------------------------------------------------------
    # Figure 4: Most-distorting contexts per safety trait
    # ------------------------------------------------------------------
    if most_distorting:
        n_panels = len(most_distorting)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, max(4, len(personas) * 0.4)),
                                 squeeze=False)
        for idx, (trait, entries) in enumerate(sorted(most_distorting.items())):
            ax = axes[0][idx]
            p_names = [e["persona"].replace("_", " ").title() for e in entries]
            deviations = [e["deviation"] for e in entries]
            colors = ["#C44E52" if d > 0.2 else "#55A868" if d < 0.05 else "#4C72B0" for d in deviations]
            ax.barh(range(len(p_names)), deviations, color=colors, alpha=0.85)
            ax.set_yticks(range(len(p_names)))
            ax.set_yticklabels(p_names, fontsize=8)
            ax.set_xlabel("Deviation (1 - cosine)")
            ax.set_title(trait.replace("_", " ").title(), fontsize=10)
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle("Most-Distorting Contexts per Safety Trait", fontsize=12)
        fig.tight_layout()
        save_fig(fig, output_dir / "most_distorting.png")

    # ------------------------------------------------------------------
    # W&B logging and summary
    # ------------------------------------------------------------------
    summary_metrics: dict[str, float] = {}
    if safety_means:
        summary_metrics["r9/safety_mean_cos"] = float(np.mean(safety_means))
    if behavioral_means:
        summary_metrics["r9/behavioral_mean_cos"] = float(np.mean(behavioral_means))
    summary_metrics["r9/mann_whitney_p"] = float(p_val)
    for t in safety_found:
        if t in context_dep:
            summary_metrics[f"r9/{t}/mean_cos"] = float(np.mean(list(context_dep[t].values())))
        if t in safety_transfer:
            summary_metrics[f"r9/{t}/transferability"] = safety_transfer[t]["mean_off_diagonal"] or 0.0
    log_summary(summary_metrics)
    log_images(output_dir, prefix="r9_safety_context")
    finish_run()

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    log.info("=== Safety Context Dependence Summary ===")
    log.info("Safety traits found: %s", safety_found)
    log.info("Safety mean cos-to-general: %.4f", np.mean(safety_means) if safety_means else 0.0)
    log.info("Behavioral mean cos-to-general: %.4f", np.mean(behavioral_means) if behavioral_means else 0.0)
    log.info("Mann-Whitney p-value: %.4f", p_val)
    for t in safety_found:
        if t in context_dep:
            vals = list(context_dep[t].values())
            most_diff = min(context_dep[t], key=context_dep[t].get)
            log.info("  %-20s: cos=%.4f +/- %.4f (most diff: %s)", t, np.mean(vals), np.std(vals), most_diff)
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
