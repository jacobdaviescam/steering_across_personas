#!/usr/bin/env python3
"""General direction cluster bias and leave-one-out analysis.

Extends r4 with statistical tests for distance asymmetry, leave-one-out
sensitivity, and (at scale) cluster-centroid bias.

Usage:
    python pipeline/r8_cluster_bias.py \\
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

from persona_steering.config import Trait, OUTPUTS_DIR, TARGET_LAYER
from persona_steering.analysis import build_transfer_matrix, cluster_persona_vectors
from persona_steering.utils import (
    log, save_json, save_fig, cosine_similarity, VectorShim,
    parse_persona_trait_from_stem, load_vectors,
)
from persona_steering.wandb_utils import init_run, finish_run, log_summary, log_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General direction cluster bias analysis")
    p.add_argument("--vectors-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    short = vectors_dir.parent.name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else OUTPUTS_DIR / short / "robustness" / "cluster_bias"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    # Load all vectors
    vectors = load_vectors(vectors_dir, layer)

    if not vectors:
        log.error("No vectors loaded")
        return

    init_run("r8_cluster_bias", short, config=vars(args))

    personas = sorted({p for p, _ in vectors})
    traits = sorted({t for _, t in vectors})
    log.info(
        "Loaded %d vectors: %d personas, %d traits",
        len(vectors), len(personas), len(traits),
    )

    # ------------------------------------------------------------------
    # 1. Distance asymmetry
    # ------------------------------------------------------------------
    distance_asymmetry: dict[str, dict] = {}
    for trait in traits:
        vecs = [vectors[(p, trait)] for p in personas if (p, trait) in vectors]
        if not vecs:
            continue
        general = torch.stack(vecs).mean(dim=0)
        cosines = {
            p: cosine_similarity(vectors[(p, trait)], general)
            for p in personas
            if (p, trait) in vectors
        }
        vals = list(cosines.values())
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        cv = std_val / mean_val if mean_val != 0 else 0.0
        min_persona = min(cosines, key=cosines.get)
        max_persona = max(cosines, key=cosines.get)
        distance_asymmetry[trait] = {
            "cv": cv,
            "mean": mean_val,
            "std": std_val,
            "min_persona": min_persona,
            "min_cosine": cosines[min_persona],
            "max_persona": max_persona,
            "max_cosine": cosines[max_persona],
        }
    save_json(distance_asymmetry, output_dir / "distance_asymmetry.json")

    # ------------------------------------------------------------------
    # 2. Leave-one-out sensitivity
    # ------------------------------------------------------------------
    loo_results: dict[str, dict[str, dict]] = {}
    for trait in traits:
        available = [p for p in personas if (p, trait) in vectors]
        if len(available) < 2:
            continue
        full_vecs = [vectors[(p, trait)] for p in available]
        general_full = torch.stack(full_vecs).mean(dim=0)
        trait_loo: dict[str, dict] = {}
        for p_leave in available:
            others = [vectors[(p, trait)] for p in available if p != p_leave]
            if not others:
                continue
            general_loo = torch.stack(others).mean(dim=0)
            shift = cosine_similarity(general_loo, general_full)
            # How much does LOO change the cosine of the left-out persona to general?
            cos_to_full = cosine_similarity(vectors[(p_leave, trait)], general_full)
            cos_to_loo = cosine_similarity(vectors[(p_leave, trait)], general_loo)
            trait_loo[p_leave] = {
                "shift_cosine": float(shift),
                "influence_score": float(1.0 - shift),
                "cosine_to_full": float(cos_to_full),
                "cosine_to_loo": float(cos_to_loo),
                "cosine_delta": float(cos_to_loo - cos_to_full),
            }
        loo_results[trait] = trait_loo
    save_json(loo_results, output_dir / "leave_one_out.json")

    # ------------------------------------------------------------------
    # 3. Weighted vs unweighted general
    # ------------------------------------------------------------------
    weighted_comparison: dict[str, dict] = {}
    for trait in traits:
        vecs_list = [vectors[(p, trait)] for p in personas if (p, trait) in vectors]
        if not vecs_list:
            continue
        simple_mean = torch.stack(vecs_list).mean(dim=0)
        # Inverse-norm weighting (downweight outlier magnitudes)
        norms = torch.tensor([v.norm().item() for v in vecs_list])
        weights = 1.0 / (norms + 1e-8)
        weights = weights / weights.sum()
        weighted_mean = sum(w * v for w, v in zip(weights, vecs_list))
        cos_diff = cosine_similarity(simple_mean, weighted_mean)
        weighted_comparison[trait] = {
            "cosine_simple_vs_weighted": float(cos_diff),
            "norm_cv": float(norms.std().item() / (norms.mean().item() + 1e-8)),
        }
    save_json(weighted_comparison, output_dir / "weighted_comparison.json")

    # ------------------------------------------------------------------
    # 4. Cluster-centroid bias (conditional on N >= 20)
    # ------------------------------------------------------------------
    if len(personas) >= 20:
        # Build transfer matrix and cluster for centroid analysis
        nested: dict[str, dict[Trait, dict[int, VectorShim]]] = {}
        for (persona, trait_str), vec in vectors.items():
            shim = VectorShim(vec, persona, Trait(trait_str), layer)
            nested.setdefault(persona, {}).setdefault(Trait(trait_str), {})[layer] = shim

        trait_enums = [Trait(t) for t in traits]
        tm = build_transfer_matrix(nested, personas, trait_enums, layer)
        clusters = cluster_persona_vectors(tm, personas)["clusters"]

        cluster_centroid_bias: dict[str, dict] = {}
        for trait in traits:
            available = [p for p in personas if (p, trait) in vectors]
            if not available:
                continue
            general = torch.stack([vectors[(p, trait)] for p in available]).mean(dim=0)
            per_cluster: dict[str, dict] = {}
            for cid, members in clusters.items():
                member_vecs = [
                    vectors[(p, trait)] for p in members if (p, trait) in vectors
                ]
                if not member_vecs:
                    continue
                centroid = torch.stack(member_vecs).mean(dim=0)
                per_cluster[str(cid)] = {
                    "members": members,
                    "centroid_cosine_to_general": float(
                        cosine_similarity(general, centroid)
                    ),
                    "n_members": len(member_vecs),
                }
            cluster_centroid_bias[trait] = per_cluster
        save_json(cluster_centroid_bias, output_dir / "cluster_centroid_bias.json")
    else:
        log.info(
            "Skipping cluster-centroid bias (need >= 20 personas, have %d)",
            len(personas),
        )

    # ------------------------------------------------------------------
    # W&B summary
    # ------------------------------------------------------------------
    summary_metrics = {}
    for t, da in distance_asymmetry.items():
        summary_metrics[f"cluster_bias/{t}/cv"] = da["cv"]
    for t, wc in weighted_comparison.items():
        summary_metrics[f"cluster_bias/{t}/cos_simple_vs_weighted"] = wc[
            "cosine_simple_vs_weighted"
        ]
    log_summary(summary_metrics)

    # ------------------------------------------------------------------
    # Figure 1: Leave-one-out heatmap
    # ------------------------------------------------------------------
    if loo_results:
        loo_traits = sorted(loo_results.keys())
        loo_personas = sorted(
            {p for t_data in loo_results.values() for p in t_data}
        )
        matrix = np.full((len(loo_personas), len(loo_traits)), np.nan)
        for ti, trait in enumerate(loo_traits):
            for pi, persona in enumerate(loo_personas):
                if persona in loo_results[trait]:
                    matrix[pi, ti] = loo_results[trait][persona]["influence_score"]

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(loo_traits)))
        ax.set_xticklabels(
            [t.replace("_", " ").title() for t in loo_traits],
            rotation=45, ha="right", fontsize=9,
        )
        ax.set_yticks(range(len(loo_personas)))
        ax.set_yticklabels(
            [p.replace("_", " ").title() for p in loo_personas], fontsize=9,
        )
        for i in range(len(loo_personas)):
            for j in range(len(loo_traits)):
                if not np.isnan(matrix[i, j]):
                    ax.text(
                        j, i, f"{matrix[i, j]:.4f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if matrix[i, j] > 0.5 * np.nanmax(matrix) else "black",
                    )
        plt.colorbar(im, ax=ax, label="Influence (1 - cos(LOO, full))", shrink=0.8)
        ax.set_title("Leave-One-Out Sensitivity: Influence on General Direction")
        fig.tight_layout()
        save_fig(fig, output_dir / "leave_one_out_heatmap.png")

    # ------------------------------------------------------------------
    # Figure 2: Distance asymmetry bar chart
    # ------------------------------------------------------------------
    if distance_asymmetry:
        sorted_traits = sorted(
            distance_asymmetry, key=lambda t: distance_asymmetry[t]["cv"],
        )
        cvs = [distance_asymmetry[t]["cv"] for t in sorted_traits]
        max_cv = max(cvs) if cvs else 1.0

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.YlOrRd([c / (max_cv + 1e-8) for c in cvs])
        ax.barh(range(len(sorted_traits)), cvs, color=colors, alpha=0.85)
        ax.set_yticks(range(len(sorted_traits)))
        ax.set_yticklabels([t.replace("_", " ").title() for t in sorted_traits])
        ax.set_xlabel("Coefficient of Variation (cosines to general)")
        ax.set_title("Distance Asymmetry: How Uniform Are Persona Distances to General?")
        for i, t in enumerate(sorted_traits):
            da = distance_asymmetry[t]
            ax.text(
                cvs[i] + 0.002, i,
                f"min: {da['min_persona'].replace('_', ' ')}",
                fontsize=7, va="center", color="gray",
            )
        fig.tight_layout()
        save_fig(fig, output_dir / "distance_asymmetry.png")

    # ------------------------------------------------------------------
    # Figure 3: Influence ranking (mean across traits)
    # ------------------------------------------------------------------
    if loo_results:
        persona_influence: dict[str, list[float]] = {}
        for trait, t_data in loo_results.items():
            for persona, p_data in t_data.items():
                persona_influence.setdefault(persona, []).append(
                    p_data["influence_score"]
                )
        mean_influence = {
            p: float(np.mean(scores)) for p, scores in persona_influence.items()
        }
        ranked = sorted(mean_influence, key=mean_influence.get, reverse=True)
        vals = [mean_influence[p] for p in ranked]

        fig, ax = plt.subplots(figsize=(8, 5))
        bar_colors = plt.cm.YlOrRd(
            [v / (max(vals) + 1e-8) for v in vals]
        )
        ax.barh(range(len(ranked)), vals, color=bar_colors, alpha=0.85)
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels(
            [p.replace("_", " ").title() for p in ranked], fontsize=9,
        )
        ax.set_xlabel("Mean Influence Score (1 - cos(LOO, full), averaged over traits)")
        ax.set_title("Persona Influence Ranking: Who Shifts the General Direction Most?")
        ax.invert_yaxis()
        fig.tight_layout()
        save_fig(fig, output_dir / "influence_ranking.png")

    log_images(output_dir, prefix="r8_cluster_bias")
    finish_run()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=== Cluster Bias Summary ===")
    for trait in sorted(distance_asymmetry, key=lambda t: distance_asymmetry[t]["cv"], reverse=True):
        da = distance_asymmetry[trait]
        # Find most influential persona for this trait
        most_influential = ""
        if trait in loo_results:
            most_influential = max(
                loo_results[trait],
                key=lambda p: loo_results[trait][p]["influence_score"],
            )
        log.info(
            "  %-15s: CV=%.4f  most influential: %s",
            trait, da["cv"], most_influential,
        )
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
