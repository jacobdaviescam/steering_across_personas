#!/usr/bin/env python3
"""Syntactic invariance: are representations driven by meaning or phrasing?

Computes a separate steering vector from each of the 5 instruction variants,
then measures cross-variant similarity.  High similarity = the model's
representation is driven by semantic content, not surface phrasing.

Also compares: within-persona cross-variant sim vs across-persona same-variant sim.

Usage:
    python pipeline/r3_syntactic_invariance.py \
        --activations-dir outputs/gemma-2-27b-it/activations --layer 22
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from persona_steering.config import Trait, OUTPUTS_DIR, TARGET_LAYER
from persona_steering.utils import (
    log, save_json, save_fig, cosine_similarity,
    discover_activation_pairs, parse_persona_trait_from_stem,
)
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_summary, log_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Syntactic invariance analysis")
    p.add_argument("--activations-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    return p.parse_args()


def split_by_variant(data: dict[str, torch.Tensor]) -> dict[int, list[torch.Tensor]]:
    """Split activation dict by variant index (keys like 'v0_q3')."""
    by_variant: dict[int, list[torch.Tensor]] = defaultdict(list)
    for key, tensor in data.items():
        m = re.match(r"v(\d+)_q\d+", key)
        if m:
            by_variant[int(m.group(1))].append(tensor)
    return dict(by_variant)


def compute_variant_vector(
    pos_variant: list[torch.Tensor], neg_variant: list[torch.Tensor],
) -> torch.Tensor | None:
    """Contrastive vector from a single variant's activations."""
    if not pos_variant or not neg_variant:
        return None
    _clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    pos_sum = sum(_clean(v[:-1].float()) for v in pos_variant)
    neg_sum = sum(_clean(v[:-1].float()) for v in neg_variant)
    return (pos_sum / len(pos_variant)) - (neg_sum / len(neg_variant))


def main() -> None:
    args = parse_args()

    activations_dir = Path(args.activations_dir)
    short = activations_dir.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "robustness" / "syntactic"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    pairs = discover_activation_pairs(activations_dir)
    if not pairs:
        log.error("No activation pairs found")
        return

    init_run("r3_syntactic", short, config=vars(args))

    log.info("Analysing syntactic invariance for %d pairs", len(pairs))

    per_pair = {}
    # For cross-persona comparison
    variant_vectors: dict[str, dict[int, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))

    for persona, trait, pos_path, neg_path in pairs:
        pos_data = torch.load(pos_path, map_location="cpu", weights_only=True)
        neg_data = torch.load(neg_path, map_location="cpu", weights_only=True)

        pos_by_v = split_by_variant(pos_data)
        neg_by_v = split_by_variant(neg_data)
        all_variants = sorted(set(pos_by_v) & set(neg_by_v))

        if len(all_variants) < 2:
            continue

        vecs: dict[int, torch.Tensor] = {}
        for vi in all_variants:
            vec = compute_variant_vector(pos_by_v[vi], neg_by_v[vi])
            if vec is not None and layer < vec.shape[0]:
                vecs[vi] = vec[layer]
                variant_vectors[trait][vi][persona] = vec[layer]

        if len(vecs) < 2:
            continue

        # Pairwise cosine across variants
        ids = sorted(vecs.keys())
        off_diag = [cosine_similarity(vecs[i], vecs[j]) for i in ids for j in ids if i < j]

        per_pair[f"{persona}_{trait}"] = {
            "n_variants": len(ids),
            "cross_variant_cosine_mean": float(np.mean(off_diag)),
            "cross_variant_cosine_std": float(np.std(off_diag)),
            "cross_variant_cosine_min": float(np.min(off_diag)),
        }

    save_json(per_pair, output_dir / "syntactic_invariance.json")

    # Per-trait summary
    trait_summary: dict[str, dict] = {}
    for key, data in per_pair.items():
        _, trait = parse_persona_trait_from_stem(key)
        if trait:
            trait_summary.setdefault(trait, []).append(data["cross_variant_cosine_mean"])
    for trait in trait_summary:
        vals = trait_summary[trait]
        trait_summary[trait] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    save_json(trait_summary, output_dir / "syntactic_by_trait.json")

    # Cross-persona per variant
    cross_persona: dict[str, dict] = {}
    across_persona_vals = []
    for trait, variants in variant_vectors.items():
        cross_persona[trait] = {}
        for vi, pvecs in variants.items():
            slugs = sorted(pvecs.keys())
            if len(slugs) < 2:
                continue
            sims = [cosine_similarity(pvecs[a], pvecs[b]) for i, a in enumerate(slugs) for b in slugs[i+1:]]
            cross_persona[trait][vi] = {"mean": float(np.mean(sims)), "std": float(np.std(sims))}
            across_persona_vals.append(np.mean(sims))
    save_json(cross_persona, output_dir / "cross_persona_per_variant.json")

    # Final comparison + significance test
    within = [d["cross_variant_cosine_mean"] for d in per_pair.values()]
    from scipy.stats import mannwhitneyu
    sig_test = {}
    if within and across_persona_vals:
        stat, p_val = mannwhitneyu(across_persona_vals, within, alternative="greater")
        sig_test = {
            "test": "Mann-Whitney U (one-sided: across-persona > within-persona)",
            "statistic": float(stat),
            "p_value": float(p_val),
            "significant_at_005": p_val < 0.05,
            "n_within": len(within),
            "n_across": len(across_persona_vals),
        }
    comparison = {
        "within_persona_across_variant": {
            "mean": float(np.mean(within)) if within else None,
            "std": float(np.std(within)) if within else None,
        },
        "across_persona_within_variant": {
            "mean": float(np.mean(across_persona_vals)) if across_persona_vals else None,
            "std": float(np.std(across_persona_vals)) if across_persona_vals else None,
        },
        "significance_test": sig_test,
    }
    save_json(comparison, output_dir / "invariance_comparison.json")

    log_summary({
        "syntactic/within_persona_mean": comparison["within_persona_across_variant"]["mean"],
        "syntactic/across_persona_mean": comparison["across_persona_within_variant"]["mean"],
    })

    # --- Figure 1: per-trait cross-variant similarity ---
    if trait_summary:
        traits_sorted = sorted(trait_summary, key=lambda t: trait_summary[t]["mean"])
        means = [trait_summary[t]["mean"] for t in traits_sorted]
        stds = [trait_summary[t]["std"] for t in traits_sorted]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(traits_sorted)), means, xerr=stds, capsize=3, alpha=0.8, color="#4C72B0")
        ax.set_yticks(range(len(traits_sorted)))
        ax.set_yticklabels([t.replace("_", " ").title() for t in traits_sorted])
        ax.set_xlabel("Cross-Variant Cosine Similarity (within persona)")
        ax.set_title("Syntactic Invariance by Trait")
        ax.axvline(1.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        save_fig(fig, output_dir / "syntactic_by_trait.png")

    # --- Figure 2: within-persona vs across-persona comparison ---
    if within and across_persona_vals:
        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot([within, across_persona_vals],
                        labels=["Within-persona\n(across variants)", "Across-persona\n(same variant)"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#C44E52")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Syntactic vs Semantic Variation")
        fig.tight_layout()
        save_fig(fig, output_dir / "invariance_comparison.png")

    log_images(output_dir, prefix="r3_syntactic")
    finish_run()

    log.info("=== Syntactic Invariance Summary ===")
    log.info("Within-persona across-variant:  %.4f ± %.4f",
             comparison["within_persona_across_variant"]["mean"] or 0,
             comparison["within_persona_across_variant"]["std"] or 0)
    log.info("Across-persona within-variant:  %.4f ± %.4f",
             comparison["across_persona_within_variant"]["mean"] or 0,
             comparison["across_persona_within_variant"]["std"] or 0)
    if sig_test:
        log.info("Significance (across > within): p=%.4f (%s)",
                 sig_test["p_value"],
                 "significant" if sig_test["significant_at_005"] else "not significant")
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
