#!/usr/bin/env python3
"""Variant convergence: how many instruction variants are needed for stable vectors?

Measures convergence along the variant axis (vs r2's question axis).
For each persona x trait, computes vectors from subsets of 1..5 variants
and measures cosine to the full (all-variant) vector.

Usage:
    python pipeline/r7_variant_convergence.py \
        --activations-dir outputs/gemma-2-27b-it/activations --layer 22
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from itertools import combinations
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
from persona_steering.wandb_utils import init_run, finish_run, log_summary, log_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Variant convergence analysis")
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
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else OUTPUTS_DIR / short / "robustness" / "variant_convergence"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    pairs = discover_activation_pairs(activations_dir)
    if not pairs:
        log.error("No activation pairs found")
        return

    init_run("r7_variant_convergence", short, config=vars(args))

    log.info("Analysing variant convergence for %d pairs", len(pairs))

    # For each persona x trait: compute per-variant vectors, then cumulative convergence
    per_pair: dict[str, dict[int, dict]] = {}

    for persona, trait, pos_path, neg_path in pairs:
        pos_data = torch.load(pos_path, map_location="cpu", weights_only=True)
        neg_data = torch.load(neg_path, map_location="cpu", weights_only=True)

        pos_by_v = split_by_variant(pos_data)
        neg_by_v = split_by_variant(neg_data)
        all_variants = sorted(set(pos_by_v) & set(neg_by_v))

        if len(all_variants) < 2:
            log.warning("Fewer than 2 variants for %s/%s, skipping", persona, trait)
            continue

        # Compute per-variant vectors (at target layer)
        variant_vecs: dict[int, torch.Tensor] = {}
        for vi in all_variants:
            vec = compute_variant_vector(pos_by_v[vi], neg_by_v[vi])
            if vec is not None and layer < vec.shape[0]:
                variant_vecs[vi] = vec[layer]

        if len(variant_vecs) < 2:
            continue

        n_variants = len(variant_vecs)
        variant_ids = sorted(variant_vecs.keys())

        # Full vector = mean of all per-variant vectors
        full_vec = sum(variant_vecs[vi] for vi in variant_ids) / n_variants

        # For each k = 1 .. n_variants: enumerate all C(n,k) combos
        key = f"{persona}_{trait}"
        per_pair[key] = {}

        for k in range(1, n_variants + 1):
            cosines = []
            for combo in combinations(variant_ids, k):
                combo_vec = sum(variant_vecs[vi] for vi in combo) / len(combo)
                cos = cosine_similarity(combo_vec, full_vec)
                cosines.append(cos)
            per_pair[key][k] = {
                "mean_cosine": float(np.mean(cosines)),
                "std_cosine": float(np.std(cosines)),
                "n_combos": len(cosines),
            }

    save_json(per_pair, output_dir / "variant_convergence.json")

    # Per-trait summary: aggregate across personas
    traits = sorted({Trait(t) for _, t, _, _ in pairs}, key=lambda t: t.value)
    personas = sorted({p for p, _, _, _ in pairs})

    # Determine the set of k values (should be 1..5 typically)
    all_ks = sorted({k for data in per_pair.values() for k in data})

    trait_conv: dict[str, dict[int, dict]] = {}
    for trait_enum in traits:
        tv = trait_enum.value
        trait_conv[tv] = {}
        for k in all_ks:
            cosines = []
            for p in personas:
                pair_key = f"{p}_{tv}"
                if pair_key in per_pair and k in per_pair[pair_key]:
                    cosines.append(per_pair[pair_key][k]["mean_cosine"])
            if cosines:
                trait_conv[tv][k] = {
                    "mean": float(np.mean(cosines)),
                    "std": float(np.std(cosines)),
                }
    save_json(trait_conv, output_dir / "variant_convergence_by_trait.json")

    # --- Figure 1: convergence curves (cosine vs k, one line per trait) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(traits)))
    for i, trait_enum in enumerate(traits):
        tv = trait_enum.value
        if tv not in trait_conv or not trait_conv[tv]:
            continue
        ks = sorted(k for k in trait_conv[tv] if isinstance(k, int))
        means = [trait_conv[tv][k]["mean"] for k in ks]
        stds = [trait_conv[tv][k]["std"] for k in ks]
        ax.errorbar(ks, means, yerr=stds, fmt="o-", color=colors[i],
                    label=tv.replace("_", " ").title(), capsize=3, lw=1.5, ms=5)

    # Bold black line for mean across all traits
    mean_line = []
    for k in all_ks:
        vals = [trait_conv[tv][k]["mean"] for tv in trait_conv if k in trait_conv[tv]]
        mean_line.append(np.mean(vals) if vals else np.nan)
    ax.plot(all_ks, mean_line, "k-", lw=2.5, alpha=0.6, label="Mean (all traits)")

    ax.set_xlabel("Number of Instruction Variants")
    ax.set_ylabel("Cosine Similarity to Full Vector")
    ax.set_title("Variant Convergence: How Many Instruction Variants Are Needed?")
    ax.set_xticks(all_ks)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, output_dir / "variant_convergence_curves.png")

    # --- Figure 2: variant vs question comparison ---
    r2_path = OUTPUTS_DIR / short / "robustness" / "convergence" / "convergence_by_trait.json"
    if r2_path.exists():
        import json
        with open(r2_path) as f:
            r2_data = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: variant convergence (fraction of total)
        ax = axes[0]
        for i, trait_enum in enumerate(traits):
            tv = trait_enum.value
            if tv not in trait_conv or not trait_conv[tv]:
                continue
            ks = sorted(k for k in trait_conv[tv] if isinstance(k, int))
            max_k = max(ks)
            fracs = [k / max_k for k in ks]
            means = [trait_conv[tv][k]["mean"] for k in ks]
            ax.plot(fracs, means, "o-", color=colors[i], lw=1.5, ms=4,
                    label=tv.replace("_", " ").title())
        ax.set_xlabel("Fraction of Variants Used")
        ax.set_ylabel("Cosine to Full Vector")
        ax.set_title("Variant Axis Convergence")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Right: question convergence (fraction of total, from r2)
        ax = axes[1]
        for i, trait_enum in enumerate(traits):
            tv = trait_enum.value
            if tv not in r2_data:
                continue
            ns = sorted(int(n) for n in r2_data[tv])
            max_n = max(ns)
            fracs = [n / max_n for n in ns]
            means = [r2_data[tv][str(n)]["mean"] for n in ns]
            ax.plot(fracs, means, "o-", color=colors[i], lw=1.5, ms=4,
                    label=tv.replace("_", " ").title())
        ax.set_xlabel("Fraction of Questions Used")
        ax.set_ylabel("Cosine to Full Vector")
        ax.set_title("Question Axis Convergence (R2)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

        fig.suptitle("Convergence: Variants vs Questions", fontsize=13, y=1.02)
        fig.tight_layout()
        save_fig(fig, output_dir / "variant_vs_question.png")
        log.info("Variant vs question comparison figure saved")
    else:
        log.info("R2 convergence data not found at %s, skipping comparison figure", r2_path)

    log_images(output_dir, prefix="r7_variant_convergence")
    log_summary({
        "variant_convergence/n_pairs": len(per_pair),
        "variant_convergence/n_ks": len(all_ks),
    })
    finish_run()

    log.info("=== Variant Convergence Summary ===")
    for k in all_ks:
        all_cos = []
        for data in per_pair.values():
            if k in data:
                all_cos.append(data[k]["mean_cosine"])
        if all_cos:
            log.info("  k=%d: cos=%.4f +/- %.4f  (%d pairs)",
                     k, np.mean(all_cos), np.std(all_cos), len(all_cos))
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
