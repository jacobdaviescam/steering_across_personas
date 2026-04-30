#!/usr/bin/env python3
"""Trait-vector robustness to context paraphrase.

Tests whether the trait steering vector stays the same when the persona
context phrasing changes. Run on activations produced with
`1_generate.py --vary context`, where the trait instruction is held fixed
(I = I_0) and the persona system prompt is swept across its 5 variants.
Each `v{si}_q{qi}` activation key indexes the system-prompt variant `si`,
not the instruction phrasing.

For each (persona, trait) we compute one steering vector per `si`, then
measure cross-`si` cosine. High similarity = the trait direction is robust
to surface paraphrasing of the persona context.

Also compares: within-persona cross-context sim vs across-persona same-context sim.

Usage:
    python pipeline/r3_b_trait_robustness_to_context.py \
        --activations-dir outputs/gemma-2-27b-it/activations_context --layer 22
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

from persona_steering.config import OUTPUTS_DIR, TARGET_LAYER
from persona_steering.utils import (
    log, save_json, save_fig, cosine_similarity,
    discover_activation_pairs, parse_persona_trait_from_stem,
)
from persona_steering.wandb_utils import init_run, finish_run, log_summary, log_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Is the trait robust when context changes?")
    p.add_argument("--activations-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    return p.parse_args()


def split_by_context(data: dict[str, torch.Tensor]) -> dict[int, list[torch.Tensor]]:
    """Split activation dict by system-prompt variant index (keys like 'v{si}_q{qi}')."""
    by_ctx: dict[int, list[torch.Tensor]] = defaultdict(list)
    for key, tensor in data.items():
        m = re.match(r"v(\d+)_q\d+", key)
        if m:
            by_ctx[int(m.group(1))].append(tensor)
    return dict(by_ctx)


def compute_context_vector(
    pos_ctx: list[torch.Tensor], neg_ctx: list[torch.Tensor],
) -> torch.Tensor | None:
    """Contrastive vector from a single context's activations."""
    if not pos_ctx or not neg_ctx:
        return None
    _clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    pos_sum = sum(_clean(v[:-1].float()) for v in pos_ctx)
    neg_sum = sum(_clean(v[:-1].float()) for v in neg_ctx)
    return (pos_sum / len(pos_ctx)) - (neg_sum / len(neg_ctx))


def main() -> None:
    args = parse_args()

    activations_dir = Path(args.activations_dir)
    short = activations_dir.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "robustness" / "trait_robustness_to_context"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    pairs = discover_activation_pairs(activations_dir)
    if not pairs:
        log.error("No activation pairs found")
        return

    init_run("r3b_trait_robustness_to_context", short, config=vars(args))

    log.info("Is the trait robust when context changes? Analysing %d pairs", len(pairs))

    per_pair = {}
    # For cross-persona comparison
    context_vectors: dict[str, dict[int, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))

    for persona, trait, pos_path, neg_path in pairs:
        pos_data = torch.load(pos_path, map_location="cpu", weights_only=True)
        neg_data = torch.load(neg_path, map_location="cpu", weights_only=True)

        pos_by_c = split_by_context(pos_data)
        neg_by_c = split_by_context(neg_data)
        all_contexts = sorted(set(pos_by_c) & set(neg_by_c))

        if len(all_contexts) < 2:
            continue

        vecs: dict[int, torch.Tensor] = {}
        for si in all_contexts:
            vec = compute_context_vector(pos_by_c[si], neg_by_c[si])
            if vec is not None and layer < vec.shape[0]:
                vecs[si] = vec[layer]
                context_vectors[trait][si][persona] = vec[layer]

        if len(vecs) < 2:
            continue

        ids = sorted(vecs.keys())
        off_diag = [cosine_similarity(vecs[i], vecs[j]) for i in ids for j in ids if i < j]

        per_pair[f"{persona}_{trait}"] = {
            "n_contexts": len(ids),
            "cross_context_cosine_mean": float(np.mean(off_diag)),
            "cross_context_cosine_std": float(np.std(off_diag)),
            "cross_context_cosine_min": float(np.min(off_diag)),
        }

    save_json(per_pair, output_dir / "trait_robustness_to_context.json")

    # Per-trait summary
    trait_summary: dict[str, dict] = {}
    for key, data in per_pair.items():
        _, trait = parse_persona_trait_from_stem(key)
        if trait:
            trait_summary.setdefault(trait, []).append(data["cross_context_cosine_mean"])
    for trait in trait_summary:
        vals = trait_summary[trait]
        trait_summary[trait] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    save_json(trait_summary, output_dir / "trait_robustness_by_trait.json")

    # Cross-persona per context
    cross_persona: dict[str, dict] = {}
    across_persona_vals = []
    for trait, contexts in context_vectors.items():
        cross_persona[trait] = {}
        for si, pvecs in contexts.items():
            slugs = sorted(pvecs.keys())
            if len(slugs) < 2:
                continue
            sims = [cosine_similarity(pvecs[a], pvecs[b]) for i, a in enumerate(slugs) for b in slugs[i+1:]]
            cross_persona[trait][si] = {"mean": float(np.mean(sims)), "std": float(np.std(sims))}
            across_persona_vals.append(np.mean(sims))
    save_json(cross_persona, output_dir / "cross_persona_per_context.json")

    within = [d["cross_context_cosine_mean"] for d in per_pair.values()]
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
        "within_persona_across_context": {
            "mean": float(np.mean(within)) if within else None,
            "std": float(np.std(within)) if within else None,
        },
        "across_persona_within_context": {
            "mean": float(np.mean(across_persona_vals)) if across_persona_vals else None,
            "std": float(np.std(across_persona_vals)) if across_persona_vals else None,
        },
        "significance_test": sig_test,
    }
    save_json(comparison, output_dir / "invariance_comparison.json")

    log_summary({
        "trait_robustness/within_persona_mean": comparison["within_persona_across_context"]["mean"],
        "trait_robustness/across_persona_mean": comparison["across_persona_within_context"]["mean"],
    })

    if trait_summary:
        traits_sorted = sorted(trait_summary, key=lambda t: trait_summary[t]["mean"])
        means = [trait_summary[t]["mean"] for t in traits_sorted]
        stds = [trait_summary[t]["std"] for t in traits_sorted]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(traits_sorted)), means, xerr=stds, capsize=3, alpha=0.8, color="#4C72B0")
        ax.set_yticks(range(len(traits_sorted)))
        ax.set_yticklabels([t.replace("_", " ").title() for t in traits_sorted])
        ax.set_xlabel("Cross-Context Cosine Similarity (within persona)")
        ax.set_title("Is the Trait Robust When Context Changes? (by Trait)")
        ax.axvline(1.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        save_fig(fig, output_dir / "trait_robustness_by_trait.png")

    if within and across_persona_vals:
        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot([within, across_persona_vals],
                        labels=["Within-persona\n(across contexts)", "Across-persona\n(same context)"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#C44E52")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Trait Robustness: Within-Persona Context Change vs Across-Persona")
        fig.tight_layout()
        save_fig(fig, output_dir / "invariance_comparison.png")

    log_images(output_dir, prefix="r3b_trait_robustness_to_context")
    finish_run()

    log.info("=== Is the Trait Robust When Context Changes? — Summary ===")
    log.info("Within-persona across-context:  %.4f ± %.4f",
             comparison["within_persona_across_context"]["mean"] or 0,
             comparison["within_persona_across_context"]["std"] or 0)
    log.info("Across-persona within-context:  %.4f ± %.4f",
             comparison["across_persona_within_context"]["mean"] or 0,
             comparison["across_persona_within_context"]["std"] or 0)
    if sig_test:
        log.info("Significance (across > within): p=%.4f (%s)",
                 sig_test["p_value"],
                 "significant" if sig_test["significant_at_005"] else "not significant")
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
