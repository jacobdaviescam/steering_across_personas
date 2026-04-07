#!/usr/bin/env python3
"""Bootstrap stability analysis for steering vectors.

For each persona x trait, resample activation pairs with replacement and
recompute the contrastive vector.  Measures how stable vectors are under
dataset perturbation.

Usage:
    python pipeline/r1_bootstrap_vectors.py \
        --activations-dir outputs/gemma-2-27b-it/activations \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --n-bootstraps 50 --layer 22
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
from persona_steering.utils import (
    log, save_json, save_fig, cosine_similarity, discover_activation_pairs,
    parse_persona_trait_from_stem,
)
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_summary, log_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap stability analysis")
    p.add_argument("--activations-dir", type=str, required=True)
    p.add_argument("--vectors-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--n-bootstraps", type=int, default=50)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bootstrap_contrastive_vector(
    pos_acts: list[torch.Tensor],
    neg_acts: list[torch.Tensor],
    rng: np.random.Generator,
) -> torch.Tensor:
    """Compute a contrastive vector from a bootstrap resample."""
    pos_idx = rng.choice(len(pos_acts), size=len(pos_acts), replace=True)
    neg_idx = rng.choice(len(neg_acts), size=len(neg_acts), replace=True)
    _clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    pos_sum = sum(_clean(pos_acts[i][:-1].float()) for i in pos_idx)
    neg_sum = sum(_clean(neg_acts[i][:-1].float()) for i in neg_idx)
    return (pos_sum / len(pos_idx)) - (neg_sum / len(neg_idx))


def main() -> None:
    args = parse_args()

    activations_dir = Path(args.activations_dir)
    vectors_dir = Path(args.vectors_dir)
    short = activations_dir.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "robustness" / "bootstrap"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    layer = args.layer
    n_boots = args.n_bootstraps

    init_run("r1_bootstrap", short, config=vars(args))

    pairs = discover_activation_pairs(activations_dir)
    if not pairs:
        log.error("No activation pairs found in %s", activations_dir)
        return

    log.info("Found %d pairs, running %d bootstraps each", len(pairs), n_boots)

    results = {}
    all_pairwise = []
    all_full = []

    for idx, (persona, trait, pos_path, neg_path) in enumerate(pairs):
        log.info("Bootstrapping %s/%s...", persona, trait)

        pos_acts = list(torch.load(pos_path, map_location="cpu", weights_only=True).values())
        neg_acts = list(torch.load(neg_path, map_location="cpu", weights_only=True).values())

        if len(pos_acts) < 2 or len(neg_acts) < 2:
            log.warning("Too few activations for %s/%s, skipping", persona, trait)
            continue

        # Load full-data vector for comparison
        full_vec_path = vectors_dir / f"{persona}_{trait}.pt"
        full_vec_layer = None
        if full_vec_path.exists():
            full_vec = torch.load(full_vec_path, map_location="cpu", weights_only=False)["vector"].float()
            if layer < full_vec.shape[0]:
                full_vec_layer = full_vec[layer]

        # Generate bootstrap vectors
        boot_layer_vecs = []
        for _ in range(n_boots):
            bv = bootstrap_contrastive_vector(pos_acts, neg_acts, rng)
            if layer < bv.shape[0]:
                boot_layer_vecs.append(bv[layer])

        if len(boot_layer_vecs) < 2:
            continue

        pairwise_sims = [
            cosine_similarity(boot_layer_vecs[i], boot_layer_vecs[j])
            for i in range(len(boot_layer_vecs))
            for j in range(i + 1, len(boot_layer_vecs))
        ]

        full_sims = []
        if full_vec_layer is not None:
            full_sims = [cosine_similarity(bv, full_vec_layer) for bv in boot_layer_vecs]

        results[f"{persona}_{trait}"] = {
            "n_pos": len(pos_acts), "n_neg": len(neg_acts),
            "pairwise_cosine_mean": float(np.mean(pairwise_sims)),
            "pairwise_cosine_std": float(np.std(pairwise_sims)),
            "pairwise_cosine_min": float(np.min(pairwise_sims)),
            "full_data_cosine_mean": float(np.mean(full_sims)) if full_sims else None,
            "full_data_cosine_std": float(np.std(full_sims)) if full_sims else None,
        }
        all_pairwise.append(np.mean(pairwise_sims))
        if full_sims:
            all_full.append(np.mean(full_sims))

        log_metrics({
            "bootstrap/done": idx + 1,
            f"bootstrap/{persona}_{trait}/pairwise_mean": float(np.mean(pairwise_sims)),
        })

    save_json(results, output_dir / "bootstrap_stability.json")

    summary = {
        "n_pairs": len(results), "n_bootstraps": n_boots, "layer": layer,
        "mean_pairwise_stability": float(np.mean(all_pairwise)) if all_pairwise else None,
        "std_pairwise_stability": float(np.std(all_pairwise)) if all_pairwise else None,
        "mean_full_data_alignment": float(np.mean(all_full)) if all_full else None,
    }
    save_json(summary, output_dir / "bootstrap_summary.json")
    log_summary(summary)

    # --- Figure: distribution of pairwise stability across persona x trait ---
    if all_pairwise:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: histogram of mean pairwise cosine
        ax = axes[0]
        ax.hist(all_pairwise, bins=20, edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(all_pairwise), color="red", ls="--", label=f"mean={np.mean(all_pairwise):.4f}")
        ax.set_xlabel("Mean Pairwise Cosine (bootstrap resamples)")
        ax.set_ylabel("Count (persona x trait)")
        ax.set_title("Bootstrap Vector Stability")
        ax.legend()

        # Right: per-trait boxplot
        ax = axes[1]
        trait_groups = {}
        for key, data in results.items():
            _, trait = parse_persona_trait_from_stem(key)
            if trait:
                trait_groups.setdefault(trait, []).append(data["pairwise_cosine_mean"])
        if trait_groups:
            labels = sorted(trait_groups.keys())
            ax.boxplot([trait_groups[t] for t in labels], labels=[t.replace("_", "\n") for t in labels])
            ax.set_ylabel("Mean Pairwise Cosine")
            ax.set_title("Stability by Trait")
            ax.tick_params(axis="x", labelsize=8)

        fig.suptitle(f"Bootstrap Stability (n={n_boots}, layer {layer})", fontsize=13, y=1.02)
        fig.tight_layout()
        save_fig(fig, output_dir / "bootstrap_stability.png")

    # --- Figure: full-data alignment per pair ---
    if all_full:
        sorted_keys = sorted(results.keys(), key=lambda k: results[k].get("full_data_cosine_mean", 0) or 0)
        vals = [results[k].get("full_data_cosine_mean", 0) or 0 for k in sorted_keys]
        stds = [results[k].get("full_data_cosine_std", 0) or 0 for k in sorted_keys]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(len(vals)), vals, xerr=stds, capsize=2, alpha=0.8)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([k.replace("_", " ") for k in sorted_keys], fontsize=6)
        ax.set_xlabel("Cosine to Full-Data Vector")
        ax.set_title(f"Bootstrap → Full-Data Alignment (n={n_boots})")
        ax.axvline(1.0, color="gray", ls=":", alpha=0.5)
        fig.tight_layout()
        save_fig(fig, output_dir / "full_data_alignment.png")

    log_images(output_dir, prefix="r1_bootstrap")
    finish_run()

    log.info("=== Bootstrap Summary ===")
    if all_pairwise:
        log.info("Pairwise stability: %.4f ± %.4f", np.mean(all_pairwise), np.std(all_pairwise))
    if all_full:
        log.info("Full-data alignment: %.4f ± %.4f", np.mean(all_full), np.std(all_full))
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
