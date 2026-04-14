#!/usr/bin/env python3
"""Convergence analysis: how many activation pairs are needed for stable vectors?

Computes vectors at subset sizes (1, 2, 5, 10, 20, 50, 100), measures cosine
to full-data vector, and tracks when transfer-matrix clusters stabilize.

Usage:
    python pipeline/r2_convergence.py \
        --activations-dir outputs/gemma-2-27b-it/activations \
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
from sklearn.metrics import adjusted_rand_score

from persona_steering.config import Trait, OUTPUTS_DIR, TARGET_LAYER
from persona_steering.analysis import build_transfer_matrix, cluster_persona_vectors
from persona_steering.utils import (
    log, save_json, save_fig, cosine_similarity, VectorShim,
    discover_activation_pairs, parse_persona_trait_from_stem,
)
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_summary, log_images


SUBSET_SIZES = [1, 2, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convergence: vector stability vs dataset size")
    p.add_argument("--activations-dir", type=str, required=True)
    p.add_argument("--vectors-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_subset_vector(
    pos_acts: list[torch.Tensor], neg_acts: list[torch.Tensor],
    n: int, rng: np.random.Generator,
) -> torch.Tensor:
    """Contrastive vector from a random subset of n activations."""
    _clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    n_pos = min(n, len(pos_acts))
    n_neg = min(n, len(neg_acts))
    pos_idx = rng.choice(len(pos_acts), size=n_pos, replace=False)
    neg_idx = rng.choice(len(neg_acts), size=n_neg, replace=False)
    pos_sum = sum(_clean(pos_acts[i][:-1].float()) for i in pos_idx)
    neg_sum = sum(_clean(neg_acts[i][:-1].float()) for i in neg_idx)
    return (pos_sum / n_pos) - (neg_sum / n_neg)


def main() -> None:
    args = parse_args()

    activations_dir = Path(args.activations_dir)
    vectors_dir = Path(args.vectors_dir)
    short = activations_dir.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "robustness" / "convergence"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    layer = args.layer

    pairs = discover_activation_pairs(activations_dir)
    if not pairs:
        log.error("No activation pairs found")
        return

    init_run("r2_convergence", short, config=vars(args))

    # Load full-data vectors
    full_vectors: dict[tuple[str, str], torch.Tensor] = {}
    for persona, trait, _, _ in pairs:
        path = vectors_dir / f"{persona}_{trait}.pt"
        if path.exists():
            vec = torch.load(path, map_location="cpu", weights_only=False)["vector"].float()
            if layer < vec.shape[0]:
                full_vectors[(persona, trait)] = vec[layer]

    # Load all activations
    all_acts: dict[tuple[str, str], tuple[list, list]] = {}
    for persona, trait, pos_path, neg_path in pairs:
        pos = list(torch.load(pos_path, map_location="cpu", weights_only=True).values())
        neg = list(torch.load(neg_path, map_location="cpu", weights_only=True).values())
        all_acts[(persona, trait)] = (pos, neg)

    personas = sorted({p for p, _ in all_acts})
    traits = sorted({Trait(t) for _, t in all_acts}, key=lambda t: t.value)

    max_n = min(min(len(p) for p, n in all_acts.values()), min(len(n) for p, n in all_acts.values()))
    sizes = [s for s in SUBSET_SIZES if s <= max_n]
    if max_n not in sizes:
        sizes.append(max_n)
    log.info("Subset sizes: %s (max: %d)", sizes, max_n)

    # Compute convergence curves
    convergence: dict[str, dict[int, float]] = {}
    subset_vecs: dict[int, dict[str, dict[Trait, dict[int, VectorShim]]]] = {}

    for persona, trait, _, _ in pairs:
        pos_acts, neg_acts = all_acts[(persona, trait)]
        full_vec = full_vectors.get((persona, trait))
        key = f"{persona}_{trait}"
        curve = {}

        for n in sizes:
            vec = compute_subset_vector(pos_acts, neg_acts, n, rng)
            layer_vec = vec[layer] if layer < vec.shape[0] else None
            if layer_vec is not None and full_vec is not None:
                curve[n] = cosine_similarity(layer_vec, full_vec)
                shim = VectorShim(layer_vec, persona, Trait(trait), layer)
                subset_vecs.setdefault(n, {}).setdefault(persona, {}).setdefault(Trait(trait), {})[layer] = shim

        convergence[key] = curve

    save_json(convergence, output_dir / "convergence_curves.json")

    # Per-trait average
    trait_conv: dict[str, dict[int, dict]] = {}
    for trait_enum in traits:
        tv = trait_enum.value
        trait_conv[tv] = {}
        for n in sizes:
            cosines = [convergence[f"{p}_{tv}"].get(n) for p in personas
                       if f"{p}_{tv}" in convergence and n in convergence[f"{p}_{tv}"]]
            cosines = [c for c in cosines if c is not None]
            if cosines:
                trait_conv[tv][n] = {"mean": float(np.mean(cosines)), "std": float(np.std(cosines))}
    save_json(trait_conv, output_dir / "convergence_by_trait.json")

    # Transfer matrix stability at each N
    full_nested: dict[str, dict[Trait, dict[int, VectorShim]]] = {}
    for (persona, trait), vec in full_vectors.items():
        shim = VectorShim(vec, persona, Trait(trait), layer)
        full_nested.setdefault(persona, {}).setdefault(Trait(trait), {})[layer] = shim

    full_tm = build_transfer_matrix(full_nested, personas, traits, layer)
    full_labels = cluster_persona_vectors(full_tm, personas)["labels"]

    transfer_stability: dict[int, dict] = {}
    for n in sizes:
        if n not in subset_vecs:
            continue
        nested = subset_vecs[n]
        avail = [p for p in personas if p in nested]
        if len(avail) < 2:
            continue
        tm = build_transfer_matrix(nested, avail, traits, layer)
        clustering = cluster_persona_vectors(tm, avail)
        ari = float(adjusted_rand_score(
            [full_labels.get(p, -1) for p in avail],
            [clustering["labels"].get(p, -1) for p in avail],
        ))
        frob = float(np.linalg.norm(tm - build_transfer_matrix(full_nested, avail, traits, layer), "fro"))
        transfer_stability[n] = {"ari": ari, "frobenius": frob, "n_clusters": len(clustering["clusters"])}

    save_json(transfer_stability, output_dir / "transfer_stability.json")

    # Log W&B metrics
    for n in sizes:
        all_cos = [convergence[k].get(n) for k in convergence if n in convergence[k]]
        all_cos = [c for c in all_cos if c is not None]
        ts = transfer_stability.get(n, {})
        if all_cos:
            log_metrics({f"convergence/N{n}/mean_cosine": float(np.mean(all_cos)),
                         f"convergence/N{n}/ari": ts.get("ari", -1)})

    # --- Figure 1: convergence curves (per trait, mean ± std) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(traits)))
    for i, trait_enum in enumerate(traits):
        tv = trait_enum.value
        if tv not in trait_conv:
            continue
        ns = sorted(n for n in trait_conv[tv] if isinstance(n, int))
        means = [trait_conv[tv][n]["mean"] for n in ns]
        stds = [trait_conv[tv][n]["std"] for n in ns]
        ax.errorbar(ns, means, yerr=stds, fmt="o-", color=colors[i],
                    label=tv.replace("_", " ").title(), capsize=3, lw=1.5, ms=5)
    # Bold mean line across all traits
    all_ns = sorted({n for tv in trait_conv for n in trait_conv[tv] if isinstance(n, int)})
    mean_line = []
    for n in all_ns:
        vals = [trait_conv[tv][n]["mean"] for tv in trait_conv if n in trait_conv[tv]]
        mean_line.append(np.mean(vals) if vals else np.nan)
    ax.plot(all_ns, mean_line, "k-", lw=2.5, alpha=0.6, label="Mean (all traits)")
    ax.set_xlabel("Number of Activation Pairs (N)")
    ax.set_ylabel("Cosine Similarity to Full-Data Vector")
    ax.set_title("Convergence: How Many Pairs Are Needed?")
    ax.set_xscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, output_dir / "convergence_curves.png")

    # --- Figure 2: transfer matrix stability ---
    if transfer_stability:
        ns = sorted(transfer_stability.keys())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        ax.plot(ns, [transfer_stability[n]["ari"] for n in ns], "o-", color="#2171b5", lw=2)
        ax.set_xlabel("N (pairs per combo)")
        ax.set_ylabel("Adjusted Rand Index vs Full Data")
        ax.set_title("Cluster Stability")
        ax.set_xscale("log")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(ns, [transfer_stability[n]["frobenius"] for n in ns], "o-", color="#d62728", lw=2)
        ax.set_xlabel("N (pairs per combo)")
        ax.set_ylabel("Frobenius Distance to Full-Data TM")
        ax.set_title("Transfer Matrix Convergence")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)

        fig.suptitle("How Quickly Do Clusters and Transfer Structure Emerge?", fontsize=12, y=1.02)
        fig.tight_layout()
        save_fig(fig, output_dir / "transfer_stability.png")

    log_images(output_dir, prefix="r2_convergence")
    log_summary({"convergence/sizes": sizes, "convergence/max_n": max_n})
    finish_run()

    log.info("=== Convergence Summary ===")
    for n in sizes:
        all_cos = [convergence[k].get(n) for k in convergence if n in convergence[k]]
        all_cos = [c for c in all_cos if c is not None]
        ts = transfer_stability.get(n, {})
        log.info("  N=%3d: cos=%.4f±%.4f, ARI=%.3f, frob=%.3f",
                 n, np.mean(all_cos) if all_cos else 0, np.std(all_cos) if all_cos else 0,
                 ts.get("ari", -1), ts.get("frobenius", -1))
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
