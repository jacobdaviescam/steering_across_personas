#!/usr/bin/env python3
"""General vs context-dependent steering vectors.

Computes the "general" (context-free) vector per trait by averaging across
personas.  Measures how each persona's vector relates to this general direction,
which traits are most context-dependent, and whether the general vector is
biased toward any persona cluster.

Usage:
    python pipeline/r4_general_vs_contextual.py \
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
from persona_steering.analysis import cluster_persona_vectors, build_transfer_matrix
from persona_steering.utils import (
    log, save_json, save_fig, cosine_similarity, VectorShim,
    parse_persona_trait_from_stem, load_vectors,
)
from persona_steering.wandb_utils import init_run, finish_run, log_summary, log_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General vs context-dependent vectors")
    p.add_argument("--vectors-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--baseline-personas", type=str, nargs="*", default=["null", "nonsense"],
                   help="Slugs of baseline personas (e.g. null=no system prompt, "
                        "nonsense=gibberish system prompt). Excluded from the main analysis "
                        "and compared against the general direction if their vectors exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    short = vectors_dir.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "robustness" / "general_vs_contextual"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    # Load all vectors
    vectors = load_vectors(vectors_dir, layer)

    if not vectors:
        log.error("No vectors loaded")
        return

    init_run("r4_general_vs_contextual", short, config=vars(args))

    baseline_slugs = set(args.baseline_personas)
    personas = sorted({p for p, _ in vectors if p not in baseline_slugs})
    traits = sorted({t for _, t in vectors})
    baselines_present = sorted(baseline_slugs & {p for p, _ in vectors})
    log.info("Loaded %d vectors: %d personas, %d baselines %s, %d traits",
             len(vectors), len(personas), len(baselines_present), baselines_present, len(traits))

    # General vector per trait = mean across personas
    general: dict[str, torch.Tensor] = {}
    for trait in traits:
        vecs = [vectors[(p, trait)] for p in personas if (p, trait) in vectors]
        if vecs:
            general[trait] = torch.stack(vecs).mean(dim=0)

    # Per pair: cosine to general, specificity ratio
    per_pair = {}
    for trait in traits:
        if trait not in general:
            continue
        gen = general[trait]
        gen_unit = gen / (gen.norm() + 1e-8)
        for persona in personas:
            if (persona, trait) not in vectors:
                continue
            v = vectors[(persona, trait)]
            cos = cosine_similarity(v, gen)
            proj = torch.dot(v, gen_unit) * gen_unit
            residual_mag = (v - proj).norm().item()
            total_mag = v.norm().item()
            per_pair[f"{persona}_{trait}"] = {
                "cosine_to_general": float(cos),
                "specificity_ratio": float(residual_mag / (total_mag + 1e-10)),
                "residual_magnitude": float(residual_mag),
            }
    save_json(per_pair, output_dir / "per_pair.json")

    # Per-trait summary
    trait_summary = {}
    for trait in traits:
        cosines = [per_pair[f"{p}_{trait}"]["cosine_to_general"]
                   for p in personas if f"{p}_{trait}" in per_pair]
        if not cosines:
            continue
        persona_cos = {p: per_pair[f"{p}_{trait}"]["cosine_to_general"]
                       for p in personas if f"{p}_{trait}" in per_pair}
        trait_summary[trait] = {
            "mean_cosine": float(np.mean(cosines)),
            "std_cosine": float(np.std(cosines)),
            "most_different": min(persona_cos, key=persona_cos.get),
            "most_similar": max(persona_cos, key=persona_cos.get),
        }
    save_json(trait_summary, output_dir / "trait_summary.json")

    # Per-persona summary
    persona_summary = {}
    for persona in personas:
        cosines = [per_pair[f"{persona}_{t}"]["cosine_to_general"]
                   for t in traits if f"{persona}_{t}" in per_pair]
        if not cosines:
            continue
        trait_cos = {t: per_pair[f"{persona}_{t}"]["cosine_to_general"]
                     for t in traits if f"{persona}_{t}" in per_pair}
        persona_summary[persona] = {
            "mean_cosine": float(np.mean(cosines)),
            "std": float(np.std(cosines)),
            "most_divergent_trait": min(trait_cos, key=trait_cos.get),
        }
    save_json(persona_summary, output_dir / "persona_summary.json")

    # Cluster bias: is general vector closer to one cluster?
    nested: dict[str, dict[Trait, dict[int, VectorShim]]] = {}
    for (persona, trait), vec in vectors.items():
        shim = VectorShim(vec, persona, Trait(trait), layer)
        nested.setdefault(persona, {}).setdefault(Trait(trait), {})[layer] = shim

    trait_enums = [Trait(t) for t in traits]
    tm = build_transfer_matrix(nested, personas, trait_enums, layer)
    clusters = cluster_persona_vectors(tm, personas)["clusters"]

    cluster_bias = {}
    for trait in traits:
        if trait not in general:
            continue
        gen = general[trait]
        per_cluster = {}
        for cid, members in clusters.items():
            member_vecs = [vectors[(p, trait)] for p in members if (p, trait) in vectors]
            if not member_vecs:
                continue
            member_cos = [cosine_similarity(vectors[(p, trait)], gen) for p in members if (p, trait) in vectors]
            centroid = torch.stack(member_vecs).mean(dim=0)
            per_cluster[str(cid)] = {
                "members": members,
                "mean_cosine_to_general": float(np.mean(member_cos)),
                "centroid_cosine_to_general": float(cosine_similarity(gen, centroid)),
            }
        cluster_bias[trait] = per_cluster
    save_json(cluster_bias, output_dir / "cluster_bias.json")

    # Baseline persona comparisons (null, nonsense, etc.)
    for baseline_slug in sorted(baseline_slugs):
        baseline_traits = {t for (p, t) in vectors if p == baseline_slug}
        if not baseline_traits:
            log.info("No '%s' baseline vectors found. Skipping. "
                     "Run the pipeline with this persona to enable.", baseline_slug)
            continue

        log.info("Found '%s' baseline vectors for %d traits", baseline_slug, len(baseline_traits))

        baseline_comparison = {}
        for trait in sorted(baseline_traits):
            baseline_vec = vectors[(baseline_slug, trait)]
            gen = general.get(trait)
            if gen is None:
                continue
            cos_to_general = cosine_similarity(baseline_vec, gen)
            persona_to_baseline = {
                p: cosine_similarity(vectors[(p, trait)], baseline_vec)
                for p in personas if (p, trait) in vectors
            }
            baseline_comparison[trait] = {
                "baseline_to_general_cosine": float(cos_to_general),
                "persona_to_baseline": {p: float(v) for p, v in persona_to_baseline.items()},
                "mean_persona_to_baseline": float(np.mean(list(persona_to_baseline.values()))),
                "most_similar_to_baseline": max(persona_to_baseline, key=persona_to_baseline.get),
                "most_different_from_baseline": min(persona_to_baseline, key=persona_to_baseline.get),
            }
        save_json(baseline_comparison, output_dir / f"{baseline_slug}_persona_comparison.json")

        log.info("=== %s Baseline Comparison ===", baseline_slug.title())
        for trait in sorted(baseline_comparison, key=lambda t: baseline_comparison[t]["baseline_to_general_cosine"]):
            bc = baseline_comparison[trait]
            log.info("  %-15s: %s->general=%.4f  mean_persona->%s=%.4f  most_diff=%s",
                     trait, baseline_slug, bc["baseline_to_general_cosine"],
                     baseline_slug, bc["mean_persona_to_baseline"],
                     bc["most_different_from_baseline"])

    log_summary({
        f"general/{t}/mean_cosine": ts["mean_cosine"]
        for t, ts in trait_summary.items()
    })

    # --- Figure 1: heatmap of cosine-to-general (persona x trait) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    matrix = np.full((len(personas), len(traits)), np.nan)
    for pi, persona in enumerate(personas):
        for ti, trait in enumerate(traits):
            key = f"{persona}_{trait}"
            if key in per_pair:
                matrix[pi, ti] = per_pair[key]["cosine_to_general"]
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([t.replace("_", " ").title() for t in traits], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels([p.replace("_", " ").title() for p in personas], fontsize=9)
    for i in range(len(personas)):
        for j in range(len(traits)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] < 0.7 else "black")
    plt.colorbar(im, ax=ax, label="Cosine to General Vector", shrink=0.8)
    ax.set_title("Context-Dependence: Cosine to General (Averaged) Steering Vector")
    fig.tight_layout()
    save_fig(fig, output_dir / "general_vs_contextual_heatmap.png")

    # --- Figure 2: trait ranking by context-dependence ---
    if trait_summary:
        sorted_traits = sorted(trait_summary, key=lambda t: trait_summary[t]["mean_cosine"])
        means = [trait_summary[t]["mean_cosine"] for t in sorted_traits]
        stds = [trait_summary[t]["std_cosine"] for t in sorted_traits]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#C44E52" if m < 0.8 else "#55A868" if m > 0.9 else "#4C72B0" for m in means]
        ax.barh(range(len(sorted_traits)), means, xerr=stds, capsize=3, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_traits)))
        ax.set_yticklabels([t.replace("_", " ").title() for t in sorted_traits])
        ax.set_xlabel("Mean Cosine to General Vector (across personas)")
        ax.set_title("Which Traits Are Most Context-Dependent?")
        ax.axvline(1.0, color="gray", ls=":", alpha=0.5)
        for i, t in enumerate(sorted_traits):
            ax.text(means[i] + stds[i] + 0.01, i,
                    f"most diff: {trait_summary[t]['most_different'].replace('_', ' ')}",
                    fontsize=7, va="center", color="gray")
        fig.tight_layout()
        save_fig(fig, output_dir / "trait_context_dependence.png")

    # --- Figure 3: cluster bias — grouped bar (trait × cluster) ---
    if cluster_bias:
        traits_with_multi = [t for t in sorted(cluster_bias) if len(cluster_bias[t]) > 1]
        if traits_with_multi:
            # Collect all cluster IDs across traits
            all_cids = sorted({c for t in traits_with_multi for c in cluster_bias[t]})
            cluster_colors = ["#4C72B0", "#C44E52", "#55A868", "#E6AB02", "#984EA3"]

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(traits_with_multi))
            n_clusters = len(all_cids)
            width = 0.7 / n_clusters

            for ci, cid in enumerate(all_cids):
                vals = [cluster_bias[t].get(cid, {}).get("centroid_cosine_to_general", np.nan)
                        for t in traits_with_multi]
                members = cluster_bias[traits_with_multi[0]].get(cid, {}).get("members", [])
                label = f"Cluster {cid} ({', '.join(members[:3])}{'…' if len(members) > 3 else ''})"
                ax.bar(x + ci * width - width * n_clusters / 2, vals, width,
                       label=label, color=cluster_colors[ci % len(cluster_colors)], alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels([t.replace("_", " ").title() for t in traits_with_multi],
                               rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Centroid Cosine to General Vector")
            ax.set_title("Is the General Vector Equidistant from Persona Clusters?")
            ax.legend(fontsize=7, loc="lower left")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            save_fig(fig, output_dir / "cluster_bias.png")

    log_images(output_dir, prefix="r4_general")
    finish_run()

    log.info("=== General vs Context-Dependent Summary ===")
    for trait in sorted(trait_summary, key=lambda t: trait_summary[t]["mean_cosine"]):
        ts = trait_summary[trait]
        log.info("  %-15s: cos=%.4f ± %.4f (most diff: %s)",
                 trait, ts["mean_cosine"], ts["std_cosine"], ts["most_different"])
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
