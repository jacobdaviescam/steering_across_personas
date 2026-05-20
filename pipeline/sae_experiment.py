#!/usr/bin/env python3
"""SAE feature comparison: do SAE features match our steering vectors?

Loads a pre-trained Sparse Autoencoder (Gemma Scope 2) and compares its
learned features to our extracted steering vectors. For each trait, finds
the SAE features most aligned with the steering vector direction, and tests
whether the same features align across different personas.

Key questions:
1. Does the SAE have features that match our steering vectors?
2. Is there one "honesty feature" or multiple context-specific ones?
3. Do different personas activate different SAE features for the same trait?

Usage:
    python pipeline/sae_experiment.py \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --layer 22 --width 262k --l0 small
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
    log, save_json, save_fig, cosine_similarity, load_vectors,
    parse_persona_trait_from_stem,
)
from persona_steering.wandb_utils import init_run, finish_run, log_summary, log_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAE feature comparison with steering vectors")
    p.add_argument("--vectors-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--width", type=str, default="262k",
                   help="SAE width (16k or 262k)")
    p.add_argument("--l0", type=str, default="small",
                   help="SAE L0 sparsity (small or big)")
    p.add_argument("--sae-repo", type=str, default="google/gemma-scope-2-27b-it",
                   help="HuggingFace repo for the SAE")
    p.add_argument("--sae-site", type=str, default="resid_post_all",
                   help="SAE site (e.g. resid_post, resid_post_all)")
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of top SAE features to report per vector")
    p.add_argument("--baseline-personas", type=str, nargs="*",
                   default=["null", "nonsense"])
    return p.parse_args()


def load_sae(layer: int, width: str, l0: str, repo: str = "google/gemma-scope-2-27b-it",
             site: str = "resid_post_all") -> tuple[torch.Tensor, dict]:
    """Load an SAE decoder from HuggingFace.

    Supports both Gemma Scope 2 (safetensors) and Gemma Scope 1 (npz) formats.

    Returns:
        (decoder_weights, config_dict) where decoder is (n_features, hidden_dim).
    """
    from huggingface_hub import hf_hub_download
    import json

    sae_id = f"layer_{layer}_width_{width}_l0_{l0}"
    sae_path = f"{site}/{sae_id}"
    log.info("Loading SAE: %s / %s", repo, sae_path)

    # Try safetensors format (Gemma Scope 2)
    try:
        cfg_path = hf_hub_download(repo, f"{sae_path}/config.json")
        params_path = hf_hub_download(repo, f"{sae_path}/params.safetensors")
        import safetensors.torch as st
        tensors = st.load_file(params_path)
        cfg = json.load(open(cfg_path))
        decoder = tensors["w_dec"]  # (n_features, hidden_dim)
        log.info("SAE loaded (safetensors): %d features, hidden_dim=%d",
                 decoder.shape[0], decoder.shape[1])
        return decoder, cfg
    except Exception:
        pass

    # Try npz format (Gemma Scope 1)
    try:
        params_path = hf_hub_download(repo, f"{sae_path}/params.npz")
        data = np.load(params_path)
        decoder = torch.from_numpy(data["w_dec"]).float()
        cfg = {"repo": repo, "sae_path": sae_path, "format": "npz"}
        log.info("SAE loaded (npz): %d features, hidden_dim=%d",
                 decoder.shape[0], decoder.shape[1])
        return decoder, cfg
    except Exception as e:
        log.error("Failed to load SAE from %s / %s: %s", repo, sae_path, e)
        raise


def top_aligned_features(
    vector: torch.Tensor,
    sae_decoder: torch.Tensor,
    k: int = 20,
) -> list[dict]:
    """Find the SAE features most aligned with a given direction.

    Args:
        vector: steering vector (hidden_dim,)
        sae_decoder: SAE decoder weights (n_features, hidden_dim)
        k: number of top features to return

    Returns:
        List of {feature_idx, cosine, dot_product} dicts, sorted by |cosine|.
    """
    vector = vector.float()
    decoder = sae_decoder.float()

    # Normalise
    vec_norm = vector / (vector.norm() + 1e-8)
    dec_norms = decoder.norm(dim=1, keepdim=True)
    dec_unit = decoder / (dec_norms + 1e-8)

    # Cosine similarity with every feature
    cosines = dec_unit @ vec_norm  # (n_features,)
    dots = decoder @ vec_norm      # (n_features,)

    # Top-k by absolute cosine
    abs_cos = cosines.abs()
    topk_vals, topk_idx = abs_cos.topk(k)

    results = []
    for i in range(k):
        idx = topk_idx[i].item()
        results.append({
            "feature_idx": idx,
            "cosine": float(cosines[idx]),
            "abs_cosine": float(abs_cos[idx]),
            "dot_product": float(dots[idx]),
        })
    return results


def main() -> None:
    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    short = vectors_dir.parent.name
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else OUTPUTS_DIR / short / "sae_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    # Load steering vectors
    vectors = load_vectors(vectors_dir, layer)
    if not vectors:
        log.error("No vectors loaded")
        return

    baseline_slugs = set(args.baseline_personas)
    personas = sorted({p for p, _ in vectors if p not in baseline_slugs})
    traits = sorted({t for _, t in vectors})
    log.info("Loaded %d vectors: %d personas, %d traits", len(vectors), len(personas), len(traits))

    # Load SAE
    decoder, sae_cfg = load_sae(layer, args.width, args.l0,
                                repo=args.sae_repo, site=args.sae_site)
    decoder = decoder.detach().cpu().float()  # (n_features, hidden_dim)
    n_features = decoder.shape[0]

    # Verify dimensions match
    sample_vec = next(iter(vectors.values()))
    if decoder.shape[1] != sample_vec.shape[0]:
        log.error("Dimension mismatch: SAE hidden_dim=%d, vector dim=%d. "
                  "Make sure the SAE matches the model used to extract vectors.",
                  decoder.shape[1], sample_vec.shape[0])
        return

    init_run("sae_comparison", short, config=vars(args))

    # ------------------------------------------------------------------
    # 1. For each persona x trait, find top-aligned SAE features
    # ------------------------------------------------------------------
    all_top_features = {}
    for persona in personas:
        for trait in traits:
            if (persona, trait) not in vectors:
                continue
            vec = vectors[(persona, trait)]
            top = top_aligned_features(vec, decoder, k=args.top_k)
            all_top_features[f"{persona}_{trait}"] = top

    save_json(all_top_features, output_dir / "top_features_per_vector.json")

    # ------------------------------------------------------------------
    # 2. For each trait: compute general vector, find its top features
    # ------------------------------------------------------------------
    general_top = {}
    general_vecs = {}
    for trait in traits:
        vecs = [vectors[(p, trait)] for p in personas if (p, trait) in vectors]
        if not vecs:
            continue
        general = torch.stack(vecs).mean(dim=0)
        general_vecs[trait] = general
        top = top_aligned_features(general, decoder, k=args.top_k)
        general_top[trait] = top
        best = top[0]
        log.info("General %s: best SAE feature #%d (cosine=%.4f)",
                 trait, best["feature_idx"], best["cosine"])

    save_json(general_top, output_dir / "top_features_general.json")

    # ------------------------------------------------------------------
    # 3. Feature overlap: do different personas activate the same features?
    # ------------------------------------------------------------------
    # For each trait, check how many of the top-k features are shared
    # across personas vs unique to specific personas
    feature_overlap = {}
    for trait in traits:
        persona_features = {}
        for persona in personas:
            key = f"{persona}_{trait}"
            if key not in all_top_features:
                continue
            top_idx = {f["feature_idx"] for f in all_top_features[key][:10]}
            persona_features[persona] = top_idx

        if len(persona_features) < 2:
            continue

        # Features appearing in ALL personas' top-10
        all_sets = list(persona_features.values())
        shared = all_sets[0].intersection(*all_sets[1:])

        # Features appearing in at least half
        from collections import Counter
        feature_counts = Counter()
        for s in all_sets:
            feature_counts.update(s)
        common = {f for f, c in feature_counts.items() if c >= len(all_sets) // 2}

        # Features unique to one persona
        unique_per_persona = {}
        for persona, feats in persona_features.items():
            others = set()
            for p2, f2 in persona_features.items():
                if p2 != persona:
                    others.update(f2)
            unique_per_persona[persona] = sorted(feats - others)

        feature_overlap[trait] = {
            "n_personas": len(persona_features),
            "top_k": 10,
            "shared_by_all": sorted(shared),
            "n_shared_by_all": len(shared),
            "shared_by_majority": sorted(common),
            "n_shared_by_majority": len(common),
            "unique_per_persona": unique_per_persona,
            "n_unique_per_persona": {p: len(v) for p, v in unique_per_persona.items()},
        }

    save_json(feature_overlap, output_dir / "feature_overlap.json")

    # ------------------------------------------------------------------
    # 4. Cosine between general steering vector and best SAE feature
    # ------------------------------------------------------------------
    best_feature_alignment = {}
    for trait in traits:
        if trait not in general_top:
            continue
        best = general_top[trait][0]
        best_feature_alignment[trait] = {
            "feature_idx": best["feature_idx"],
            "cosine": best["cosine"],
            "abs_cosine": best["abs_cosine"],
        }
    save_json(best_feature_alignment, output_dir / "best_feature_alignment.json")

    # ------------------------------------------------------------------
    # 5. Per-persona cosine to the SAME best feature
    # ------------------------------------------------------------------
    # For each trait, take the best SAE feature for the general vector,
    # then measure how well each persona's vector aligns with that feature
    persona_vs_best_feature = {}
    for trait in traits:
        if trait not in general_top:
            continue
        best_idx = general_top[trait][0]["feature_idx"]
        best_feature_dir = decoder[best_idx]

        per_persona = {}
        for persona in personas:
            if (persona, trait) not in vectors:
                continue
            cos = cosine_similarity(vectors[(persona, trait)], best_feature_dir)
            per_persona[persona] = float(cos)

        persona_vs_best_feature[trait] = {
            "feature_idx": best_idx,
            "general_cosine": general_top[trait][0]["cosine"],
            "per_persona": per_persona,
            "mean": float(np.mean(list(per_persona.values()))),
            "std": float(np.std(list(per_persona.values()))),
            "most_aligned": max(per_persona, key=per_persona.get),
            "least_aligned": min(per_persona, key=per_persona.get),
        }

    save_json(persona_vs_best_feature, output_dir / "persona_vs_best_feature.json")

    # ------------------------------------------------------------------
    # Figure 1: Best SAE feature alignment per trait
    # ------------------------------------------------------------------
    if best_feature_alignment:
        sorted_traits = sorted(best_feature_alignment,
                               key=lambda t: best_feature_alignment[t]["abs_cosine"])
        cosines = [best_feature_alignment[t]["cosine"] for t in sorted_traits]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#C44E52" if abs(c) < 0.3 else "#55A868" if abs(c) > 0.5 else "#4C72B0"
                  for c in cosines]
        ax.barh(range(len(sorted_traits)), cosines, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_traits)))
        ax.set_yticklabels([t.replace("_", " ").title() for t in sorted_traits])
        ax.set_xlabel("Cosine Similarity (steering vector vs best SAE feature)")
        ax.set_title("How Well Does the Best SAE Feature Match Each Trait's Steering Vector?")
        ax.axvline(0, color="gray", ls=":", alpha=0.5)
        for i, t in enumerate(sorted_traits):
            idx = best_feature_alignment[t]["feature_idx"]
            ax.text(cosines[i] + 0.01 * (1 if cosines[i] >= 0 else -1), i,
                    f"#{idx}", fontsize=7, va="center", color="gray")
        fig.tight_layout()
        save_fig(fig, output_dir / "best_feature_alignment.png")

    # ------------------------------------------------------------------
    # Figure 2: Feature overlap heatmap (trait x trait shared features)
    # ------------------------------------------------------------------
    if feature_overlap:
        fig, ax = plt.subplots(figsize=(8, 5))
        trait_list = sorted(feature_overlap.keys())
        n_shared = [feature_overlap[t]["n_shared_by_all"] for t in trait_list]
        n_majority = [feature_overlap[t]["n_shared_by_majority"] for t in trait_list]

        x = np.arange(len(trait_list))
        ax.bar(x - 0.2, n_shared, 0.35, label="Shared by all personas", color="#4C72B0")
        ax.bar(x + 0.2, n_majority, 0.35, label="Shared by majority", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", " ").title() for t in trait_list],
                           rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Number of SAE features (out of top 10)")
        ax.set_title("Do Personas Share the Same SAE Features for Each Trait?")
        ax.legend()
        ax.set_ylim(0, 11)
        fig.tight_layout()
        save_fig(fig, output_dir / "feature_overlap.png")

    # ------------------------------------------------------------------
    # Figure 3: Per-persona alignment to the best general feature
    # ------------------------------------------------------------------
    if persona_vs_best_feature:
        sorted_traits = sorted(persona_vs_best_feature,
                               key=lambda t: persona_vs_best_feature[t]["mean"])
        fig, ax = plt.subplots(figsize=(10, 6))
        for ti, trait in enumerate(sorted_traits):
            data = persona_vs_best_feature[trait]
            vals = [data["per_persona"][p] for p in personas if p in data["per_persona"]]
            ax.scatter([ti] * len(vals), vals, alpha=0.6, s=30, zorder=3)
            ax.plot([ti - 0.3, ti + 0.3], [data["mean"], data["mean"]],
                    color="black", lw=2, zorder=4)
        ax.set_xticks(range(len(sorted_traits)))
        ax.set_xticklabels([t.replace("_", " ").title() for t in sorted_traits],
                           rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Cosine to Best SAE Feature")
        ax.set_title("Per-Persona Alignment to the General Trait's Best SAE Feature")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save_fig(fig, output_dir / "persona_vs_best_feature.png")

    log_images(output_dir, prefix="sae")
    log_summary({
        f"sae/{t}/best_cosine": best_feature_alignment[t]["cosine"]
        for t in best_feature_alignment
    })
    finish_run()

    # Summary
    log.info("=== SAE Comparison Summary ===")
    log.info("SAE: layer %d, width %s, L0 %s (%d features)", layer, args.width, args.l0, n_features)
    for trait in sorted(best_feature_alignment, key=lambda t: abs(best_feature_alignment[t]["cosine"]), reverse=True):
        ba = best_feature_alignment[trait]
        pv = persona_vs_best_feature.get(trait, {})
        fo = feature_overlap.get(trait, {})
        log.info("  %-15s: best_feature=#%d cos=%.4f  persona_spread=%.4f  shared_by_all=%d/10",
                 trait, ba["feature_idx"], ba["cosine"],
                 pv.get("std", 0), fo.get("n_shared_by_all", 0))
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
