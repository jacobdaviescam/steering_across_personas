#!/usr/bin/env python3
"""General vs context-dependent steering vectors.

Computes the "general" (context-free) steering vector for each trait by averaging
across all personas.  Then measures how each persona's context-dependent vector
relates to this general direction.

Key questions:
  - How different is each persona's vector from the average?
  - Is the general vector equidistant from all clusters, or biased toward one?
  - Which traits show the most/least context-dependence?

Usage:
    python pipeline/r4_general_vs_contextual.py \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --output-dir outputs/gemma-2-27b-it/robustness/general_vs_contextual \
        --layer 22
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from persona_steering.config import Trait, OUTPUTS_DIR, TARGET_LAYER
from persona_steering.analysis import cluster_persona_vectors, build_transfer_matrix
from persona_steering.utils import log, save_json, cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General vs context-dependent steering vector analysis"
    )
    parser.add_argument(
        "--vectors-dir", type=str, required=True,
        help="Directory with vector .pt files from step 3",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
    )
    return parser.parse_args()


class _Shim:
    def __init__(self, vector, persona, trait, layer):
        self.vector = vector
        self.persona = persona
        self.trait = trait
        self.layer = layer


def main() -> None:
    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    short = vectors_dir.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "robustness" / "general_vs_contextual"
    output_dir.mkdir(parents=True, exist_ok=True)

    layer = args.layer
    trait_values = {t.value for t in Trait}

    # Load all vectors
    vectors: dict[tuple[str, str], torch.Tensor] = {}  # (persona, trait) -> layer vector
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        full_vec = data["vector"].float()
        persona = data.get("persona", "")
        trait = data.get("trait", "")
        if not persona or not trait:
            stem = pt_file.stem
            for tv in trait_values:
                if stem.endswith(f"_{tv}"):
                    persona = stem[:-(len(tv) + 1)]
                    trait = tv
                    break
        if persona and trait and layer < full_vec.shape[0]:
            vectors[(persona, trait)] = full_vec[layer]

    if not vectors:
        log.error("No vectors loaded from %s", vectors_dir)
        return

    personas = sorted({p for p, _ in vectors})
    traits = sorted({t for _, t in vectors})
    log.info("Loaded %d vectors: %d personas, %d traits", len(vectors), len(personas), len(traits))

    # Compute general (context-free) vector per trait: mean of persona vectors
    general_vectors: dict[str, torch.Tensor] = {}
    for trait in traits:
        trait_vecs = [vectors[(p, trait)] for p in personas if (p, trait) in vectors]
        if trait_vecs:
            general_vectors[trait] = torch.stack(trait_vecs).mean(dim=0)

    # Per persona x trait: cosine to general vector
    per_pair = {}
    for trait in traits:
        if trait not in general_vectors:
            continue
        gen_vec = general_vectors[trait]
        for persona in personas:
            if (persona, trait) not in vectors:
                continue
            ctx_vec = vectors[(persona, trait)]
            cos = cosine_similarity(ctx_vec, gen_vec)

            # Magnitude ratio: how much does context scale the vector?
            mag_ratio = ctx_vec.norm().item() / (gen_vec.norm().item() + 1e-10)

            # Residual: component orthogonal to general direction
            gen_unit = gen_vec / (gen_vec.norm() + 1e-8)
            proj = torch.dot(ctx_vec, gen_unit) * gen_unit
            residual = ctx_vec - proj
            residual_mag = residual.norm().item()
            total_mag = ctx_vec.norm().item()

            per_pair[f"{persona}_{trait}"] = {
                "cosine_to_general": float(cos),
                "magnitude_ratio": float(mag_ratio),
                "residual_magnitude": float(residual_mag),
                "total_magnitude": float(total_mag),
                "specificity_ratio": float(residual_mag / (total_mag + 1e-10)),
            }

    save_json(per_pair, output_dir / "per_pair.json")

    # Per-trait summary: which traits are most context-dependent?
    trait_summary = {}
    for trait in traits:
        cosines = []
        specificities = []
        for persona in personas:
            key = f"{persona}_{trait}"
            if key in per_pair:
                cosines.append(per_pair[key]["cosine_to_general"])
                specificities.append(per_pair[key]["specificity_ratio"])

        if cosines:
            trait_summary[trait] = {
                "mean_cosine_to_general": float(np.mean(cosines)),
                "std_cosine_to_general": float(np.std(cosines)),
                "min_cosine_to_general": float(np.min(cosines)),
                "mean_specificity": float(np.mean(specificities)),
                "std_specificity": float(np.std(specificities)),
                "most_different_persona": None,
                "most_similar_persona": None,
            }
            # Find extremes
            persona_cosines = {p: per_pair[f"{p}_{trait}"]["cosine_to_general"]
                               for p in personas if f"{p}_{trait}" in per_pair}
            if persona_cosines:
                trait_summary[trait]["most_different_persona"] = min(persona_cosines, key=persona_cosines.get)
                trait_summary[trait]["most_similar_persona"] = max(persona_cosines, key=persona_cosines.get)

    save_json(trait_summary, output_dir / "trait_summary.json")

    # Cluster bias: is the general vector equidistant from persona clusters?
    # Build transfer matrix and clusters
    nested: dict[str, dict[Trait, dict[int, _Shim]]] = {}
    for (persona, trait), vec in vectors.items():
        shim = _Shim(vec, persona, Trait(trait), layer)
        nested.setdefault(persona, {}).setdefault(Trait(trait), {})[layer] = shim

    trait_enums = [Trait(t) for t in traits]
    tm = build_transfer_matrix(nested, personas, trait_enums, layer)
    clustering = cluster_persona_vectors(tm, personas)
    clusters = clustering["clusters"]

    cluster_bias = {}
    for trait in traits:
        if trait not in general_vectors:
            continue
        gen_vec = general_vectors[trait]

        per_cluster = {}
        for cluster_id, members in clusters.items():
            member_cosines = []
            for persona in members:
                if (persona, trait) in vectors:
                    member_cosines.append(cosine_similarity(vectors[(persona, trait)], gen_vec))
            if member_cosines:
                per_cluster[str(cluster_id)] = {
                    "members": members,
                    "mean_cosine_to_general": float(np.mean(member_cosines)),
                    "std": float(np.std(member_cosines)),
                }

        # Also: cosine between general vector and cluster centroid
        for cluster_id, members in clusters.items():
            member_vecs = [vectors[(p, trait)] for p in members if (p, trait) in vectors]
            if member_vecs:
                centroid = torch.stack(member_vecs).mean(dim=0)
                cos_gen_centroid = cosine_similarity(gen_vec, centroid)
                per_cluster[str(cluster_id)]["centroid_cosine_to_general"] = float(cos_gen_centroid)

        cluster_bias[trait] = per_cluster

    save_json(cluster_bias, output_dir / "cluster_bias.json")
    save_json({"clusters": {str(k): v for k, v in clusters.items()},
               "labels": clustering["labels"]},
              output_dir / "clusters_used.json")

    # Per-persona summary: which personas deviate most from general across all traits?
    persona_summary = {}
    for persona in personas:
        cosines = []
        for trait in traits:
            key = f"{persona}_{trait}"
            if key in per_pair:
                cosines.append(per_pair[key]["cosine_to_general"])
        if cosines:
            persona_summary[persona] = {
                "mean_cosine_to_general": float(np.mean(cosines)),
                "std": float(np.std(cosines)),
                "most_divergent_trait": None,
                "most_aligned_trait": None,
            }
            trait_cosines = {t: per_pair[f"{persona}_{t}"]["cosine_to_general"]
                            for t in traits if f"{persona}_{t}" in per_pair}
            if trait_cosines:
                persona_summary[persona]["most_divergent_trait"] = min(trait_cosines, key=trait_cosines.get)
                persona_summary[persona]["most_aligned_trait"] = max(trait_cosines, key=trait_cosines.get)

    save_json(persona_summary, output_dir / "persona_summary.json")

    # Log summary
    log.info("=== General vs Context-Dependent Summary ===")
    log.info("")
    log.info("Per-trait context-dependence (lower cosine = more context-dependent):")
    for trait in sorted(trait_summary, key=lambda t: trait_summary[t]["mean_cosine_to_general"]):
        ts = trait_summary[trait]
        log.info("  %-15s: cos=%.4f ± %.4f  (most different: %s)",
                 trait, ts["mean_cosine_to_general"], ts["std_cosine_to_general"],
                 ts["most_different_persona"])

    log.info("")
    log.info("Per-persona divergence from general:")
    for persona in sorted(persona_summary, key=lambda p: persona_summary[p]["mean_cosine_to_general"]):
        ps = persona_summary[persona]
        log.info("  %-20s: cos=%.4f ± %.4f  (most divergent trait: %s)",
                 persona, ps["mean_cosine_to_general"], ps["std"],
                 ps["most_divergent_trait"])

    log.info("")
    log.info("Cluster bias (per trait):")
    for trait in sorted(cluster_bias):
        parts = []
        for cid, cd in cluster_bias[trait].items():
            parts.append(f"cluster {cid}: {cd['mean_cosine_to_general']:.3f}")
        log.info("  %s: %s", trait, ", ".join(parts))

    log.info("")
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
