#!/usr/bin/env python3
"""Syntactic invariance analysis: are representations driven by meaning or phrasing?

The pipeline uses 5 instruction variants per trait — semantically equivalent but
syntactically different phrasings.  This script computes a separate steering vector
from each variant independently, then measures how similar those vectors are.

High similarity = the model's representation is driven by semantic content, not
surface-level phrasing.  This directly supports the claim about smoothness of the
mapping from semantic content to representation.

Outputs:
  - Per (persona, trait): 5x5 cosine similarity matrix across variants
  - Per trait: mean within-persona cross-variant similarity
  - Comparison: within-persona variant sim vs between-persona sim (same variant)

Usage:
    python pipeline/r3_syntactic_invariance.py \
        --activations-dir outputs/gemma-2-27b-it/activations \
        --output-dir outputs/gemma-2-27b-it/robustness/syntactic \
        --layer 22
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from persona_steering.config import Trait, OUTPUTS_DIR, TARGET_LAYER
from persona_steering.utils import log, save_json, cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Syntactic invariance: per-variant vector similarity"
    )
    parser.add_argument(
        "--activations-dir", type=str, required=True,
        help="Directory with activation .pt files from step 2",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
    )
    return parser.parse_args()


def discover_pairs(activations_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """Find matching pos/neg activation file pairs."""
    trait_values = {t.value for t in Trait}
    files: dict[tuple[str, str], dict[str, Path]] = {}

    for pt_file in sorted(activations_dir.glob("*.pt")):
        stem = pt_file.stem
        if stem.endswith("_pos"):
            direction, rest = "pos", stem[:-4]
        elif stem.endswith("_neg"):
            direction, rest = "neg", stem[:-4]
        else:
            continue

        for tv in trait_values:
            if rest.endswith(f"_{tv}"):
                persona = rest[: -(len(tv) + 1)]
                files.setdefault((persona, tv), {})[direction] = pt_file
                break

    pairs = []
    for (persona, trait), directions in sorted(files.items()):
        if "pos" in directions and "neg" in directions:
            pairs.append((persona, trait, directions["pos"], directions["neg"]))
    return pairs


def split_by_variant(data: dict[str, torch.Tensor]) -> dict[int, list[torch.Tensor]]:
    """Split activation dict by variant index.

    Keys are like 'v0_q3', 'v2_q15', etc.
    Returns {variant_index: [tensors]}.
    """
    by_variant: dict[int, list[torch.Tensor]] = defaultdict(list)
    for key, tensor in data.items():
        m = re.match(r"v(\d+)_q\d+", key)
        if m:
            by_variant[int(m.group(1))].append(tensor)
    return dict(by_variant)


def compute_variant_vector(
    pos_variant: list[torch.Tensor],
    neg_variant: list[torch.Tensor],
) -> torch.Tensor | None:
    """Compute contrastive vector from a single variant's activations."""
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

    pairs = discover_pairs(activations_dir)
    if not pairs:
        log.error("No activation pairs found")
        return

    log.info("Analysing syntactic invariance for %d persona x trait pairs", len(pairs))

    per_pair_results = {}
    # For cross-persona comparison: store per-variant vectors
    variant_vectors: dict[str, dict[int, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
    # variant_vectors[trait][variant_idx][persona] = layer_vector

    for persona, trait, pos_path, neg_path in pairs:
        pos_data = torch.load(pos_path, map_location="cpu", weights_only=True)
        neg_data = torch.load(neg_path, map_location="cpu", weights_only=True)

        pos_by_variant = split_by_variant(pos_data)
        neg_by_variant = split_by_variant(neg_data)

        all_variants = sorted(set(pos_by_variant) & set(neg_by_variant))
        if len(all_variants) < 2:
            log.warning("Only %d variants for %s/%s, skipping", len(all_variants), persona, trait)
            continue

        # Compute per-variant vectors
        vecs: dict[int, torch.Tensor] = {}
        for vi in all_variants:
            vec = compute_variant_vector(pos_by_variant[vi], neg_by_variant[vi])
            if vec is not None and layer < vec.shape[0]:
                vecs[vi] = vec[layer]
                variant_vectors[trait][vi][persona] = vec[layer]

        if len(vecs) < 2:
            continue

        # Pairwise cosine similarity across variants (within this persona)
        variant_ids = sorted(vecs.keys())
        n_v = len(variant_ids)
        sim_matrix = np.zeros((n_v, n_v))
        for i, vi in enumerate(variant_ids):
            for j, vj in enumerate(variant_ids):
                sim_matrix[i, j] = cosine_similarity(vecs[vi], vecs[vj])

        # Off-diagonal mean = cross-variant similarity
        off_diag = []
        for i in range(n_v):
            for j in range(i + 1, n_v):
                off_diag.append(sim_matrix[i, j])

        per_pair_results[f"{persona}_{trait}"] = {
            "n_variants": n_v,
            "variant_indices": variant_ids,
            "cross_variant_cosine_mean": float(np.mean(off_diag)),
            "cross_variant_cosine_std": float(np.std(off_diag)),
            "cross_variant_cosine_min": float(np.min(off_diag)),
            "sim_matrix": sim_matrix.tolist(),
            "n_pos_per_variant": {vi: len(pos_by_variant.get(vi, [])) for vi in variant_ids},
            "n_neg_per_variant": {vi: len(neg_by_variant.get(vi, [])) for vi in variant_ids},
        }

        log.info("  %s/%s: cross-variant cos = %.4f ± %.4f (n=%d variants)",
                 persona, trait, np.mean(off_diag), np.std(off_diag), n_v)

    save_json(per_pair_results, output_dir / "syntactic_invariance.json")

    # Per-trait summary
    trait_summary = {}
    personas_seen = set()
    for key, data in per_pair_results.items():
        persona, trait = key.rsplit("_", 1)  # This is imprecise for multi-word traits
        # Re-parse properly
        for tv in {t.value for t in Trait}:
            if key.endswith(f"_{tv}"):
                persona = key[:-(len(tv) + 1)]
                trait = tv
                break
        personas_seen.add(persona)
        trait_summary.setdefault(trait, []).append(data["cross_variant_cosine_mean"])

    for trait, vals in trait_summary.items():
        trait_summary[trait] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "n_personas": len(vals),
        }
    save_json(trait_summary, output_dir / "syntactic_by_trait.json")

    # Cross-persona comparison: for each variant, how similar are different personas?
    # This gives: within-persona-across-variant vs within-variant-across-persona
    cross_persona = {}
    for trait, variants in variant_vectors.items():
        cross_persona[trait] = {}
        for vi, persona_vecs in variants.items():
            slugs = sorted(persona_vecs.keys())
            if len(slugs) < 2:
                continue
            cross_sims = []
            for i in range(len(slugs)):
                for j in range(i + 1, len(slugs)):
                    cross_sims.append(cosine_similarity(persona_vecs[slugs[i]], persona_vecs[slugs[j]]))
            cross_persona[trait][vi] = {
                "cross_persona_cosine_mean": float(np.mean(cross_sims)),
                "cross_persona_cosine_std": float(np.std(cross_sims)),
                "n_personas": len(slugs),
            }

    save_json(cross_persona, output_dir / "cross_persona_per_variant.json")

    # Final comparison: within-persona-across-variant vs across-persona-within-variant
    within_persona = [d["cross_variant_cosine_mean"] for d in per_pair_results.values()]
    across_persona = []
    for trait_data in cross_persona.values():
        for v_data in trait_data.values():
            across_persona.append(v_data["cross_persona_cosine_mean"])

    comparison = {
        "within_persona_across_variant": {
            "mean": float(np.mean(within_persona)) if within_persona else None,
            "std": float(np.std(within_persona)) if within_persona else None,
        },
        "across_persona_within_variant": {
            "mean": float(np.mean(across_persona)) if across_persona else None,
            "std": float(np.std(across_persona)) if across_persona else None,
        },
    }
    save_json(comparison, output_dir / "invariance_comparison.json")

    log.info("=== Syntactic Invariance Summary ===")
    log.info("Within-persona across-variant:  %.4f ± %.4f",
             comparison["within_persona_across_variant"]["mean"] or 0,
             comparison["within_persona_across_variant"]["std"] or 0)
    log.info("Across-persona within-variant:  %.4f ± %.4f",
             comparison["across_persona_within_variant"]["mean"] or 0,
             comparison["across_persona_within_variant"]["std"] or 0)
    log.info("(Higher within-persona = representation tracks meaning, not phrasing)")
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
