#!/usr/bin/env python3
"""Convergence analysis: how many activation pairs are needed for stable vectors?

For each persona x trait, computes vectors using subsets of increasing size
(1, 2, 5, 10, 20, 50, 100 pairs) and measures cosine similarity to the
full-data vector.  Also checks when transfer-matrix clusters stabilize.

Outputs:
  - Per (persona, trait): convergence curve (N -> cosine to full vector)
  - Transfer matrices at each subset size
  - Cluster stability (ARI) vs full-data clusters at each N

Usage:
    python pipeline/r2_convergence.py \
        --activations-dir outputs/gemma-2-27b-it/activations \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --output-dir outputs/gemma-2-27b-it/robustness/convergence \
        --layer 22
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from persona_steering.config import Trait, OUTPUTS_DIR, TARGET_LAYER
from persona_steering.analysis import build_transfer_matrix, cluster_persona_vectors
from persona_steering.utils import log, save_json, cosine_similarity


SUBSET_SIZES = [1, 2, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convergence analysis: vector stability vs dataset size"
    )
    parser.add_argument(
        "--activations-dir", type=str, required=True,
        help="Directory with activation .pt files from step 2",
    )
    parser.add_argument(
        "--vectors-dir", type=str, required=True,
        help="Directory with full-data vector .pt files from step 3",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
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


class _Shim:
    """Minimal vector wrapper for analysis functions."""
    def __init__(self, vector, persona, trait, layer):
        self.vector = vector
        self.persona = persona
        self.trait = trait
        self.layer = layer


def compute_subset_vector(
    pos_acts: list[torch.Tensor],
    neg_acts: list[torch.Tensor],
    n: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Compute contrastive vector from first n pos and neg activations (shuffled)."""
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

    pairs = discover_pairs(activations_dir)
    if not pairs:
        log.error("No activation pairs found")
        return

    # Load full-data vectors
    full_vectors: dict[tuple[str, str], torch.Tensor] = {}
    for persona, trait, _, _ in pairs:
        path = vectors_dir / f"{persona}_{trait}.pt"
        if path.exists():
            data = torch.load(path, map_location="cpu", weights_only=False)
            vec = data["vector"].float()
            if layer < vec.shape[0]:
                full_vectors[(persona, trait)] = vec[layer]

    # Load all activation data
    all_activations: dict[tuple[str, str], tuple[list, list]] = {}
    for persona, trait, pos_path, neg_path in pairs:
        pos_data = torch.load(pos_path, map_location="cpu", weights_only=True)
        neg_data = torch.load(neg_path, map_location="cpu", weights_only=True)
        all_activations[(persona, trait)] = (list(pos_data.values()), list(neg_data.values()))

    log.info("Loaded %d pairs, computing convergence curves...", len(pairs))

    personas = sorted({p for p, _ in all_activations})
    traits = sorted({Trait(t) for _, t in all_activations}, key=lambda t: t.value)

    # Filter subset sizes to what's achievable
    max_n = min(
        min(len(pos) for pos, neg in all_activations.values()),
        min(len(neg) for pos, neg in all_activations.values()),
    )
    sizes = [s for s in SUBSET_SIZES if s <= max_n]
    if max_n not in sizes:
        sizes.append(max_n)
    log.info("Subset sizes: %s (max available: %d)", sizes, max_n)

    # Per-pair convergence curves
    convergence = {}
    # Subset vectors for transfer matrix computation
    subset_vectors_by_n: dict[int, dict[str, dict[Trait, dict[int, _Shim]]]] = {}

    for persona, trait, _, _ in pairs:
        pos_acts, neg_acts = all_activations[(persona, trait)]
        full_vec = full_vectors.get((persona, trait))
        key = f"{persona}_{trait}"

        curve = {}
        for n in sizes:
            vec = compute_subset_vector(pos_acts, neg_acts, n, rng)
            layer_vec = vec[layer] if layer < vec.shape[0] else None

            if layer_vec is not None and full_vec is not None:
                cos = cosine_similarity(layer_vec, full_vec)
                curve[n] = cos

                # Store for transfer matrix
                if n not in subset_vectors_by_n:
                    subset_vectors_by_n[n] = {}
                shim = _Shim(layer_vec, persona, Trait(trait), layer)
                subset_vectors_by_n[n].setdefault(persona, {}).setdefault(Trait(trait), {})[layer] = shim

        convergence[key] = curve

    save_json(convergence, output_dir / "convergence_curves.json")

    # Per-trait average convergence
    trait_convergence = {}
    for trait_enum in traits:
        tv = trait_enum.value
        trait_convergence[tv] = {}
        for n in sizes:
            cosines = [convergence[f"{p}_{tv}"].get(n) for p in personas
                       if f"{p}_{tv}" in convergence and n in convergence[f"{p}_{tv}"]]
            cosines = [c for c in cosines if c is not None]
            if cosines:
                trait_convergence[tv][n] = {
                    "mean": float(np.mean(cosines)),
                    "std": float(np.std(cosines)),
                    "min": float(np.min(cosines)),
                }
    save_json(trait_convergence, output_dir / "convergence_by_trait.json")

    # Transfer matrix at each subset size + cluster stability
    # First, compute full-data transfer matrix and clusters for reference
    full_nested: dict[str, dict[Trait, dict[int, _Shim]]] = {}
    for (persona, trait), vec in full_vectors.items():
        shim = _Shim(vec, persona, Trait(trait), layer)
        full_nested.setdefault(persona, {}).setdefault(Trait(trait), {})[layer] = shim

    full_tm = build_transfer_matrix(full_nested, personas, traits, layer)
    full_clustering = cluster_persona_vectors(full_tm, personas)
    full_labels = full_clustering["labels"]

    from sklearn.metrics import adjusted_rand_score

    transfer_stability = {}
    for n in sizes:
        if n not in subset_vectors_by_n:
            continue
        nested = subset_vectors_by_n[n]
        avail_personas = [p for p in personas if p in nested]
        if len(avail_personas) < 2:
            continue

        tm = build_transfer_matrix(nested, avail_personas, traits, layer)
        clustering = cluster_persona_vectors(tm, avail_personas)

        # Compare clusters to full-data
        labels_sub = [clustering["labels"].get(p, -1) for p in avail_personas]
        labels_full = [full_labels.get(p, -1) for p in avail_personas]
        ari = float(adjusted_rand_score(labels_full, labels_sub))

        # Frobenius distance to full transfer matrix
        # Align to same persona order
        full_tm_aligned = build_transfer_matrix(full_nested, avail_personas, traits, layer)
        frob = float(np.linalg.norm(tm - full_tm_aligned, "fro"))

        transfer_stability[n] = {
            "adjusted_rand_index": ari,
            "frobenius_distance": frob,
            "n_clusters": len(clustering["clusters"]),
        }

    save_json(transfer_stability, output_dir / "transfer_stability.json")

    # Summary
    log.info("=== Convergence Summary ===")
    for n in sizes:
        all_cos = [convergence[k].get(n) for k in convergence if n in convergence[k]]
        all_cos = [c for c in all_cos if c is not None]
        ts = transfer_stability.get(n, {})
        log.info("  N=%3d: mean cos=%.4f ± %.4f, ARI=%.3f, frob=%.3f",
                 n,
                 np.mean(all_cos) if all_cos else 0,
                 np.std(all_cos) if all_cos else 0,
                 ts.get("adjusted_rand_index", -1),
                 ts.get("frobenius_distance", -1))

    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
