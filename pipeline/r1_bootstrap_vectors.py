#!/usr/bin/env python3
"""Bootstrap stability analysis for steering vectors.

For each persona x trait, resample activation pairs with replacement and
recompute the contrastive vector.  Measures how stable vectors are under
dataset perturbation.

Outputs:
  - Per (persona, trait): pairwise cosine sim across bootstrap resamples
  - Per (persona, trait): mean cosine to full-data vector
  - Aggregate summary statistics
  - Transfer matrix confidence intervals (via bootstrapped transfer matrices)

Usage:
    python pipeline/r1_bootstrap_vectors.py \
        --activations-dir outputs/gemma-2-27b-it/activations \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --output-dir outputs/gemma-2-27b-it/robustness/bootstrap \
        --n-bootstraps 50 --layer 22
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from persona_steering.config import Trait, OUTPUTS_DIR, TARGET_LAYER
from persona_steering.utils import log, save_json, cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap stability analysis for steering vectors"
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
        help="Output directory (default: outputs/{model}/robustness/bootstrap)",
    )
    parser.add_argument(
        "--n-bootstraps", type=int, default=50,
        help="Number of bootstrap resamples (default: 50)",
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Layer for transfer matrix analysis (default: {TARGET_LAYER})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
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


def bootstrap_contrastive_vector(
    pos_acts: list[torch.Tensor],
    neg_acts: list[torch.Tensor],
    rng: np.random.Generator,
) -> torch.Tensor:
    """Compute a contrastive vector from a bootstrap resample of activations."""
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

    pairs = discover_pairs(activations_dir)
    if not pairs:
        log.error("No activation pairs found in %s", activations_dir)
        return

    log.info("Found %d persona x trait pairs, running %d bootstraps each", len(pairs), n_boots)

    results = {}

    for persona, trait, pos_path, neg_path in pairs:
        log.info("Bootstrapping %s/%s...", persona, trait)

        # Load activations as lists
        pos_data = torch.load(pos_path, map_location="cpu", weights_only=True)
        neg_data = torch.load(neg_path, map_location="cpu", weights_only=True)
        pos_acts = list(pos_data.values())
        neg_acts = list(neg_data.values())

        if len(pos_acts) < 2 or len(neg_acts) < 2:
            log.warning("Too few activations for %s/%s (pos=%d, neg=%d), skipping",
                        persona, trait, len(pos_acts), len(neg_acts))
            continue

        # Load the full-data vector for comparison
        full_vec_path = vectors_dir / f"{persona}_{trait}.pt"
        if full_vec_path.exists():
            full_data = torch.load(full_vec_path, map_location="cpu", weights_only=False)
            full_vec = full_data["vector"].float()  # (n_layers-1, hidden_dim)
            if layer < full_vec.shape[0]:
                full_vec_layer = full_vec[layer]
            else:
                full_vec_layer = None
        else:
            full_vec_layer = None

        # Generate bootstrap vectors
        boot_vectors = []
        for b in range(n_boots):
            bv = bootstrap_contrastive_vector(pos_acts, neg_acts, rng)
            boot_vectors.append(bv)

        # Pairwise cosine similarity between bootstrap vectors (at target layer)
        boot_layer_vecs = [bv[layer] for bv in boot_vectors if layer < bv.shape[0]]
        n_valid = len(boot_layer_vecs)

        if n_valid < 2:
            continue

        pairwise_sims = []
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                pairwise_sims.append(cosine_similarity(boot_layer_vecs[i], boot_layer_vecs[j]))

        # Cosine to full-data vector
        full_sims = []
        if full_vec_layer is not None:
            for bv in boot_layer_vecs:
                full_sims.append(cosine_similarity(bv, full_vec_layer))

        results[f"{persona}_{trait}"] = {
            "n_pos": len(pos_acts),
            "n_neg": len(neg_acts),
            "n_bootstraps": n_valid,
            "pairwise_cosine_mean": float(np.mean(pairwise_sims)),
            "pairwise_cosine_std": float(np.std(pairwise_sims)),
            "pairwise_cosine_min": float(np.min(pairwise_sims)),
            "full_data_cosine_mean": float(np.mean(full_sims)) if full_sims else None,
            "full_data_cosine_std": float(np.std(full_sims)) if full_sims else None,
        }

        log.info("  pairwise cos: %.4f ± %.4f, vs full: %.4f ± %.4f",
                 np.mean(pairwise_sims), np.std(pairwise_sims),
                 np.mean(full_sims) if full_sims else 0,
                 np.std(full_sims) if full_sims else 0)

    save_json(results, output_dir / "bootstrap_stability.json")

    # Aggregate summary
    all_pairwise = [r["pairwise_cosine_mean"] for r in results.values() if r["pairwise_cosine_mean"] is not None]
    all_full = [r["full_data_cosine_mean"] for r in results.values() if r["full_data_cosine_mean"] is not None]

    summary = {
        "n_pairs": len(results),
        "n_bootstraps": n_boots,
        "layer": layer,
        "mean_pairwise_stability": float(np.mean(all_pairwise)) if all_pairwise else None,
        "std_pairwise_stability": float(np.std(all_pairwise)) if all_pairwise else None,
        "mean_full_data_alignment": float(np.mean(all_full)) if all_full else None,
        "std_full_data_alignment": float(np.std(all_full)) if all_full else None,
    }
    save_json(summary, output_dir / "bootstrap_summary.json")

    log.info("=== Bootstrap Summary ===")
    log.info("Pairs: %d, Bootstraps: %d, Layer: %d", len(results), n_boots, layer)
    if all_pairwise:
        log.info("Pairwise stability: %.4f ± %.4f", np.mean(all_pairwise), np.std(all_pairwise))
    if all_full:
        log.info("Full-data alignment: %.4f ± %.4f", np.mean(all_full), np.std(all_full))
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
