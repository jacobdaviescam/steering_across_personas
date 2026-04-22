#!/usr/bin/env python3
"""E6: Basin geometry — test monotonic decay of cosine similarity along trait gradients.

Hypothesis: Trait representations form basins in activation space. Personas whose
relationship to a trait is semantically close to the default assistant sit near the
basin floor (high cosine similarity). Personas further away sit near the peaks.

For each trait gradient defined in config.BASIN_GRADIENTS:
  1. Load vectors for all gradient personas + default
  2. Compute cosine_sim(v_{trait,persona}, v_{trait,default}) for each persona
  3. Compute Spearman rank correlation between ring and cosine similarity
  4. Permutation test: shuffle ring assignments 10K times for null distribution
  5. Cross-trait control: check that ring ordering is trait-specific

Usage:
    python pipeline/e6_basin_geometry.py --vectors-dir outputs/gemma-2-27b-it/vectors --layer 22
    python pipeline/e6_basin_geometry.py --vectors-dir outputs/gemma-2-27b-it/vectors --layer 22 --n-permutations 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from persona_steering.config import (
    BASIN_GRADIENTS,
    Trait,
    TARGET_LAYER,
    ANALYSIS_SUBDIR,
)
from persona_steering.utils import (
    cosine_similarity,
    log,
    save_json,
    parse_persona_trait_from_stem,
)
from persona_steering.wandb_utils import (
    init_run,
    finish_run,
    log_metrics,
    log_summary,
    ensure_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6: Basin geometry analysis")
    parser.add_argument(
        "--vectors-dir", type=str, required=True,
        help="Directory containing vector .pt files from step 3",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: sibling 'analysis/basin' dir)",
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Target layer (default: {TARGET_LAYER})",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=10_000,
        help="Number of permutations for null distribution (default: 10000)",
    )
    return parser.parse_args()


def load_vector(vectors_dir: Path, persona: str, trait: str, layer: int) -> torch.Tensor | None:
    """Load a single persona x trait vector at a given layer."""
    path = vectors_dir / f"{persona}_{trait}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    vec = data["vector"]  # (n_layers, hidden_dim)
    if layer >= vec.shape[0]:
        log.warning("Layer %d out of range for %s (max %d)", layer, path.name, vec.shape[0] - 1)
        return None
    return vec[layer].float()


def compute_basin_similarities(
    vectors_dir: Path,
    trait: str,
    gradient: list[tuple[str, int]],
    layer: int,
) -> dict:
    """Compute cosine similarity of each gradient persona to the default.

    Returns dict with per-persona results and summary statistics.
    """
    # Load default vector as reference
    default_vec = load_vector(vectors_dir, "default", trait, layer)
    if default_vec is None:
        log.error("Missing default vector for trait %s", trait)
        return {}

    results = []
    for slug, ring in gradient:
        if slug == "default":
            continue  # skip self-comparison
        vec = load_vector(vectors_dir, slug, trait, layer)
        if vec is None:
            log.warning("Missing vector for %s/%s, skipping", slug, trait)
            continue
        sim = cosine_similarity(vec, default_vec)
        results.append({
            "persona": slug,
            "ring": ring,
            "cosine_sim_to_default": sim,
            "vector_norm": vec.norm().item(),
        })

    return results


def permutation_test(
    rings: np.ndarray,
    similarities: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    """Permutation test for Spearman correlation between ring and similarity.

    Returns observed rho, p-value, and null distribution statistics.
    """
    observed_rho, _ = spearmanr(rings, similarities)

    null_rhos = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(rings)
        null_rhos[i], _ = spearmanr(perm, similarities)

    # Two-sided p-value
    p_value = np.mean(np.abs(null_rhos) >= np.abs(observed_rho))

    return {
        "observed_rho": float(observed_rho),
        "p_value": float(p_value),
        "null_mean": float(null_rhos.mean()),
        "null_std": float(null_rhos.std()),
        "null_percentile_5": float(np.percentile(null_rhos, 5)),
        "null_percentile_95": float(np.percentile(null_rhos, 95)),
    }


def count_monotonicity_violations(rings: np.ndarray, similarities: np.ndarray) -> int:
    """Count pairs where a higher ring has higher similarity (violates basin prediction)."""
    violations = 0
    n = len(rings)
    for i in range(n):
        for j in range(i + 1, n):
            if rings[i] < rings[j] and similarities[i] < similarities[j]:
                violations += 1
            elif rings[i] > rings[j] and similarities[i] > similarities[j]:
                violations += 1
    return violations


def cross_trait_control(
    vectors_dir: Path,
    gradients: dict[str, list[tuple[str, int]]],
    layer: int,
) -> dict:
    """Test that ring ordering is trait-specific.

    For each pair of traits, apply trait A's ring ordering to trait B's
    similarities. If basins are trait-specific, the Spearman correlation
    should be weak when using the wrong trait's ordering.
    """
    # First, load all similarities per trait
    trait_sims: dict[str, dict[str, float]] = {}
    for trait, gradient in gradients.items():
        default_vec = load_vector(vectors_dir, "default", trait, layer)
        if default_vec is None:
            continue
        sims = {}
        for slug, ring in gradient:
            if slug == "default":
                continue
            vec = load_vector(vectors_dir, slug, trait, layer)
            if vec is not None:
                sims[slug] = cosine_similarity(vec, default_vec)
        trait_sims[trait] = sims

    # For each pair: apply trait_a's ring ordering to trait_b's similarities
    results = {}
    for trait_a, gradient_a in gradients.items():
        for trait_b in gradients:
            if trait_a == trait_b:
                continue
            # Find shared personas between gradient_a ordering and trait_b sims
            ring_map = {slug: ring for slug, ring in gradient_a if slug != "default"}
            shared = [s for s in ring_map if s in trait_sims.get(trait_b, {})]
            if len(shared) < 3:
                continue
            rings = np.array([ring_map[s] for s in shared])
            sims = np.array([trait_sims[trait_b][s] for s in shared])
            rho, p = spearmanr(rings, sims)
            results[f"{trait_a}_ordering_on_{trait_b}_sims"] = {
                "rho": float(rho),
                "p_value": float(p),
                "n_shared": len(shared),
                "shared_personas": shared,
            }

    return results


def main() -> None:
    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    short = vectors_dir.parent.name
    vectors_dir = ensure_dir(f"{short}-vectors", vectors_dir, "*.pt")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = vectors_dir.parent / ANALYSIS_SUBDIR / "basin"
    output_dir.mkdir(parents=True, exist_ok=True)

    layer = args.layer
    rng = np.random.default_rng(42)

    init_run("e6_basin_geometry", short, config=vars(args))

    all_results = {}

    for trait, gradient in BASIN_GRADIENTS.items():
        log.info("=" * 60)
        log.info("Trait: %s (%d personas in gradient)", trait, len(gradient))
        log.info("=" * 60)

        # Compute similarities
        persona_results = compute_basin_similarities(vectors_dir, trait, gradient, layer)
        if not persona_results:
            log.warning("No results for %s, skipping", trait)
            continue

        rings = np.array([r["ring"] for r in persona_results])
        sims = np.array([r["cosine_sim_to_default"] for r in persona_results])

        # Spearman correlation
        rho, rho_p = spearmanr(rings, sims)
        log.info("Spearman rho = %.4f (p = %.4f)", rho, rho_p)

        # Permutation test
        perm_results = permutation_test(rings, sims, args.n_permutations, rng)
        log.info("Permutation p-value = %.4f (observed rho = %.4f)",
                 perm_results["p_value"], perm_results["observed_rho"])

        # Monotonicity violations
        n_pairs = len(rings) * (len(rings) - 1) // 2
        violations = count_monotonicity_violations(rings, sims)
        log.info("Monotonicity violations: %d / %d pairs (%.1f%%)",
                 violations, n_pairs, 100 * violations / max(n_pairs, 1))

        trait_result = {
            "trait": trait,
            "layer": layer,
            "n_personas": len(persona_results),
            "personas": persona_results,
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "permutation_test": perm_results,
            "monotonicity_violations": violations,
            "monotonicity_total_pairs": n_pairs,
            "monotonicity_violation_rate": violations / max(n_pairs, 1),
            "mean_similarity": float(sims.mean()),
            "std_similarity": float(sims.std()),
            "similarity_range": [float(sims.min()), float(sims.max())],
        }
        all_results[trait] = trait_result

        log_metrics({
            f"basin/{trait}/spearman_rho": float(rho),
            f"basin/{trait}/perm_p_value": perm_results["p_value"],
            f"basin/{trait}/violation_rate": violations / max(n_pairs, 1),
            f"basin/{trait}/mean_similarity": float(sims.mean()),
        })

        # Print gradient
        log.info("Gradient (ring -> cosine_sim):")
        for r in sorted(persona_results, key=lambda x: x["ring"]):
            log.info("  ring %2d  %-20s  cos_sim = %.4f",
                     r["ring"], r["persona"], r["cosine_sim_to_default"])

    # Cross-trait control
    log.info("=" * 60)
    log.info("Cross-trait control analysis")
    log.info("=" * 60)
    cross_results = cross_trait_control(vectors_dir, BASIN_GRADIENTS, layer)
    for key, val in cross_results.items():
        log.info("  %s: rho = %.4f (p = %.4f, n = %d)",
                 key, val["rho"], val["p_value"], val["n_shared"])

    # Save everything
    save_json(all_results, output_dir / "basin_results.json")
    save_json(cross_results, output_dir / "cross_trait_control.json")

    # Summary table
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("%-15s  %8s  %8s  %10s  %10s", "Trait", "Rho", "Perm-p", "Violations", "Mean Sim")
    for trait, res in all_results.items():
        log.info("%-15s  %8.4f  %8.4f  %5d/%-5d  %10.4f",
                 trait, res["spearman_rho"], res["permutation_test"]["p_value"],
                 res["monotonicity_violations"], res["monotonicity_total_pairs"],
                 res["mean_similarity"])

    log_summary({
        f"basin/{t}/rho": r["spearman_rho"]
        for t, r in all_results.items()
    })
    finish_run()

    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
