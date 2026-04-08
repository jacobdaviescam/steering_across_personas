#!/usr/bin/env python3
"""E3: IV–CAA geometric decomposition.

For each trait × context, decomposes the CAA vector into:
  - A component along the IV vector direction
  - An orthogonal residual

Tests whether the residual direction is consistent across contexts
(would suggest a systematic context-modulation component captured by CAA
but not by IV).

Usage:
    python pipeline/e3_iv_caa_decomposition.py \
        --iv-dir outputs/gemma-2-27b-it/vectors \
        --caa-dir outputs/gemma-2-27b-it/caa_vectors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from persona_steering.config import Trait, TARGET_LAYER, PERSONA_SLUGS
from persona_steering.utils import log, cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IV–CAA geometric decomposition")
    parser.add_argument("--iv-dir", type=str, default="outputs/gemma-2-27b-it/vectors")
    parser.add_argument("--caa-dir", type=str, default="outputs/gemma-2-27b-it/caa_vectors")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def load_vector(path: Path, layer: int) -> torch.Tensor:
    """Load a single vector .pt file and return the layer slice."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["vector"][layer].float()


def main() -> None:
    args = parse_args()
    iv_dir = Path(args.iv_dir)
    caa_dir = Path(args.caa_dir)
    layer = args.layer
    output_dir = Path(args.output_dir) if args.output_dir else iv_dir.parent / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    traits = [t for t in Trait]
    personas = PERSONA_SLUGS

    results = {}

    for trait in traits:
        trait_results = {}
        residuals = {}  # persona -> residual vector (for cross-persona consistency check)

        for persona in personas:
            iv_path = iv_dir / f"{persona}_{trait.value}.pt"
            caa_path = caa_dir / f"{persona}_{trait.value}.pt"

            if not iv_path.exists() or not caa_path.exists():
                log.warning("Missing vector: %s or %s", iv_path, caa_path)
                continue

            v_iv = load_vector(iv_path, layer)
            v_caa = load_vector(caa_path, layer)

            # Cosine between IV and CAA
            cos_iv_caa = cosine_similarity(v_iv, v_caa)

            # Project CAA onto IV direction
            iv_unit = v_iv / v_iv.norm()
            proj_mag = torch.dot(v_caa, iv_unit).item()
            proj = proj_mag * iv_unit
            residual = v_caa - proj

            residuals[persona] = residual

            trait_results[persona] = {
                "cosine_iv_caa": cos_iv_caa,
                "iv_magnitude": v_iv.norm().item(),
                "caa_magnitude": v_caa.norm().item(),
                "projection_magnitude": abs(proj_mag),
                "residual_magnitude": residual.norm().item(),
                "projection_ratio": abs(proj_mag) / (v_caa.norm().item() + 1e-10),
                "residual_ratio": residual.norm().item() / (v_caa.norm().item() + 1e-10),
            }

        # Cross-context residual consistency: pairwise cosine among residuals
        persona_list = sorted(residuals.keys())
        residual_cosines = []
        for i in range(len(persona_list)):
            for j in range(i + 1, len(persona_list)):
                rc = cosine_similarity(residuals[persona_list[i]], residuals[persona_list[j]])
                residual_cosines.append(rc)

        # Mean magnitude of residual across contexts
        mean_residual_ratio = np.mean([
            trait_results[p]["residual_ratio"] for p in persona_list
        ])
        mean_cosine_iv_caa = np.mean([
            trait_results[p]["cosine_iv_caa"] for p in persona_list
        ])

        results[trait.value] = {
            "per_context": trait_results,
            "summary": {
                "mean_cosine_iv_caa": float(mean_cosine_iv_caa),
                "mean_residual_ratio": float(mean_residual_ratio),
                "residual_pairwise_cosine_mean": float(np.mean(residual_cosines)) if residual_cosines else None,
                "residual_pairwise_cosine_std": float(np.std(residual_cosines)) if residual_cosines else None,
                "residual_pairwise_cosine_min": float(np.min(residual_cosines)) if residual_cosines else None,
                "residual_pairwise_cosine_max": float(np.max(residual_cosines)) if residual_cosines else None,
                "n_contexts": len(persona_list),
            },
        }

    # Print summary
    print(f"{'Trait':<16} {'cos(IV,CAA)':>12} {'resid_ratio':>12} {'resid_cos_mean':>15} {'resid_cos_std':>14}")
    print("-" * 75)
    for trait in traits:
        s = results[trait.value]["summary"]
        print(
            f"{trait.value:<16} "
            f"{s['mean_cosine_iv_caa']:>12.3f} "
            f"{s['mean_residual_ratio']:>12.3f} "
            f"{s['residual_pairwise_cosine_mean']:>15.3f} "
            f"{s['residual_pairwise_cosine_std']:>14.3f}"
        )

    print("\nInterpretation:")
    print("- cos(IV,CAA): How aligned the two extraction methods are (higher = more similar)")
    print("- resid_ratio: Fraction of CAA vector orthogonal to IV (higher = CAA captures more beyond IV)")
    print("- resid_cos_mean: Consistency of the CAA residual across contexts")
    print("  (high = systematic component; low = noise or context-specific)")

    # Save JSON
    output_path = output_dir / "iv_caa_decomposition.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Generate figure: bar chart of cos(IV,CAA) and residual ratio per trait
    trait_names = [t.value.replace("_", " ") for t in traits]
    cos_vals = [results[t.value]["summary"]["mean_cosine_iv_caa"] for t in traits]
    resid_vals = [results[t.value]["summary"]["mean_residual_ratio"] for t in traits]
    resid_cos_vals = [results[t.value]["summary"]["residual_pairwise_cosine_mean"] for t in traits]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: IV-CAA cosine per trait
    ax = axes[0]
    bars = ax.barh(trait_names, cos_vals, color="steelblue")
    ax.set_xlabel("Cosine Similarity")
    ax.set_title("IV–CAA Alignment")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    # Panel 2: Residual ratio per trait
    ax = axes[1]
    ax.barh(trait_names, resid_vals, color="coral")
    ax.set_xlabel("Residual / CAA Magnitude")
    ax.set_title("CAA Beyond-IV Component")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    # Panel 3: Residual consistency per trait
    ax = axes[2]
    ax.barh(trait_names, resid_cos_vals, color="mediumpurple")
    ax.set_xlabel("Mean Pairwise Cosine")
    ax.set_title("Residual Cross-Context Consistency")
    ax.set_xlim(-0.2, 1)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.invert_yaxis()

    fig.suptitle("IV–CAA Geometric Decomposition (Gemma-2-27B-IT, Layer 22)", fontsize=13)
    fig.tight_layout()

    fig_path = output_dir / "iv_caa_decomposition.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
