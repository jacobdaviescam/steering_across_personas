#!/usr/bin/env python3
"""E2: Bootstrap confidence intervals on shared variance ratio (rho_t).

Resamples contrastive activation pairs with replacement, recomputes
contrastive vectors per context, and recomputes rho_t. Reports 95% CIs.

Usage:
    python pipeline/e2_bootstrap_rho.py --activations-dir outputs/gemma-2-27b-it/activations
    python pipeline/e2_bootstrap_rho.py --activations-dir outputs/gemma-2-27b-it/activations \
        --caa-activations-dir outputs/gemma-2-27b-it/caa_activations
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from persona_steering.config import Trait, TARGET_LAYER, PERSONA_SLUGS
from persona_steering.utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap CIs on shared variance ratio")
    parser.add_argument("--activations-dir", type=str, required=True)
    parser.add_argument("--caa-activations-dir", type=str, default=None,
                        help="CAA activation directory (optional, for dual-method CIs)")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def discover_pairs(activations_dir: Path) -> dict[str, dict[str, tuple[Path, Path]]]:
    """Find pos/neg activation file pairs, organised by persona x trait.

    Returns:
        {persona: {trait_value: (pos_path, neg_path)}}
    """
    trait_values = {t.value for t in Trait}
    files = {f.stem: f for f in activations_dir.glob("*.pt")}

    pairs: dict[str, dict[str, tuple[Path, Path]]] = {}
    seen = set()

    for stem, path in files.items():
        for direction in ("_pos", "_neg"):
            if not stem.endswith(direction):
                continue
            base = stem[: -len(direction)]

            # Match trait
            persona_slug = None
            trait_name = None
            for tv in trait_values:
                if base.endswith(f"_{tv}"):
                    persona_slug = base[: -(len(tv) + 1)]
                    trait_name = tv
                    break

            if persona_slug is None or trait_name is None or base in seen:
                continue
            seen.add(base)

            pos_path = activations_dir / f"{base}_pos.pt"
            neg_path = activations_dir / f"{base}_neg.pt"
            if pos_path.exists() and neg_path.exists():
                pairs.setdefault(persona_slug, {})[trait_name] = (pos_path, neg_path)

    return pairs


def bootstrap_contrastive_vector(
    pos_data: dict, neg_data: dict, layer: int, rng: np.random.Generator
) -> torch.Tensor:
    """Compute a contrastive vector from bootstrap-resampled activation pairs."""
    _clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    pos_keys = list(pos_data.keys())
    neg_keys = list(neg_data.keys())

    # Resample with replacement
    pos_idx = rng.choice(len(pos_keys), size=len(pos_keys), replace=True)
    neg_idx = rng.choice(len(neg_keys), size=len(neg_keys), replace=True)

    pos_sum = torch.zeros_like(_clean(pos_data[pos_keys[0]][layer].float()))
    for i in pos_idx:
        pos_sum += _clean(pos_data[pos_keys[i]][layer].float())

    neg_sum = torch.zeros_like(pos_sum)
    for i in neg_idx:
        neg_sum += _clean(neg_data[neg_keys[i]][layer].float())

    return (pos_sum / len(pos_keys)) - (neg_sum / len(neg_keys))


def compute_rho(vectors: dict[str, torch.Tensor]) -> float:
    """Compute shared variance ratio from dict of persona -> vector."""
    vecs = torch.stack(list(vectors.values()))
    unit_vecs = vecs / vecs.norm(dim=1, keepdim=True)
    mean_dir = unit_vecs.mean(dim=0)
    shared_unit = mean_dir / mean_dir.norm()

    total_sq = sum(v.norm().item() ** 2 for v in vectors.values())
    shared_sq = sum(torch.dot(v, shared_unit).item() ** 2 for v in vectors.values())
    return shared_sq / (total_sq + 1e-10)


def main() -> None:
    args = parse_args()
    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir) if args.output_dir else activations_dir.parent / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    layer = args.layer
    n_boot = args.n_bootstrap
    rng = np.random.default_rng(args.seed)

    pairs = discover_pairs(activations_dir)
    personas = sorted(pairs.keys())
    traits = sorted({t for p in pairs.values() for t in p.keys()})

    log.info("Found %d personas, %d traits", len(personas), len(traits))
    log.info("Running %d bootstrap iterations at layer %d", n_boot, layer)

    # Pre-load all activation data
    log.info("Loading activation files...")
    activation_data: dict[str, dict[str, tuple[dict, dict]]] = {}
    for persona in personas:
        activation_data[persona] = {}
        for trait in traits:
            if trait not in pairs[persona]:
                continue
            pos_path, neg_path = pairs[persona][trait]
            pos_data = torch.load(pos_path, map_location="cpu", weights_only=True)
            neg_data = torch.load(neg_path, map_location="cpu", weights_only=True)
            activation_data[persona][trait] = (pos_data, neg_data)
    log.info("Loaded all activations")

    # Compute point estimate rho_t
    point_estimates: dict[str, float] = {}
    for trait in traits:
        vectors = {}
        for persona in personas:
            if trait not in activation_data[persona]:
                continue
            pos_data, neg_data = activation_data[persona][trait]
            _clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            pos_keys = list(pos_data.keys())
            neg_keys = list(neg_data.keys())
            pos_mean = sum(_clean(pos_data[k][layer].float()) for k in pos_keys) / len(pos_keys)
            neg_mean = sum(_clean(neg_data[k][layer].float()) for k in neg_keys) / len(neg_keys)
            vectors[persona] = pos_mean - neg_mean
        point_estimates[trait] = compute_rho(vectors)

    # Bootstrap
    results = {}
    for trait in traits:
        log.info("Bootstrapping %s...", trait)
        boot_rhos = []
        for b in range(n_boot):
            vectors = {}
            for persona in personas:
                if trait not in activation_data[persona]:
                    continue
                pos_data, neg_data = activation_data[persona][trait]
                vectors[persona] = bootstrap_contrastive_vector(
                    pos_data, neg_data, layer, rng
                )
            boot_rhos.append(compute_rho(vectors))

        boot_rhos = np.array(boot_rhos)
        results[trait] = {
            "point_estimate": point_estimates[trait],
            "bootstrap_mean": float(np.mean(boot_rhos)),
            "bootstrap_std": float(np.std(boot_rhos)),
            "ci_lower": float(np.percentile(boot_rhos, 2.5)),
            "ci_upper": float(np.percentile(boot_rhos, 97.5)),
            "n_bootstrap": n_boot,
        }

    # Print results
    print(f"\n{'Trait':<16} {'ρ_t':>8} {'95% CI':>20} {'±':>8}")
    print("-" * 58)
    for trait in traits:
        r = results[trait]
        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        print(f"{trait:<16} {r['point_estimate']:>8.3f} {ci_str:>20} {r['bootstrap_std']:>8.4f}")

    # Save
    output_path = output_dir / "bootstrap_rho.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
