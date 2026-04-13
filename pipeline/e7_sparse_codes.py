#!/usr/bin/env python3
"""E7: SAE sparse codes as persona trait fingerprints.

Extends E5 by keeping full contrastive sparse codes (not just top-K sets) and
using them to build a mechanistic persona landscape. Each persona's trait
profile becomes a sparse code over 131K SAE features, enabling:

  1. Sparse code cosine similarity across personas (should decay along E6 basin gradient)
  2. Feature classification: universal (active in all personas) vs discriminative
  3. Ratio of universal/discriminative features per trait (should correlate with shared variance)
  4. Trait dictionary atoms: clusters of co-activating features across personas

Usage:
    python pipeline/e7_sparse_codes.py \
        --activations-dir outputs/gemma-2-27b-it/activations \
        --sae-repo google/gemma-scope-27b-pt-res \
        --sae-folder layer_22/width_131k/average_l0_82

    # With pre-downloaded SAE:
    python pipeline/e7_sparse_codes.py \
        --activations-dir outputs/gemma-2-27b-it/activations \
        --sae-path /path/to/params.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from persona_steering.config import (
    BASIN_GRADIENTS,
    Trait,
    TARGET_LAYER,
    PERSONA_SLUGS,
)
from persona_steering.utils import log, save_json
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_summary
from persona_steering.sae_loader import JumpReLUSAE, download_sae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E7: SAE sparse code persona fingerprints")
    parser.add_argument("--activations-dir", type=str, required=True)
    parser.add_argument("--sae-path", type=str, default=None,
                        help="Path to pre-downloaded params.npz")
    parser.add_argument("--sae-repo", type=str, default="google/gemma-scope-27b-pt-res")
    parser.add_argument("--sae-folder", type=str,
                        default="layer_22/width_131k/average_l0_82")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--universality-threshold", type=float, default=0.8,
                        help="Fraction of personas a feature must be active in to count as universal (default: 0.8)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Activation loading
# ---------------------------------------------------------------------------

def discover_pairs(activations_dir: Path) -> dict[str, dict[str, tuple[Path, Path]]]:
    """Find pos/neg activation file pairs, grouped by persona and trait."""
    trait_values = {t.value for t in Trait}
    pairs: dict[str, dict[str, tuple[Path, Path]]] = {}
    seen = set()

    for f in sorted(activations_dir.glob("*.pt")):
        stem = f.stem
        for direction in ("_pos", "_neg"):
            if not stem.endswith(direction):
                continue
            base = stem[: -len(direction)]
            for tv in trait_values:
                if base.endswith(f"_{tv}"):
                    persona = base[: -(len(tv) + 1)]
                    if base not in seen:
                        seen.add(base)
                        pos_path = activations_dir / f"{base}_pos.pt"
                        neg_path = activations_dir / f"{base}_neg.pt"
                        if pos_path.exists() and neg_path.exists():
                            pairs.setdefault(persona, {})[tv] = (pos_path, neg_path)
                    break

    return pairs


def load_mean_activation(path: Path, layer: int) -> torch.Tensor:
    """Load activation file and return mean activation at given layer."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    _clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    vecs = [_clean(v[layer].float()) for v in data.values()]
    return torch.stack(vecs).mean(dim=0)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_contrastive_sparse_codes(
    pairs: dict[str, dict[str, tuple[Path, Path]]],
    sae: JumpReLUSAE,
    layer: int,
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute contrastive sparse codes: SAE(pos_mean) - SAE(neg_mean) per persona x trait.

    Returns:
        persona -> trait -> (d_sae,) contrastive sparse code tensor
    """
    codes: dict[str, dict[str, torch.Tensor]] = {}

    for persona, trait_pairs in sorted(pairs.items()):
        codes[persona] = {}
        for trait, (pos_path, neg_path) in sorted(trait_pairs.items()):
            pos_mean = load_mean_activation(pos_path, layer).to(device)
            neg_mean = load_mean_activation(neg_path, layer).to(device)

            pos_features = sae.encode(pos_mean.unsqueeze(0)).squeeze(0)
            neg_features = sae.encode(neg_mean.unsqueeze(0)).squeeze(0)

            contrastive = pos_features - neg_features  # (d_sae,)
            codes[persona][trait] = contrastive.cpu()

    return codes


def classify_features(
    codes: dict[str, dict[str, torch.Tensor]],
    trait: str,
    threshold: float,
    universality_fraction: float,
) -> dict:
    """Classify SAE features as universal, discriminative, or inactive for a trait.

    A feature is:
      - active for persona P if |contrastive_code[feature]| > threshold
      - universal if active in >= universality_fraction of personas
      - discriminative if active in 1-2 personas only
      - inactive if active in 0 personas
    """
    personas = [p for p in codes if trait in codes[p]]
    if not personas:
        return {}

    d_sae = codes[personas[0]][trait].shape[0]

    # Count how many personas each feature is active in
    active_counts = torch.zeros(d_sae)
    for persona in personas:
        code = codes[persona][trait]
        active_counts += (code.abs() > threshold).float()

    n_personas = len(personas)
    universal_min = int(n_personas * universality_fraction)

    universal_mask = active_counts >= universal_min
    discriminative_mask = (active_counts >= 1) & (active_counts <= 2)
    inactive_mask = active_counts == 0

    universal_indices = universal_mask.nonzero(as_tuple=True)[0].tolist()
    discriminative_indices = discriminative_mask.nonzero(as_tuple=True)[0].tolist()

    return {
        "n_universal": int(universal_mask.sum()),
        "n_discriminative": int(discriminative_mask.sum()),
        "n_inactive": int(inactive_mask.sum()),
        "n_total_active": int((active_counts > 0).sum()),
        "universal_ratio": int(universal_mask.sum()) / max(int((active_counts > 0).sum()), 1),
        "universal_indices": universal_indices[:200],  # cap for JSON
        "discriminative_indices": discriminative_indices[:200],
        "active_counts_histogram": {
            int(k): int(v) for k, v in
            zip(*torch.unique(active_counts[active_counts > 0], return_counts=True))
        },
    }


def sparse_code_similarity_matrix(
    codes: dict[str, dict[str, torch.Tensor]],
    trait: str,
) -> tuple[np.ndarray, list[str]]:
    """Cosine similarity matrix of contrastive sparse codes for a trait."""
    personas = sorted(p for p in codes if trait in codes[p])
    n = len(personas)
    matrix = np.zeros((n, n))

    for i, pa in enumerate(personas):
        for j, pb in enumerate(personas):
            a = codes[pa][trait].float()
            b = codes[pb][trait].float()
            sim = torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)
            matrix[i, j] = sim.item()

    return matrix, personas


def basin_gradient_analysis(
    codes: dict[str, dict[str, torch.Tensor]],
    gradients: dict[str, list[tuple[str, int]]],
) -> dict:
    """Test whether sparse code similarity decays along basin gradients.

    Mirrors E6 analysis but in SAE feature space instead of steering vector space.
    """
    results = {}

    for trait, gradient in gradients.items():
        # Get default sparse code as reference
        if "default" not in codes or trait not in codes.get("default", {}):
            log.warning("No default sparse code for %s, skipping basin analysis", trait)
            continue

        default_code = codes["default"][trait].float()

        persona_sims = []
        for slug, ring in gradient:
            if slug == "default":
                continue
            if slug not in codes or trait not in codes[slug]:
                continue
            code = codes[slug][trait].float()
            sim = torch.dot(code, default_code) / (code.norm() * default_code.norm() + 1e-8)
            persona_sims.append({
                "persona": slug,
                "ring": ring,
                "sparse_code_sim_to_default": sim.item(),
            })

        if len(persona_sims) < 3:
            continue

        rings = np.array([p["ring"] for p in persona_sims])
        sims = np.array([p["sparse_code_sim_to_default"] for p in persona_sims])
        rho, p_val = spearmanr(rings, sims)

        results[trait] = {
            "personas": persona_sims,
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "mean_similarity": float(sims.mean()),
        }

        log.info("  %s basin gradient: rho=%.4f, p=%.4f", trait, rho, p_val)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    activations_dir = Path(args.activations_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else activations_dir.parent / "experiments" / "e7_sparse_codes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    # Load SAE — handles both Gemma Scope 1 (params.npz) and Gemma Scope 2 (params.safetensors)
    if args.sae_path:
        sae_path = args.sae_path
    else:
        sae_path = download_sae(args.sae_repo, args.sae_folder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE(sae_path, device=device)

    short = Path(args.activations_dir).parent.name
    init_run("e7_sparse_codes", short, config=vars(args))

    # Discover activation pairs
    pairs = discover_pairs(activations_dir)
    personas = sorted(pairs.keys())
    traits = sorted({t for p in pairs.values() for t in p.keys()})
    log.info("Found %d personas, %d traits", len(personas), len(traits))

    # 1. Compute contrastive sparse codes for all persona x trait combos
    log.info("Computing contrastive sparse codes...")
    codes = compute_contrastive_sparse_codes(pairs, sae, layer, device)

    # Save sparse codes as .pt (useful for downstream analysis)
    codes_for_save = {
        f"{persona}_{trait}": code.half()
        for persona, trait_codes in codes.items()
        for trait, code in trait_codes.items()
    }
    torch.save(codes_for_save, output_dir / "contrastive_sparse_codes.pt")
    log.info("Saved %d sparse codes to contrastive_sparse_codes.pt", len(codes_for_save))

    # 2. Per-trait analysis
    all_results = {}

    # Compute a sensible activation threshold from the data
    # Use median of non-zero absolute contrastive values
    all_nonzero = []
    for persona in codes:
        for trait in codes[persona]:
            nz = codes[persona][trait][codes[persona][trait].abs() > 0].abs()
            if len(nz) > 0:
                all_nonzero.append(nz.median().item())
    activation_threshold = np.median(all_nonzero) if all_nonzero else 0.1
    log.info("Activation threshold (median of medians): %.4f", activation_threshold)

    for trait in traits:
        log.info("=" * 50)
        log.info("Trait: %s", trait)

        # Similarity matrix
        sim_matrix, trait_personas = sparse_code_similarity_matrix(codes, trait)
        mean_off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / max(len(trait_personas) * (len(trait_personas) - 1), 1)
        log.info("  Sparse code similarity: mean off-diagonal = %.4f", mean_off_diag)

        # Feature classification
        classification = classify_features(
            codes, trait, activation_threshold, args.universality_threshold,
        )
        if classification:
            log.info("  Universal features: %d, Discriminative: %d, Total active: %d",
                     classification["n_universal"], classification["n_discriminative"],
                     classification["n_total_active"])
            log.info("  Universal ratio: %.4f", classification["universal_ratio"])

        # Sparsity stats
        sparsity_stats = {}
        for persona in trait_personas:
            code = codes[persona][trait]
            n_active = (code.abs() > activation_threshold).sum().item()
            sparsity_stats[persona] = {
                "n_active_features": int(n_active),
                "sparsity": 1.0 - n_active / sae.d_sae,
                "code_norm": code.norm().item(),
            }

        all_results[trait] = {
            "personas": trait_personas,
            "similarity_matrix": sim_matrix.tolist(),
            "mean_off_diagonal_similarity": float(mean_off_diag),
            "feature_classification": classification,
            "sparsity_stats": sparsity_stats,
            "mean_sparsity": float(np.mean([s["sparsity"] for s in sparsity_stats.values()])),
            "mean_active_features": float(np.mean([s["n_active_features"] for s in sparsity_stats.values()])),
        }

        log_metrics({
            f"e7/{trait}/mean_off_diag_sim": float(mean_off_diag),
            f"e7/{trait}/n_universal": classification.get("n_universal", 0),
            f"e7/{trait}/n_discriminative": classification.get("n_discriminative", 0),
            f"e7/{trait}/universal_ratio": classification.get("universal_ratio", 0),
        })

    # 3. Basin gradient analysis (sparse code version)
    log.info("=" * 50)
    log.info("Basin gradient analysis (sparse code space)")
    basin_results = basin_gradient_analysis(codes, BASIN_GRADIENTS)

    # 4. Cross-method comparison: do universal/discriminative ratios
    # correlate with shared variance from the main decomposition?
    log.info("=" * 50)
    log.info("SUMMARY")
    log.info("%-16s %8s %8s %8s %8s", "Trait", "OffDiag", "Univ", "Discrim", "UnivRatio")
    for trait in traits:
        r = all_results[trait]
        fc = r["feature_classification"]
        log.info("%-16s %8.4f %8d %8d %8.4f",
                 trait, r["mean_off_diagonal_similarity"],
                 fc.get("n_universal", 0), fc.get("n_discriminative", 0),
                 fc.get("universal_ratio", 0))

    if basin_results:
        log.info("")
        log.info("Basin gradient (sparse code space):")
        log.info("%-16s %8s %8s", "Trait", "Rho", "p-value")
        for trait, br in basin_results.items():
            log.info("%-16s %8.4f %8.4f", trait, br["spearman_rho"], br["spearman_p"])

    # Save everything
    save_json(all_results, output_dir / "sparse_code_analysis.json")
    save_json(basin_results, output_dir / "sparse_code_basin.json")
    save_json({
        "layer": layer,
        "sae_repo": args.sae_repo,
        "sae_folder": args.sae_folder,
        "d_sae": sae.d_sae,
        "activation_threshold": float(activation_threshold),
        "universality_threshold": args.universality_threshold,
        "device": device,
    }, output_dir / "e7_meta.json")

    # W&B summary
    summary = {}
    for trait, r in all_results.items():
        summary[f"e7/{trait}/universal_ratio"] = r["feature_classification"].get("universal_ratio", 0)
        summary[f"e7/{trait}/mean_off_diag_sim"] = r["mean_off_diagonal_similarity"]
    for trait, br in basin_results.items():
        summary[f"e7/{trait}/basin_rho"] = br["spearman_rho"]
    log_summary(summary)
    finish_run()

    log.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
