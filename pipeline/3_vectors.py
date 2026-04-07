#!/usr/bin/env python3
"""Compute contrastive steering vectors: mean(pos activations) - mean(neg activations).

For each persona x trait:
  1. Load {persona}_{trait}_pos.pt and {persona}_{trait}_neg.pt
  2. Compute vector = mean(positive activations) - mean(negative activations)
  3. Save to outputs/{model}/vectors/{persona}_{trait}.pt

Usage:
    python pipeline/3_vectors.py --activations-dir outputs/gemma-2-9b-it/activations
    python pipeline/3_vectors.py --activations-dir outputs/gemma-2-9b-it/activations --output-dir outputs/gemma-2-9b-it/vectors
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from persona_steering.config import OUTPUTS_DIR
from persona_steering.utils import log
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_artifact, ensure_dir, infer_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute contrastive steering vectors from activation files"
    )
    parser.add_argument(
        "--activations-dir", type=str, required=True,
        help="Directory containing activation .pt files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for vector .pt files (default: sibling 'vectors' dir)",
    )
    return parser.parse_args()


def discover_pairs(activations_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """Find matching pos/neg activation file pairs.

    Returns list of (persona, trait, pos_path, neg_path).
    """
    from persona_steering.config import Trait
    trait_values = {t.value for t in Trait}
    files: dict[tuple[str, str], dict[str, Path]] = {}

    for pt_file in sorted(activations_dir.glob("*.pt")):
        stem = pt_file.stem  # e.g. "con_artist_assertiveness_pos"

        # Parse direction suffix
        if stem.endswith("_pos"):
            direction = "pos"
            rest = stem[:-4]
        elif stem.endswith("_neg"):
            direction = "neg"
            rest = stem[:-4]
        else:
            continue

        # Match against known trait names (handles multi-word persona slugs)
        persona, trait = None, None
        for tv in trait_values:
            if rest.endswith(f"_{tv}"):
                persona = rest[: -(len(tv) + 1)]
                trait = tv
                break

        if persona is None or trait is None:
            continue

        key = (persona, trait)
        files.setdefault(key, {})[direction] = pt_file

    pairs = []
    for (persona, trait), directions in sorted(files.items()):
        if "pos" in directions and "neg" in directions:
            pairs.append((persona, trait, directions["pos"], directions["neg"]))
        else:
            missing = "neg" if "pos" in directions else "pos"
            log.warning("Missing %s file for %s/%s, skipping", missing, persona, trait)

    return pairs


def compute_contrastive_vector(
    pos_path: Path, neg_path: Path
) -> tuple[torch.Tensor, int, int]:
    """Compute mean(pos) - mean(neg) from activation files.

    Each .pt file is a dict mapping key -> tensor of shape (n_layers, hidden_dim).

    Returns:
        (vector, n_positive, n_negative) where vector has shape (n_layers, hidden_dim).
    """
    pos_data = torch.load(pos_path, map_location="cpu", weights_only=True)
    neg_data = torch.load(neg_path, map_location="cpu", weights_only=True)

    # Compute mean directly in float32 without stacking all tensors in memory
    # Drop the last layer (float16 inf at final layer)
    n_pos = len(pos_data)
    n_neg = len(neg_data)

    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Empty activation files: pos={n_pos}, neg={n_neg}")

    # Running sum to avoid large intermediate tensors
    # Replace nan/inf with 0 before accumulating (float16 overflow in activations)
    _clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    pos_iter = iter(pos_data.values())
    pos_sum = _clean(next(pos_iter)[:-1].float())
    for v in pos_iter:
        pos_sum += _clean(v[:-1].float())

    neg_iter = iter(neg_data.values())
    neg_sum = _clean(next(neg_iter)[:-1].float())
    for v in neg_iter:
        neg_sum += _clean(v[:-1].float())

    vector = (pos_sum / n_pos) - (neg_sum / n_neg)  # (n_layers-1, hidden_dim)

    return vector, n_pos, n_neg


def main() -> None:
    args = parse_args()

    activations_dir = Path(args.activations_dir)
    # Derive model short name from path (e.g. outputs/gemma-2-9b-it/activations -> gemma-2-9b-it)
    short = activations_dir.parent.name
    activations_dir = ensure_dir(f"{short}-activations", activations_dir, "*.pt")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = activations_dir.parent / "vectors"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(activations_dir)
    if not pairs:
        log.error("No matching pos/neg activation file pairs found in %s", activations_dir)
        return

    log.info("Found %d persona x trait pairs", len(pairs))

    # W&B tracking
    method = infer_method(activations_dir)
    init_run("step3_vectors", short, config=vars(args), method=method)

    for i, (persona, trait, pos_path, neg_path) in enumerate(pairs):
        log.info("Computing vector for %s/%s...", persona, trait)

        vector, n_pos, n_neg = compute_contrastive_vector(pos_path, neg_path)

        output_path = output_dir / f"{persona}_{trait}.pt"
        torch.save(
            {
                "vector": vector,  # (n_layers, hidden_dim)
                "persona": persona,
                "trait": trait,
                "n_positive": n_pos,
                "n_negative": n_neg,
            },
            output_path,
        )

        norms = vector.norm(dim=1)
        log.info(
            "  Saved %s: shape %s, norm range [%.4f, %.4f], n_pos=%d, n_neg=%d",
            output_path.name, list(vector.shape),
            norms.min().item(), norms.max().item(), n_pos, n_neg,
        )
        log_metrics({
            "vectors/done": i + 1,
            "vectors/total": len(pairs),
            f"vectors/{persona}_{trait}/norm_max": norms.max().item(),
        })

    log.info("Done. Saved %d vectors to %s", len(pairs), output_dir)

    log_artifact(f"{short}-vectors", "vectors", output_dir, glob_pattern="*.pt")
    finish_run()


if __name__ == "__main__":
    main()
