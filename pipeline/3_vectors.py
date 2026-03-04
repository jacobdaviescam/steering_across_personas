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
    pattern = re.compile(r"^(.+)_(.+)_(pos|neg)\.pt$")
    files: dict[tuple[str, str], dict[str, Path]] = {}

    for pt_file in sorted(activations_dir.glob("*.pt")):
        m = pattern.match(pt_file.name)
        if not m:
            continue
        persona, trait, direction = m.groups()
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

    # Stack all activation tensors (cast to float32 first to avoid float16 overflow)
    pos_tensors = [v.float() for v in pos_data.values()]
    neg_tensors = [v.float() for v in neg_data.values()]

    if not pos_tensors or not neg_tensors:
        raise ValueError(f"Empty activation files: pos={len(pos_tensors)}, neg={len(neg_tensors)}")

    pos_stack = torch.stack(pos_tensors)  # (n_pos, n_layers, hidden_dim)
    neg_stack = torch.stack(neg_tensors)  # (n_neg, n_layers, hidden_dim)

    pos_mean = pos_stack.mean(dim=0)  # (n_layers, hidden_dim)
    neg_mean = neg_stack.mean(dim=0)  # (n_layers, hidden_dim)

    vector = pos_mean - neg_mean  # (n_layers, hidden_dim)

    return vector, len(pos_tensors), len(neg_tensors)


def main() -> None:
    args = parse_args()

    activations_dir = Path(args.activations_dir)
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

    for persona, trait, pos_path, neg_path in pairs:
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

    log.info("Done. Saved %d vectors to %s", len(pairs), output_dir)


if __name__ == "__main__":
    main()
