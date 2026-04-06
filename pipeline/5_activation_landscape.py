#!/usr/bin/env python3
"""Activation-space landscape analysis: do similar personas have similar raw
activations, and does this explain steering vector similarity?

Implements the bound from the triangle-inequality derivation:
    cos(v_p1, v_p2) > 1 - (δ+ + δ-)² / (2 ||v_p1|| ||v_p2||)
where δ+ = ||a⁺_p1 - a⁺_p2||, δ- = ||a⁻_p1 - a⁻_p2||.

Modes:
  --activations-dir : full analysis with raw pos/neg activation files
  --vectors-dir     : partial analysis using only precomputed steering vectors

Outputs (to --output-dir, default: outputs/{model}/analysis_landscape/):
  activation_distances.json   — pairwise δ+, δ- per trait
  bound_vs_actual.json        — predicted lower bound vs actual cosine sim
  landscape_coords.json       — 2D UMAP/PCA coords for pos/neg activation means
  landscape_coords.npy        — raw coordinate array
  summary.json                — aggregate statistics

Usage:
    # Full analysis (requires raw activation .pt files from pod)
    python pipeline/5_activation_landscape.py \\
        --activations-dir outputs/gemma-2-27b-it/activations \\
        --vectors-dir outputs/gemma-2-27b-it/vectors \\
        --layer 22

    # Vectors-only mode (partial — no δ+/δ- computation)
    python pipeline/5_activation_landscape.py \\
        --vectors-dir outputs/gemma-2-27b-it/vectors \\
        --layer 22
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from persona_steering.config import PERSONA_SLUGS, Trait, TARGET_LAYER, OUTPUTS_DIR
from persona_steering.utils import cosine_similarity, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation landscape analysis")
    parser.add_argument(
        "--activations-dir", type=str, default=None,
        help="Directory with raw {persona}_{trait}_{pos|neg}.pt files",
    )
    parser.add_argument(
        "--vectors-dir", type=str, default=None,
        help="Directory with precomputed {persona}_{trait}.pt vector files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: sibling 'analysis_landscape' dir)",
    )
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument(
        "--personas", nargs="+", default=PERSONA_SLUGS,
        help="Persona slugs to include",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def compute_activation_mean(act_path: Path, layer: int) -> torch.Tensor:
    """Load a raw activation file and compute the mean activation at a layer.

    Each .pt file is a dict: key -> tensor(n_layers, hidden_dim).
    Returns tensor(hidden_dim,) in float32.
    """
    data = torch.load(act_path, map_location="cpu", weights_only=True)
    clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    vals = list(data.values())
    n = len(vals)
    if n == 0:
        raise ValueError(f"Empty activation file: {act_path}")
    acc = clean(vals[0][layer].float())
    for v in vals[1:]:
        acc += clean(v[layer].float())
    return acc / n


def load_vector(vec_path: Path, layer: int) -> torch.Tensor:
    """Load a steering vector at a given layer. Returns tensor(hidden_dim,)."""
    data = torch.load(vec_path, map_location="cpu", weights_only=True)
    return data["vector"][layer].float()


# ---------------------------------------------------------------------------
# Core analysis: pairwise activation distances
# ---------------------------------------------------------------------------

def compute_pairwise_distances(
    activations_dir: Path,
    personas: list[str],
    traits: list[Trait],
    layer: int,
) -> dict:
    """Compute pairwise ||a⁺_p1 - a⁺_p2|| and ||a⁻_p1 - a⁻_p2|| for all
    persona pairs and traits.

    Returns:
        {trait: {(p1, p2): {"delta_pos": float, "delta_neg": float}}}
    Also returns the mean activation vectors for landscape embedding.
    """
    # First, load all mean activations
    means: dict[str, dict[str, dict[str, torch.Tensor]]] = {}  # persona -> trait -> {pos, neg} -> tensor
    for persona in personas:
        means[persona] = {}
        for trait in traits:
            pos_path = activations_dir / f"{persona}_{trait.value}_pos.pt"
            neg_path = activations_dir / f"{persona}_{trait.value}_neg.pt"
            if not pos_path.exists() or not neg_path.exists():
                log.warning("Missing activation files for %s/%s, skipping", persona, trait.value)
                continue
            means[persona][trait.value] = {
                "pos": compute_activation_mean(pos_path, layer),
                "neg": compute_activation_mean(neg_path, layer),
            }
            log.info("Loaded activations: %s / %s", persona, trait.value)

    # Compute pairwise distances
    distances = {}
    for trait in traits:
        tv = trait.value
        distances[tv] = {}
        for p1, p2 in combinations(personas, 2):
            if tv not in means.get(p1, {}) or tv not in means.get(p2, {}):
                continue
            delta_pos = (means[p1][tv]["pos"] - means[p2][tv]["pos"]).norm().item()
            delta_neg = (means[p1][tv]["neg"] - means[p2][tv]["neg"]).norm().item()
            distances[tv][f"{p1}___{p2}"] = {
                "delta_pos": delta_pos,
                "delta_neg": delta_neg,
            }

    return distances, means


# ---------------------------------------------------------------------------
# Bound verification
# ---------------------------------------------------------------------------

def verify_bound(
    distances: dict,
    vectors_dir: Path,
    personas: list[str],
    traits: list[Trait],
    layer: int,
) -> list[dict]:
    """For each persona pair × trait, compare the theoretical lower bound on
    cos(v_p1, v_p2) against the actual cosine similarity.

    Returns list of records with bound, actual, and tightness.
    """
    records = []
    for trait in traits:
        tv = trait.value
        for p1, p2 in combinations(personas, 2):
            key = f"{p1}___{p2}"
            if key not in distances.get(tv, {}):
                continue

            v1_path = vectors_dir / f"{p1}_{tv}.pt"
            v2_path = vectors_dir / f"{p2}_{tv}.pt"
            if not v1_path.exists() or not v2_path.exists():
                continue

            v1 = load_vector(v1_path, layer)
            v2 = load_vector(v2_path, layer)

            d = distances[tv][key]
            delta_sum = d["delta_pos"] + d["delta_neg"]
            norm1 = v1.norm().item()
            norm2 = v2.norm().item()

            bound = 1.0 - (delta_sum ** 2) / (2 * norm1 * norm2 + 1e-10)
            actual = cosine_similarity(v1, v2)

            records.append({
                "persona_1": p1,
                "persona_2": p2,
                "trait": tv,
                "delta_pos": d["delta_pos"],
                "delta_neg": d["delta_neg"],
                "delta_sum": delta_sum,
                "v1_norm": norm1,
                "v2_norm": norm2,
                "bound": bound,
                "actual_cosine": actual,
                "gap": actual - bound,  # should be >= 0
                "bound_tight": actual - bound < 0.1,
            })

    return records


# ---------------------------------------------------------------------------
# Landscape embedding (UMAP/PCA on mean activations)
# ---------------------------------------------------------------------------

def build_landscape_embedding(
    means: dict[str, dict[str, dict[str, torch.Tensor]]],
    personas: list[str],
    traits: list[Trait],
    method: str = "pca",
) -> dict:
    """Embed all persona × trait × direction mean activations into 2D.

    Each point is a (persona, trait, direction) triple, labelled for
    visualisation. This places personas and traits on a shared manifold.

    Returns:
        {"coords": list of {persona, trait, direction, x, y},
         "raw_coords": np.ndarray}
    """
    vectors = []
    labels = []
    for persona in personas:
        for trait in traits:
            tv = trait.value
            if tv not in means.get(persona, {}):
                continue
            for direction in ("pos", "neg"):
                vectors.append(means[persona][tv][direction].numpy())
                labels.append({
                    "persona": persona,
                    "trait": tv,
                    "direction": direction,
                })

    if not vectors:
        return {"coords": [], "raw_coords": np.array([])}

    mat = np.stack(vectors)  # (N, hidden_dim)

    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, metric="cosine", random_state=42)
            coords_2d = reducer.fit_transform(mat)
        except ImportError:
            log.warning("umap-learn not installed, falling back to PCA")
            method = "pca"

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(mat)

    result_coords = []
    for i, label in enumerate(labels):
        result_coords.append({
            **label,
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
        })

    return {"coords": result_coords, "raw_coords": coords_2d}


# ---------------------------------------------------------------------------
# Vectors-only analysis
# ---------------------------------------------------------------------------

def vectors_only_analysis(
    vectors_dir: Path,
    personas: list[str],
    traits: list[Trait],
    layer: int,
) -> dict:
    """Compute steering vector cosine similarities and norms when raw
    activations are not available. Useful for previewing which persona pairs
    are most/least similar before pulling activations from the pod."""
    records = []
    for trait in traits:
        tv = trait.value
        for p1, p2 in combinations(personas, 2):
            v1_path = vectors_dir / f"{p1}_{tv}.pt"
            v2_path = vectors_dir / f"{p2}_{tv}.pt"
            if not v1_path.exists() or not v2_path.exists():
                continue
            v1 = load_vector(v1_path, layer)
            v2 = load_vector(v2_path, layer)
            records.append({
                "persona_1": p1,
                "persona_2": p2,
                "trait": tv,
                "cosine_sim": cosine_similarity(v1, v2),
                "v1_norm": v1.norm().item(),
                "v2_norm": v2.norm().item(),
                "v_diff_norm": (v1 - v2).norm().item(),
            })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    traits = list(Trait)
    personas = args.personas
    layer = args.layer

    has_activations = args.activations_dir is not None
    has_vectors = args.vectors_dir is not None

    if not has_activations and not has_vectors:
        log.error("Must provide at least one of --activations-dir or --vectors-dir")
        return

    # Determine output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif has_activations:
        output_dir = Path(args.activations_dir).parent / "analysis_landscape"
    else:
        output_dir = Path(args.vectors_dir).parent / "analysis_landscape"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"layer": layer, "personas": personas, "traits": [t.value for t in traits]}

    if has_activations:
        activations_dir = Path(args.activations_dir)
        vectors_dir = Path(args.vectors_dir) if has_vectors else None

        log.info("=== Full activation landscape analysis ===")
        log.info("Activations: %s", activations_dir)
        log.info("Layer: %d", layer)

        # 1. Pairwise activation distances
        distances, means = compute_pairwise_distances(
            activations_dir, personas, traits, layer,
        )
        with open(output_dir / "activation_distances.json", "w") as f:
            json.dump(distances, f, indent=2)
        log.info("Saved activation_distances.json")

        # 2. Bound verification (requires vectors)
        if vectors_dir:
            bound_records = verify_bound(distances, vectors_dir, personas, traits, layer)
            with open(output_dir / "bound_vs_actual.json", "w") as f:
                json.dump(bound_records, f, indent=2)

            n_tight = sum(1 for r in bound_records if r["bound_tight"])
            mean_gap = np.mean([r["gap"] for r in bound_records]) if bound_records else 0
            mean_bound = np.mean([r["bound"] for r in bound_records]) if bound_records else 0
            mean_actual = np.mean([r["actual_cosine"] for r in bound_records]) if bound_records else 0

            summary["bound_analysis"] = {
                "n_pairs": len(bound_records),
                "n_tight_bound": n_tight,
                "frac_tight": n_tight / len(bound_records) if bound_records else 0,
                "mean_gap": float(mean_gap),
                "mean_bound": float(mean_bound),
                "mean_actual_cosine": float(mean_actual),
            }
            log.info(
                "Bound analysis: %d pairs, %.1f%% tight, mean gap=%.4f",
                len(bound_records), 100 * n_tight / max(len(bound_records), 1), mean_gap,
            )

        # 3. Landscape embedding
        landscape = build_landscape_embedding(means, personas, traits, method="pca")
        coords_serializable = landscape["coords"]
        with open(output_dir / "landscape_coords.json", "w") as f:
            json.dump(coords_serializable, f, indent=2)
        if landscape["raw_coords"].size > 0:
            np.save(output_dir / "landscape_coords.npy", landscape["raw_coords"])
        log.info("Saved landscape_coords.json")

        # 4. Per-trait activation distance summary
        # For each trait: mean δ+ and δ- across persona pairs, and correlation
        # between δ+ and δ- (do similar personas cluster on both sides?)
        from scipy.stats import pearsonr
        trait_summaries = {}
        for trait in traits:
            tv = trait.value
            if tv not in distances or not distances[tv]:
                continue
            deltas_pos = [v["delta_pos"] for v in distances[tv].values()]
            deltas_neg = [v["delta_neg"] for v in distances[tv].values()]
            r, p = pearsonr(deltas_pos, deltas_neg) if len(deltas_pos) > 2 else (0, 1)
            trait_summaries[tv] = {
                "mean_delta_pos": float(np.mean(deltas_pos)),
                "mean_delta_neg": float(np.mean(deltas_neg)),
                "std_delta_pos": float(np.std(deltas_pos)),
                "std_delta_neg": float(np.std(deltas_neg)),
                "pos_neg_correlation": float(r),
                "pos_neg_corr_pvalue": float(p),
            }
        summary["trait_distances"] = trait_summaries

    else:
        log.info("=== Vectors-only analysis (no raw activations) ===")
        vectors_dir = Path(args.vectors_dir)

        records = vectors_only_analysis(vectors_dir, personas, traits, layer)
        with open(output_dir / "vector_similarities.json", "w") as f:
            json.dump(records, f, indent=2)

        if records:
            mean_cos = np.mean([r["cosine_sim"] for r in records])
            std_cos = np.std([r["cosine_sim"] for r in records])
            summary["vectors_only"] = {
                "n_pairs": len(records),
                "mean_cosine": float(mean_cos),
                "std_cosine": float(std_cos),
                "note": "Raw activations needed for full δ+/δ- analysis and landscape embedding",
            }
        log.info("Saved vector_similarities.json (%d pair records)", len(records))

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary.json to %s", output_dir)


if __name__ == "__main__":
    main()
