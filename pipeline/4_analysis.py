#!/usr/bin/env python3
"""Compare steering vectors across personas: transfer matrices, clustering, decomposition,
and alignment with the assistant axis.

Loads all vectors from pipeline step 3, picks a target layer, and runs the
analysis module to produce transfer matrices, clusters, decompositions, and
assistant axis alignment metrics.

Usage:
    python pipeline/4_analysis.py --vectors-dir outputs/gemma-2-9b-it/vectors --layer 22
    python pipeline/4_analysis.py --vectors-dir outputs/gemma-2-9b-it/vectors --axis outputs/gemma-2-9b-it/axis.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from persona_steering.config import Trait, TARGET_LAYER
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_summary, log_artifact, ensure_dir, infer_method
from persona_steering.analysis import (
    build_transfer_matrix,
    build_per_trait_transfer,
    cluster_persona_vectors,
    decompose_shared_specific,
    compare_steering_vs_interpersona,
)
from persona_steering.utils import log, save_json, load_json, cosine_similarity, VectorShim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse steering vectors across personas"
    )
    parser.add_argument(
        "--vectors-dir", type=str, required=True,
        help="Directory containing vector .pt files from step 3",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for analysis results (default: sibling 'analysis' dir)",
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Target layer for single-layer analyses (default: {TARGET_LAYER})",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=None,
        help="Number of clusters (default: auto via distance threshold)",
    )
    parser.add_argument(
        "--axis", type=str, default=None,
        help="Path to assistant axis .pt file (from assistant-axis pipeline). "
             "If provided, computes alignment between trait vectors and the axis.",
    )
    return parser.parse_args()


def load_vectors(
    vectors_dir: Path, layer: int
) -> tuple[
    dict[str, dict[Trait, dict[int, object]]],
    list[str],
    list[Trait],
]:
    """Load all vector .pt files and organise into nested dict for analysis.

    Returns:
        (vectors_nested, personas, traits) where vectors_nested has the shape
        expected by analysis.py: persona_slug -> Trait -> layer -> VectorShim.
    """
    trait_values = {t.value for t in Trait}

    vectors: dict[str, dict[Trait, dict[int, object]]] = {}
    persona_set: set[str] = set()
    trait_set: set[Trait] = set()

    for pt_file in sorted(vectors_dir.glob("*.pt")):
        stem = pt_file.stem  # e.g. "con_artist_assertiveness"

        # Match against known trait names (handles multi-word persona slugs)
        persona_slug = None
        trait_name = None
        for tv in trait_values:
            if stem.endswith(f"_{tv}"):
                persona_slug = stem[: -(len(tv) + 1)]
                trait_name = tv
                break

        if persona_slug is None or trait_name is None:
            continue

        trait = Trait(trait_name)
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        full_vector = data["vector"]  # (n_layers, hidden_dim)

        if layer >= full_vector.shape[0]:
            log.warning("Layer %d out of range for %s (max %d), skipping",
                        layer, pt_file.name, full_vector.shape[0] - 1)
            continue

        layer_vector = full_vector[layer]  # (hidden_dim,)

        shim = VectorShim(
            vector=layer_vector.float(),
            persona=persona_slug,
            trait=trait,
            layer=layer,
        )

        vectors.setdefault(persona_slug, {}).setdefault(trait, {})[layer] = shim
        persona_set.add(persona_slug)
        trait_set.add(trait)

    personas = sorted(persona_set)
    traits = sorted(trait_set, key=lambda t: t.value)

    return vectors, personas, traits


def main() -> None:
    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    short = vectors_dir.parent.name
    vectors_dir = ensure_dir(f"{short}-vectors", vectors_dir, "*.pt")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = vectors_dir.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    layer = args.layer

    # Load vectors
    vectors, personas, traits = load_vectors(vectors_dir, layer)
    log.info("Loaded vectors: %d personas, %d traits, layer %d", len(personas), len(traits), layer)

    if not personas or not traits:
        log.error("No valid vectors found in %s", vectors_dir)
        return

    # W&B tracking
    wb_config = {"layer": layer, "n_personas": len(personas), "n_traits": len(traits)}
    method = infer_method(vectors_dir)
    init_run("step4_analysis", short, config=wb_config, method=method)

    # 1. Build transfer matrix (average cosine sim across traits)
    log.info("Building transfer matrix...")
    transfer = build_transfer_matrix(vectors, personas, traits, layer)
    np.save(output_dir / "transfer_matrix.npy", transfer)
    save_json({"personas": personas, "traits": [t.value for t in traits], "layer": layer},
              output_dir / "transfer_meta.json")
    log.info("Transfer matrix saved. Mean off-diagonal sim: %.4f",
             (transfer.sum() - np.trace(transfer)) / (len(personas) * (len(personas) - 1)))

    # 2. Per-trait transfer matrices
    log.info("Building per-trait transfer matrices...")
    for trait in traits:
        per_trait = build_per_trait_transfer(vectors, personas, trait, layer)
        np.save(output_dir / f"transfer_{trait.value}.npy", per_trait)
        off_diag = (per_trait.sum() - np.trace(per_trait)) / max(len(personas) * (len(personas) - 1), 1)
        log.info("  %s: mean off-diagonal sim = %.4f", trait.value, off_diag)

    # 3. Clustering
    log.info("Clustering personas...")
    clustering = cluster_persona_vectors(transfer, personas, n_clusters=args.n_clusters)
    save_json(
        {
            "labels": clustering["labels"],
            "clusters": clustering["clusters"],
        },
        output_dir / "clusters.json",
    )
    log.info("Clusters: %s", clustering["clusters"])

    # 4. Shared vs specific decomposition (per trait)
    log.info("Decomposing shared vs persona-specific components...")
    decomp_results = {}
    for trait in traits:
        # Collect vectors for this trait across personas
        trait_vectors = {}
        for persona in personas:
            shim = vectors.get(persona, {}).get(trait, {}).get(layer)
            if shim is not None:
                trait_vectors[persona] = shim

        if len(trait_vectors) < 2:
            log.warning("Not enough vectors for %s decomposition (need >= 2)", trait.value)
            continue

        decomp = decompose_shared_specific(trait_vectors)
        decomp_results[trait.value] = {
            "variance_explained": decomp.variance_explained,
            "shared_magnitudes": decomp.shared_magnitudes,
            "specific_magnitudes": decomp.specific_magnitudes,
        }
        log.info("  %s: shared variance explained = %.4f", trait.value, decomp.variance_explained)

    save_json(decomp_results, output_dir / "decomposition.json")

    # 5. Assistant axis alignment (if axis provided)
    if args.axis:
        log.info("Loading assistant axis from %s...", args.axis)
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))
        from assistant_axis import load_axis

        axis_full = load_axis(args.axis)  # (n_layers, hidden_dim)
        if layer >= axis_full.shape[0]:
            log.error("Layer %d out of range for axis (max %d)", layer, axis_full.shape[0] - 1)
        else:
            axis_vec = axis_full[layer].float()  # (hidden_dim,)
            axis_unit = axis_vec / (axis_vec.norm() + 1e-8)

            # 5a. Per-vector alignment: how much does each persona×trait vector
            #     align with the assistant axis?
            log.info("Computing per-vector assistant axis alignment...")
            alignment_results: dict[str, dict[str, dict]] = {}
            for persona in personas:
                alignment_results[persona] = {}
                for trait in traits:
                    shim = vectors.get(persona, {}).get(trait, {}).get(layer)
                    if shim is None:
                        continue
                    metrics = compare_steering_vs_interpersona(shim, axis_vec)
                    alignment_results[persona][trait.value] = {
                        "cosine_with_axis": metrics["cosine_similarity"],
                        "projection_onto_axis": metrics["projection_onto_persona_axis"],
                        "orthogonal_magnitude": metrics["orthogonal_magnitude"],
                        "alignment_ratio": metrics["alignment_ratio"],
                        "vector_magnitude": metrics["steering_magnitude"],
                    }
            save_json(alignment_results, output_dir / "axis_alignment.json")

            # 5b. Per-trait summary: average alignment across personas
            log.info("Per-trait axis alignment (mean |cosine| across personas):")
            trait_alignment_summary = {}
            for trait in traits:
                cosines = []
                for persona in personas:
                    entry = alignment_results.get(persona, {}).get(trait.value)
                    if entry:
                        cosines.append(entry["cosine_with_axis"])
                if cosines:
                    mean_abs_cos = np.mean(np.abs(cosines))
                    mean_cos = np.mean(cosines)
                    trait_alignment_summary[trait.value] = {
                        "mean_cosine": float(mean_cos),
                        "mean_abs_cosine": float(mean_abs_cos),
                        "std_cosine": float(np.std(cosines)),
                    }
                    log.info("  %s: mean cos=%.4f, mean |cos|=%.4f, std=%.4f",
                             trait.value, mean_cos, mean_abs_cos, np.std(cosines))
            save_json(trait_alignment_summary, output_dir / "axis_alignment_summary.json")

            # 5c. Decompose persona-specific residuals against the axis.
            # For each trait: is the persona-specific component (from step 4)
            # aligned with the axis, or orthogonal to it?
            log.info("Checking if persona-specific residuals align with assistant axis...")
            residual_axis_results = {}
            for trait in traits:
                trait_vectors = {}
                for persona in personas:
                    shim = vectors.get(persona, {}).get(trait, {}).get(layer)
                    if shim is not None:
                        trait_vectors[persona] = shim
                if len(trait_vectors) < 2:
                    continue

                decomp = decompose_shared_specific(trait_vectors)
                residual_cosines = {}
                for persona in trait_vectors:
                    residual = decomp.specific_vectors[persona].float()
                    res_norm = residual.norm().item()
                    if res_norm < 1e-8:
                        residual_cosines[persona] = 0.0
                    else:
                        residual_cosines[persona] = cosine_similarity(residual, axis_vec)

                mean_abs = np.mean([abs(v) for v in residual_cosines.values()])
                residual_axis_results[trait.value] = {
                    "per_persona_cosine": residual_cosines,
                    "mean_abs_cosine": float(mean_abs),
                }
                log.info("  %s: residual-axis mean |cos|=%.4f  (low = persona differences are NOT about the axis)",
                         trait.value, mean_abs)

            save_json(residual_axis_results, output_dir / "residual_axis_alignment.json")

    # Summary
    log.info("Analysis complete. Results saved to %s", output_dir)
    log.info("Files:")
    for f in sorted(output_dir.glob("*")):
        log.info("  %s", f.name)

    # Log final W&B metrics
    decomp_path = output_dir / "decomposition.json"
    if decomp_path.exists():
        decomp = load_json(decomp_path)
        wb_metrics = {}
        for trait_name, data in decomp.items():
            wb_metrics[f"decomposition/{trait_name}/variance_explained"] = data["variance_explained"]
        log_metrics(wb_metrics)
    # Log transfer matrix stats
    tm_path = output_dir / "transfer_matrix.npy"
    if tm_path.exists():
        tm = np.load(tm_path)
        off_diag = (tm.sum() - np.trace(tm)) / (tm.size - len(personas))
        log_summary({"transfer/mean_off_diagonal": float(off_diag)})
    log_artifact(f"{short}-analysis", "analysis", output_dir)
    finish_run()


if __name__ == "__main__":
    main()
