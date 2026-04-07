#!/usr/bin/env python3
"""Cross-stage trajectory analysis for OLMo training checkpoints.

Loads vectors from all stages, computes:
  1. Transfer matrix at each stage + distance trajectory
  2. Per-(persona, trait) vector alignment across stages
  3. Subspace overlap (PCA principal angles) across stages
  4. Cluster stability across stages (adjusted Rand index)
  5. Shared vs specific variance ratio trajectory

Usage:
    python pipeline/t4_trajectory_analysis.py --layer 15
    python pipeline/t4_trajectory_analysis.py --layer 15 --stages base sft dpo instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from persona_steering.config import (
    OLMO_2_7B,
    OLMO_TRAINING_STAGES,
    OUTPUTS_DIR,
    CheckpointSpec,
    TARGET_LAYER,
    Trait,
)
from persona_steering.analysis import (
    build_transfer_matrix,
    cluster_persona_vectors,
    decompose_shared_specific,
    transfer_matrix_distance,
    vector_alignment_across_stages,
    subspace_overlap,
    cluster_stability,
)
from persona_steering.utils import log, model_short_name, save_json
from persona_steering.wandb_utils import init_run, finish_run, log_summary, log_artifact


class _VectorShim:
    """Minimal stand-in for SteeringVector used by analysis functions."""

    def __init__(self, vector: torch.Tensor, persona: str, trait: Trait, layer: int):
        self.vector = vector
        self.persona = persona
        self.trait = trait
        self.layer = layer

    @property
    def magnitude(self) -> float:
        return self.vector.norm().item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-stage trajectory analysis")
    parser.add_argument(
        "--stages", nargs="+", default=None,
        help="Stage labels (default: all with vectors)",
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Target layer (default: {TARGET_LAYER})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output dir (default: outputs/{model}/trajectory)",
    )
    return parser.parse_args()


def load_stage_vectors(
    spec: CheckpointSpec, layer: int,
) -> tuple[
    dict[str, dict[Trait, dict[int, _VectorShim]]],
    list[str],
    list[Trait],
]:
    """Load vectors for a single stage, returning the nested dict structure."""
    base_short = model_short_name(spec.model.hf_id)
    vectors_dir = OUTPUTS_DIR / base_short / spec.stage_label / "vectors"

    trait_values = {t.value for t in Trait}
    vectors: dict[str, dict[Trait, dict[int, _VectorShim]]] = {}
    persona_set: set[str] = set()
    trait_set: set[Trait] = set()

    if not vectors_dir.exists():
        return vectors, [], []

    for pt_file in sorted(vectors_dir.glob("*.pt")):
        stem = pt_file.stem

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
        full_vector = data["vector"]

        if layer >= full_vector.shape[0]:
            continue

        layer_vector = full_vector[layer].float()
        shim = _VectorShim(vector=layer_vector, persona=persona_slug, trait=trait, layer=layer)

        vectors.setdefault(persona_slug, {}).setdefault(trait, {})[layer] = shim
        persona_set.add(persona_slug)
        trait_set.add(trait)

    return vectors, sorted(persona_set), sorted(trait_set, key=lambda t: t.value)


def main() -> None:
    args = parse_args()
    layer = args.layer

    base_short = model_short_name(OLMO_2_7B.hf_id)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / base_short / "trajectory"
    output_dir.mkdir(parents=True, exist_ok=True)

    init_run("t3_trajectory_analysis", base_short, config=vars(args), method="caa")

    # Select stages
    if args.stages:
        stage_set = set(args.stages)
        stages = [s for s in OLMO_TRAINING_STAGES if s.stage_label in stage_set]
    else:
        stages = OLMO_TRAINING_STAGES

    # Load vectors for all stages
    stage_data: dict[str, tuple] = {}
    for spec in stages:
        vectors, personas, traits = load_stage_vectors(spec, layer)
        if personas:
            stage_data[spec.stage_label] = (vectors, personas, traits)
            log.info("Loaded [%s]: %d personas, %d traits", spec.stage_label, len(personas), len(traits))
        else:
            log.warning("No vectors for stage [%s], skipping", spec.stage_label)

    if len(stage_data) < 2:
        log.error("Need at least 2 stages with vectors, got %d", len(stage_data))
        return

    stage_labels = [s.stage_label for s in stages if s.stage_label in stage_data]

    # Use the intersection of personas and traits across all stages
    all_personas = sorted(set.intersection(*(set(d[1]) for d in stage_data.values())))
    all_traits = sorted(
        set.intersection(*(set(d[2]) for d in stage_data.values())),
        key=lambda t: t.value,
    )
    log.info("Common across stages: %d personas, %d traits", len(all_personas), len(all_traits))

    # -----------------------------------------------------------------------
    # 1. Transfer matrices at each stage + distance trajectory
    # -----------------------------------------------------------------------
    log.info("=== 1. Transfer matrix trajectory ===")
    transfer_matrices: dict[str, np.ndarray] = {}
    for sl in stage_labels:
        vectors, _, _ = stage_data[sl]
        tm = build_transfer_matrix(vectors, all_personas, all_traits, layer)
        transfer_matrices[sl] = tm
        np.save(output_dir / f"transfer_{sl}.npy", tm)

    # Pairwise distances between stages
    tm_distances: dict[str, dict[str, dict]] = {}
    for sa in stage_labels:
        tm_distances[sa] = {}
        for sb in stage_labels:
            tm_distances[sa][sb] = transfer_matrix_distance(
                transfer_matrices[sa], transfer_matrices[sb]
            )

    save_json(tm_distances, output_dir / "transfer_matrix_distances.json")
    save_json({"personas": all_personas, "traits": [t.value for t in all_traits],
               "layer": layer, "stages": stage_labels},
              output_dir / "trajectory_meta.json")

    # Log key comparisons vs final (instruct or last stage)
    ref_stage = stage_labels[-1]
    log.info("Transfer matrix distances vs [%s]:", ref_stage)
    for sl in stage_labels[:-1]:
        d = tm_distances[sl][ref_stage]
        log.info("  [%s]: frobenius=%.4f, spearman_rho=%.4f",
                 sl, d.get("frobenius", -1), d.get("spearman_rho", -1))

    # -----------------------------------------------------------------------
    # 2. Per-(persona, trait) vector alignment across stages
    # -----------------------------------------------------------------------
    log.info("=== 2. Vector alignment across stages ===")
    alignment_results: dict[str, dict[str, dict]] = {}

    for persona in all_personas:
        alignment_results[persona] = {}
        for trait in all_traits:
            vecs_by_stage: dict[str, torch.Tensor] = {}
            for sl in stage_labels:
                vectors, _, _ = stage_data[sl]
                shim = vectors.get(persona, {}).get(trait, {}).get(layer)
                if shim is not None:
                    vecs_by_stage[sl] = shim.vector

            if len(vecs_by_stage) >= 2:
                alignment_results[persona][trait.value] = vector_alignment_across_stages(vecs_by_stage)

    save_json(alignment_results, output_dir / "vector_alignment.json")

    # Summary: average alignment of each stage vs reference
    log.info("Mean vector alignment vs [%s] (cosine sim, averaged over all persona x trait):", ref_stage)
    for sl in stage_labels:
        cosines = []
        for persona in all_personas:
            for trait in all_traits:
                entry = alignment_results.get(persona, {}).get(trait.value, {})
                if sl in entry and ref_stage in entry.get(sl, {}):
                    cosines.append(entry[sl][ref_stage])
        if cosines:
            log.info("  [%s]: mean=%.4f, std=%.4f", sl, np.mean(cosines), np.std(cosines))

    # -----------------------------------------------------------------------
    # 3. Subspace overlap (PCA principal angles) across stages
    # -----------------------------------------------------------------------
    log.info("=== 3. Subspace overlap ===")
    subspace_results: dict[str, dict[str, dict[str, dict]]] = {}

    for trait in all_traits:
        subspace_results[trait.value] = {}

        stage_vecs: dict[str, list[torch.Tensor]] = {}
        for sl in stage_labels:
            vectors, _, _ = stage_data[sl]
            vecs = []
            for persona in all_personas:
                shim = vectors.get(persona, {}).get(trait, {}).get(layer)
                if shim is not None:
                    vecs.append(shim.vector)
            if vecs:
                stage_vecs[sl] = vecs

        for sa in stage_vecs:
            subspace_results[trait.value][sa] = {}
            for sb in stage_vecs:
                subspace_results[trait.value][sa][sb] = subspace_overlap(
                    stage_vecs[sa], stage_vecs[sb], n_components=5
                )

    save_json(subspace_results, output_dir / "subspace_overlap.json")

    log.info("Subspace overlap vs [%s] (mean cos^2 of principal angles):", ref_stage)
    for trait in all_traits:
        for sl in stage_labels[:-1]:
            entry = subspace_results.get(trait.value, {}).get(sl, {}).get(ref_stage, {})
            log.info("  %s [%s]: overlap=%.4f", trait.value, sl, entry.get("mean_overlap", -1))

    # -----------------------------------------------------------------------
    # 4. Cluster stability across stages
    # -----------------------------------------------------------------------
    log.info("=== 4. Cluster stability ===")
    labels_by_stage: dict[str, dict[str, int]] = {}

    for sl in stage_labels:
        tm = transfer_matrices[sl]
        clustering = cluster_persona_vectors(tm, all_personas)
        labels_by_stage[sl] = clustering["labels"]

    stability = cluster_stability(labels_by_stage)
    save_json(stability, output_dir / "cluster_stability.json")
    save_json(labels_by_stage, output_dir / "cluster_labels.json")

    log.info("Cluster stability (adjusted Rand index) vs [%s]:", ref_stage)
    for sl in stage_labels[:-1]:
        log.info("  [%s]: ARI=%.4f", sl, stability[sl][ref_stage])

    # -----------------------------------------------------------------------
    # 5. Shared vs specific variance trajectory
    # -----------------------------------------------------------------------
    log.info("=== 5. Shared/specific variance trajectory ===")
    variance_trajectory: dict[str, dict[str, float]] = {}

    for sl in stage_labels:
        variance_trajectory[sl] = {}
        vectors, _, _ = stage_data[sl]

        for trait in all_traits:
            trait_vectors = {}
            for persona in all_personas:
                shim = vectors.get(persona, {}).get(trait, {}).get(layer)
                if shim is not None:
                    trait_vectors[persona] = shim

            if len(trait_vectors) >= 2:
                decomp = decompose_shared_specific(trait_vectors)
                variance_trajectory[sl][trait.value] = decomp.variance_explained

    save_json(variance_trajectory, output_dir / "variance_trajectory.json")

    log.info("Shared variance explained trajectory:")
    for sl in stage_labels:
        vals = list(variance_trajectory.get(sl, {}).values())
        if vals:
            log.info("  [%s]: mean=%.4f (across %d traits)", sl, np.mean(vals), len(vals))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log_summary({
        "n_stages": len(stage_labels),
        "n_personas": len(all_personas),
        "n_traits": len(all_traits),
        "layer": layer,
    })
    log_artifact(f"{base_short}-trajectory", "trajectory_analysis", output_dir, glob_pattern="*.json")
    finish_run()

    log.info("=== Analysis complete ===")
    log.info("Results saved to %s", output_dir)
    for f in sorted(output_dir.glob("*")):
        log.info("  %s", f.name)


if __name__ == "__main__":
    main()
