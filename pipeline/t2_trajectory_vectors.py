#!/usr/bin/env python3
"""Compute contrastive vectors for all OLMo training-stage checkpoints.

Thin wrapper: runs the same vector computation as 3_vectors.py for each stage.

Usage:
    python pipeline/t3_trajectory_vectors.py
    python pipeline/t3_trajectory_vectors.py --stages base sft dpo instruct
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

from persona_steering.config import OLMO_TRAINING_STAGES, OUTPUTS_DIR
from persona_steering.utils import log, model_short_name
from persona_steering.wandb_utils import init_run, finish_run, log_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute contrastive vectors across OLMo training stages"
    )
    parser.add_argument(
        "--stages", nargs="+", default=None,
        help="Stage labels to run (default: all)",
    )
    return parser.parse_args()


def _load_vectors_module():
    """Import discover_pairs and compute_contrastive_vector from 3_vectors.py."""
    spec = importlib.util.spec_from_file_location(
        "vectors_mod",
        Path(__file__).resolve().parent / "3_vectors.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    args = parse_args()
    vectors_mod = _load_vectors_module()

    if args.stages:
        stage_set = set(args.stages)
        stages = [s for s in OLMO_TRAINING_STAGES if s.stage_label in stage_set]
    else:
        stages = OLMO_TRAINING_STAGES

    log.info("=== Training Trajectory Vectors ===")

    base_short = model_short_name(stages[0].model.hf_id) if stages else "olmo"
    init_run("t2_trajectory_vectors", base_short, config=vars(args), method="caa")

    for stage in stages:
        base_short = model_short_name(stage.model.hf_id)
        activations_dir = OUTPUTS_DIR / base_short / stage.stage_label / "caa_activations"
        vectors_dir = OUTPUTS_DIR / base_short / stage.stage_label / "vectors"

        if not activations_dir.exists():
            log.warning("[%s] No activations dir at %s, skipping", stage.stage_label, activations_dir)
            continue

        vectors_dir.mkdir(parents=True, exist_ok=True)

        pairs = vectors_mod.discover_pairs(activations_dir)
        if not pairs:
            log.warning("[%s] No pos/neg pairs found", stage.stage_label)
            continue

        log.info("[%s] Computing %d vectors...", stage.stage_label, len(pairs))

        for persona, trait, pos_path, neg_path in pairs:
            output_path = vectors_dir / f"{persona}_{trait}.pt"
            if output_path.exists():
                continue

            vector, n_pos, n_neg = vectors_mod.compute_contrastive_vector(pos_path, neg_path)
            torch.save(
                {
                    "vector": vector,
                    "persona": persona,
                    "trait": trait,
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                },
                output_path,
            )

        log.info("[%s] Vectors saved to %s", stage.stage_label, vectors_dir)
        log_metrics({"trajectory_vectors/stages_done": stages.index(stage) + 1})

    finish_run()
    log.info("=== All stages complete ===")


if __name__ == "__main__":
    main()
