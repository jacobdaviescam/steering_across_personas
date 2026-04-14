#!/usr/bin/env python3
"""C1 — Assistant ≈ centroid of persona vectors?

For each trait:

* Load the 10 persona trait vectors (original and C9-residualized if present).
* Load the Assistant-baseline trait vector (``assistant_default`` slug).
* Compute:

  * centroid similarity: ``cos(v_assistant, mean_p v_p)``
  * best single persona: ``max_p cos(v_assistant, v_p)``
  * random-persona baseline: mean pairwise cosine across personas
  * relative residual: ``||v_assistant - centroid|| / ||v_assistant||``

* Optionally fit non-negative weights summing to 1 that maximize
  ``cos(v_assistant, sum_p w_p v_p)`` (scipy SLSQP).

Usage:
    python pipeline/c1_assistant_centroid.py --model google/gemma-2-27b-it
    python pipeline/c1_assistant_centroid.py --model google/gemma-2-27b-it --use-residualized
    python pipeline/c1_assistant_centroid.py --skip-weights
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

from persona_steering.config import TARGET_LAYER
from persona_steering.council import (
    all_personas,
    all_traits,
    cosine,
    council_dir,
    load_assistant_vector,
    load_trait_vector,
    mean_pairwise_cosine,
    model_dir,
)
from persona_steering.utils import log, save_fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C1 Assistant-centroid test")
    p.add_argument("--model", type=str, default="google/gemma-2-27b-it")
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument(
        "--use-residualized",
        action="store_true",
        help="Use C9-residualized persona vectors (requires c9 to have run).",
    )
    p.add_argument("--skip-weights", action="store_true", help="Skip weighted-mean fit.")
    return p.parse_args()


def load_residualized(
    model: str, personas: list[str], traits: list[str]
) -> dict[tuple[str, str], torch.Tensor]:
    path = model_dir(model) / "council" / "c9" / "residualized_vectors.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"--use-residualized set but {path} missing; run c9 first."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for p in personas:
        for t in traits:
            key = f"{p}__{t}"
            if key in blob:
                out[(p, t)] = blob[key].float()
    return out


def fit_nonneg_weights(V: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Maximize cosine(target, V.T @ w) with w >= 0, sum(w) = 1.

    V has shape (n_personas, hidden). Returns w of shape (n_personas,).
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        log.warning("scipy not available; skipping weighted-mean fit")
        return np.full(V.shape[0], 1.0 / V.shape[0])

    t = target / (np.linalg.norm(target) + 1e-12)

    def neg_cos(w):
        v = V.T @ w
        n = np.linalg.norm(v)
        if n == 0:
            return 1.0
        return -(v @ t) / n

    n = V.shape[0]
    w0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    res = minimize(neg_cos, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x if res.success else w0


def main() -> None:
    args = parse_args()
    out_dir = council_dir(args.model, "c1")
    personas = all_personas()
    traits = all_traits()

    resid_vectors: dict[tuple[str, str], torch.Tensor] = {}
    if args.use_residualized:
        resid_vectors = load_residualized(args.model, personas, traits)

    summary: dict[str, dict] = {}
    weights: dict[str, dict[str, float]] = {}
    for trait in traits:
        persona_vecs = {}
        for p in personas:
            if args.use_residualized and (p, trait) in resid_vectors:
                persona_vecs[p] = resid_vectors[(p, trait)]
                continue
            try:
                persona_vecs[p] = load_trait_vector(args.model, p, trait, args.layer)
            except FileNotFoundError as err:
                log.warning("c1 missing %s/%s: %s", p, trait, err)

        try:
            v_assistant = load_assistant_vector(args.model, trait, args.layer)
        except FileNotFoundError as err:
            log.error("c1 assistant vector missing for %s: %s", trait, err)
            continue

        if len(persona_vecs) < 2:
            log.warning("c1 skipping %s — not enough personas", trait)
            continue

        stack = torch.stack(list(persona_vecs.values())).numpy()
        centroid = stack.mean(axis=0)
        v_a = v_assistant.numpy()

        cos_centroid = cosine(v_a, centroid)
        per_persona_cos = {p: cosine(v_a, v) for p, v in persona_vecs.items()}
        best_p = max(per_persona_cos, key=per_persona_cos.get)
        random_baseline = mean_pairwise_cosine(list(persona_vecs.values()))
        rel_resid = float(np.linalg.norm(v_a - centroid) / (np.linalg.norm(v_a) + 1e-12))

        summary[trait] = {
            "cos_centroid": cos_centroid,
            "cos_best_persona": per_persona_cos[best_p],
            "best_persona": best_p,
            "mean_pairwise_persona_cos": random_baseline,
            "relative_residual": rel_resid,
            "per_persona_cos": per_persona_cos,
        }

        if not args.skip_weights:
            w = fit_nonneg_weights(stack, v_a)
            weighted = stack.T @ w
            summary[trait]["cos_weighted"] = cosine(v_a, weighted)
            weights[trait] = {p: float(w_i) for p, w_i in zip(persona_vecs.keys(), w)}

        log.info(
            "c1 trait=%s cos_centroid=%.3f cos_best=%.3f (%s) random=%.3f resid=%.3f",
            trait, cos_centroid, per_persona_cos[best_p], best_p,
            random_baseline, rel_resid,
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {"layer": args.layer, "use_residualized": args.use_residualized, "traits": summary},
            f, indent=2,
        )
    if weights:
        with open(out_dir / "weighted_mean_weights.json", "w") as f:
            json.dump(weights, f, indent=2)

    # figure
    trait_labels = list(summary.keys())
    x = np.arange(len(trait_labels))
    width = 0.25
    centroid_vals = [summary[t]["cos_centroid"] for t in trait_labels]
    best_vals = [summary[t]["cos_best_persona"] for t in trait_labels]
    random_vals = [summary[t]["mean_pairwise_persona_cos"] for t in trait_labels]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width, centroid_vals, width, label="cos(assistant, centroid)")
    ax.bar(x, best_vals, width, label="cos(assistant, best persona)")
    ax.bar(x + width, random_vals, width, label="mean pairwise cos (personas)")
    ax.set_xticks(x)
    ax.set_xticklabels(trait_labels, rotation=30, ha="right")
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(-0.1, 1.05)
    title_suffix = " (residualized)" if args.use_residualized else ""
    ax.set_title(f"C1 — Assistant vs. persona centroid{title_suffix}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / "fig_centroid_comparison.png")


if __name__ == "__main__":
    main()
