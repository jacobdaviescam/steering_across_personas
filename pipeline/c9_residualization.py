#!/usr/bin/env python3
"""C9 — Persona-mean residualization (gate).

Does cross-persona trait-vector variation survive removal of persona-specific
activation baselines?

For each persona p, pool all layer-``L`` activations across every
(trait, direction) file and take the mean ``mu_p``. Subtract ``mu_p`` from each
activation, then recompute the contrastive trait vector
``v_resid_{p,t} = mean(R_{p,t,pos}) - mean(R_{p,t,neg})``.

Per trait we compare:

* mean pairwise cosine across the 10 original persona vectors (``cos_orig_t``)
* mean pairwise cosine across the 10 residualized persona vectors
  (``cos_resid_t``)
* cosine between the residualized centroid and the original centroid

Gate:

* **pass** — ``cos_resid_t < 0.90`` for >= 6 of 8 traits.
* **fail** — ``cos_resid_t > 0.95`` for most traits (reframe paper).
* **ambiguous** — intermediate, flag and proceed with caution.

Usage:
    python pipeline/c9_residualization.py --model google/gemma-2-27b-it
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
    load_persona_activations,
    load_trait_vector,
    mean_pairwise_cosine,
)
from persona_steering.utils import log, save_fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C9 persona-mean residualization gate")
    p.add_argument("--model", type=str, default="google/gemma-2-27b-it")
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    return p.parse_args()


def compute_persona_mean(model: str, persona: str, traits: list[str], layer: int) -> torch.Tensor:
    pools = []
    for trait in traits:
        for direction in ("pos", "neg"):
            try:
                pools.append(load_persona_activations(model, persona, trait, direction, layer))
            except FileNotFoundError as err:
                log.warning("c9 skipping %s/%s/%s: %s", persona, trait, direction, err)
    if not pools:
        raise RuntimeError(f"no activations for persona={persona}")
    return torch.cat(pools, dim=0).mean(dim=0)


def residualized_vector(
    model: str,
    persona: str,
    trait: str,
    mu: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    pos = load_persona_activations(model, persona, trait, "pos", layer)
    neg = load_persona_activations(model, persona, trait, "neg", layer)
    return (pos - mu).mean(dim=0) - (neg - mu).mean(dim=0)


def classify_verdict(cos_resid_by_trait: dict[str, float]) -> str:
    below_90 = sum(1 for v in cos_resid_by_trait.values() if v < 0.90)
    above_95 = sum(1 for v in cos_resid_by_trait.values() if v > 0.95)
    n = len(cos_resid_by_trait)
    if below_90 >= max(1, int(0.75 * n)):
        return "pass"
    if above_95 >= max(1, int(0.75 * n)):
        return "fail"
    return "ambiguous"


def main() -> None:
    args = parse_args()
    out_dir = council_dir(args.model, "c9")
    personas = all_personas()
    traits = all_traits()

    # 1) persona means (one per persona) from pooled activations
    log.info("c9: computing persona means for %d personas", len(personas))
    persona_means: dict[str, torch.Tensor] = {}
    for slug in personas:
        persona_means[slug] = compute_persona_mean(args.model, slug, traits, args.layer)

    # 2) residualized and original vectors per (persona, trait)
    residualized: dict[tuple[str, str], torch.Tensor] = {}
    originals: dict[tuple[str, str], torch.Tensor] = {}
    for slug in personas:
        for trait in traits:
            try:
                residualized[(slug, trait)] = residualized_vector(
                    args.model, slug, trait, persona_means[slug], args.layer
                )
            except FileNotFoundError as err:
                log.warning("c9 residual missing %s/%s: %s", slug, trait, err)
                continue
            try:
                originals[(slug, trait)] = load_trait_vector(
                    args.model, slug, trait, layer=args.layer
                )
            except FileNotFoundError as err:
                log.warning("c9 original vector missing %s/%s: %s", slug, trait, err)

    # 3) per-trait statistics
    summary: dict[str, dict] = {}
    for trait in traits:
        orig_vs = [originals[(p, trait)] for p in personas if (p, trait) in originals]
        resid_vs = [residualized[(p, trait)] for p in personas if (p, trait) in residualized]
        if len(orig_vs) < 2 or len(resid_vs) < 2:
            log.warning("c9 skipping trait=%s (insufficient vectors)", trait)
            continue
        cos_orig = mean_pairwise_cosine(orig_vs)
        cos_resid = mean_pairwise_cosine(resid_vs)
        centroid_orig = torch.stack(orig_vs).mean(dim=0)
        centroid_resid = torch.stack(resid_vs).mean(dim=0)
        summary[trait] = {
            "cos_orig": cos_orig,
            "cos_resid": cos_resid,
            "centroid_cos": cosine(centroid_orig, centroid_resid),
            "n_personas": len(resid_vs),
        }
        log.info(
            "c9 trait=%s cos_orig=%.3f cos_resid=%.3f centroid_cos=%.3f",
            trait, cos_orig, cos_resid, summary[trait]["centroid_cos"],
        )

    verdict = classify_verdict({t: s["cos_resid"] for t, s in summary.items()})
    log.info("c9 verdict: %s", verdict)

    # 4) persist artifacts
    torch.save(
        {f"{p}__{t}": v for (p, t), v in residualized.items()},
        out_dir / "residualized_vectors.pt",
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"verdict": verdict, "layer": args.layer, "traits": summary}, f, indent=2)

    # 5) figure
    fig, ax = plt.subplots(figsize=(10, 4.5))
    trait_labels = list(summary.keys())
    x = np.arange(len(trait_labels))
    width = 0.4
    orig_vals = [summary[t]["cos_orig"] for t in trait_labels]
    resid_vals = [summary[t]["cos_resid"] for t in trait_labels]
    ax.bar(x - width / 2, orig_vals, width, label="original")
    ax.bar(x + width / 2, resid_vals, width, label="residualized")
    ax.axhline(0.95, color="red", linestyle="--", label="collapse threshold 0.95")
    ax.axhline(0.90, color="orange", linestyle=":", label="pass threshold 0.90")
    ax.set_xticks(x)
    ax.set_xticklabels(trait_labels, rotation=30, ha="right")
    ax.set_ylabel("mean pairwise cosine across personas")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"C9 residualization gate — verdict: {verdict.upper()}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / "fig_residualization.png")


if __name__ == "__main__":
    main()
