#!/usr/bin/env python3
"""X4c: Pairwise cosine heatmaps of CAA trait vectors across all 15 personas
plus null and nonsense (17 contexts) for each of 8 traits.

Writes:
  {trait}_cosine_17.pdf/png           — per-trait 17x17 heatmap
  all_traits_cosine_17.pdf/png        — 2x4 grid with all traits
  cosine_stats_17.json                — per-trait mean/min/max cosine
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

TRAITS = ["empathy", "warmth", "honesty", "confidence",
          "assertiveness", "impulsivity", "risk_taking", "deference"]

# 10 mainstream + 5 weird + null + nonsense; grouped for readability.
PERSONA_ORDER = [
    # mainstream
    "farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
    "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist",
    # weird
    "pathological_liar", "six_year_old", "sociopath",
    "contrarian_deceiver", "actor_in_rehearsal",
    # baselines
    "null", "nonsense",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, default=22)
    return p.parse_args()


def load_vec(d: Path, persona: str, trait: str, layer: int) -> np.ndarray | None:
    path = d / f"{persona}_{trait}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    full = data["vector"].float().numpy()
    if layer >= full.shape[0]:
        return None
    return full[layer]


def cosine_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    X = np.stack(vectors, 0)
    n = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    return n @ n.T


def plot_single(mat: np.ndarray, labels: list[str], trait: str,
                out: Path, highlight=("null", "nonsense")) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Highlight null/nonsense rows & columns
    for name in highlight:
        if name in labels:
            i = labels.index(name)
            for j in range(len(labels)):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="gold", linewidth=1.2))
                ax.add_patch(plt.Rectangle(
                    (i - 0.5, j - 0.5), 1, 1,
                    fill=False, edgecolor="gold", linewidth=1.2))

    # Annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.5,
                    color="black" if abs(v) < 0.55 else "white")

    fig.colorbar(im, ax=ax, fraction=0.04)
    mean_off = (mat.sum() - np.trace(mat)) / (mat.size - mat.shape[0])
    ax.set_title(f"Pairwise cosine — {trait}   "
                 f"(mean off-diag = {mean_off:.3f})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    vec_dir = Path(args.vectors_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_mats = {}
    all_labels = {}
    stats = {}

    for trait in TRAITS:
        vecs = []
        labels = []
        for p in PERSONA_ORDER:
            v = load_vec(vec_dir, p, trait, args.layer)
            if v is None:
                print(f"[warn] missing {p}_{trait}")
                continue
            vecs.append(v)
            labels.append(p)
        mat = cosine_matrix(vecs)
        all_mats[trait] = mat
        all_labels[trait] = labels

        off = (mat.sum() - np.trace(mat)) / (mat.size - mat.shape[0])
        stats[trait] = {
            "n": len(labels),
            "mean_off_diag": float(off),
            "min": float(mat[np.triu_indices_from(mat, k=1)].min()),
            "max": float(mat[np.triu_indices_from(mat, k=1)].max()),
        }
        plot_single(mat, labels, trait,
                    out_dir / f"{trait}_cosine_17")
        print(f"Wrote {trait}_cosine_17.pdf (mean off-diag = {off:.3f})")

    # Grid of all traits
    fig, axes = plt.subplots(2, 4, figsize=(26, 13))
    for ax, trait in zip(axes.ravel(), TRAITS):
        mat = all_mats[trait]
        labels = all_labels[trait]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        for name in ("null", "nonsense"):
            if name in labels:
                i = labels.index(name)
                for j in range(len(labels)):
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="gold", linewidth=0.7))
                    ax.add_patch(plt.Rectangle(
                        (i - 0.5, j - 0.5), 1, 1,
                        fill=False, edgecolor="gold", linewidth=0.7))
        off = stats[trait]["mean_off_diag"]
        ax.set_title(f"{trait} (mean={off:.2f})", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    fig.suptitle("Pairwise cosine of CAA trait vectors — 15 personas + "
                 "null + nonsense (gold highlight)",
                 fontsize=13)
    fig.savefig(out_dir / "all_traits_cosine_17.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "all_traits_cosine_17.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Wrote all_traits_cosine_17.pdf")

    with open(out_dir / "cosine_stats_17.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Wrote cosine_stats_17.json")


if __name__ == "__main__":
    main()
