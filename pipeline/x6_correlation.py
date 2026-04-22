#!/usr/bin/env python3
"""X6: pairwise cosine vs pairwise probe transfer (what Raya asked for).

For each trait, for every pair of contexts (i, j) with i != j:
  x = 1 - cos(vec[i, trait], vec[j, trait])
  y = AUROC of probe trained on context i, evaluated on context j's IV responses

Plot + correlate. Expect negative correlation: more different vectors -> worse transfer.

Usage:
    python pipeline/x6_correlation.py \
        --matrix-dir outputs/gemma-2-27b-it/v2/caa_probes \
        --vectors-dir outputs/gemma-2-27b-it/v2/caa_vectors \
        --output-dir outputs/gemma-2-27b-it/v2/x6_correlation \
        --layer 22
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
from scipy.stats import pearsonr, spearmanr

from persona_steering.config import Trait
from persona_steering.utils import derive_model_short_from_path
from persona_steering.wandb_utils import (
    finish_run, init_run, log_images, log_metrics, log_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", type=str, required=True,
                   help="dir containing iv_cross_transfer_{trait}.npy from x5")
    p.add_argument("--vectors-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--layer", type=int, default=22)
    return p.parse_args()


def load_vector(vectors_dir: Path, ctx: str, trait: str, layer: int) -> np.ndarray:
    obj = torch.load(vectors_dir / f"{ctx}_{trait}.pt", map_location="cpu",
                     weights_only=True)
    vec = obj["vector"] if isinstance(obj, dict) and "vector" in obj else obj
    vec = vec[layer] if vec.ndim == 2 else vec
    return vec.float().numpy()


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    args = parse_args()
    mat_dir = Path(args.matrix_dir)
    vec_dir = Path(args.vectors_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_short = derive_model_short_from_path(vec_dir)
    init_run("x6_correlation", model_short, config=vars(args), method="caa")

    traits = [t.value for t in Trait]
    all_points = []
    per_trait_stats = {}

    for trait in traits:
        mat_path = mat_dir / f"iv_cross_transfer_{trait}.npy"
        ctx_path = mat_dir / f"iv_cross_transfer_{trait}_contexts.json"
        if not mat_path.exists() or not ctx_path.exists():
            print(f"skip {trait}: matrix missing")
            continue
        mat = np.load(mat_path)
        contexts = json.loads(ctx_path.read_text())["contexts"]

        xs, ys = [], []
        for i, ci in enumerate(contexts):
            for j, cj in enumerate(contexts):
                if i == j or np.isnan(mat[i, j]):
                    continue
                try:
                    vi = load_vector(vec_dir, ci, trait, args.layer)
                    vj = load_vector(vec_dir, cj, trait, args.layer)
                except FileNotFoundError:
                    continue
                xs.append(1.0 - cos(vi, vj))
                ys.append(float(mat[i, j]))
                all_points.append({"trait": trait, "train": ci, "eval": cj,
                                   "vec_dist": xs[-1], "auroc": ys[-1]})

        if len(xs) < 3:
            continue
        xs = np.array(xs); ys = np.array(ys)
        pr = pearsonr(xs, ys)
        sr = spearmanr(xs, ys)
        per_trait_stats[trait] = {
            "n": len(xs),
            "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue),
        }
        log_metrics({
            f"trait/{trait}/pearson_r": per_trait_stats[trait]["pearson_r"],
            f"trait/{trait}/spearman_r": per_trait_stats[trait]["spearman_r"],
        })
        print(f"  {trait:15s} n={len(xs):3d}  r={pr.statistic:+.3f} (p={pr.pvalue:.3f})")

    # Aggregate
    all_x = np.array([p["vec_dist"] for p in all_points])
    all_y = np.array([p["auroc"] for p in all_points])
    overall_pr = pearsonr(all_x, all_y)
    overall_sr = spearmanr(all_x, all_y)
    summary = {
        "n": len(all_points),
        "pearson_r": float(overall_pr.statistic),
        "pearson_p": float(overall_pr.pvalue),
        "spearman_r": float(overall_sr.statistic),
        "spearman_p": float(overall_sr.pvalue),
        "per_trait": per_trait_stats,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "points.json").write_text(json.dumps(all_points, indent=2))

    log_summary({
        "overall/n": summary["n"],
        "overall/pearson_r": summary["pearson_r"],
        "overall/pearson_p": summary["pearson_p"],
        "overall/spearman_r": summary["spearman_r"],
        "overall/spearman_p": summary["spearman_p"],
    })

    # --- per-trait scatter grid ---
    trait_names = list(per_trait_stats.keys())
    n = len(trait_names)
    cols = 4
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n), squeeze=False)
    for k, trait in enumerate(trait_names):
        ax = axes[k // cols][k % cols]
        tx = np.array([p["vec_dist"] for p in all_points if p["trait"] == trait])
        ty = np.array([p["auroc"] for p in all_points if p["trait"] == trait])
        ax.scatter(tx, ty, alpha=0.5, s=15)
        if len(tx) >= 2:
            coef = np.polyfit(tx, ty, 1)
            xl = np.linspace(tx.min(), tx.max(), 50)
            ax.plot(xl, coef[0] * xl + coef[1], "k--", alpha=0.5)
        s = per_trait_stats[trait]
        ax.set_title(f"{trait}\nr={s['pearson_r']:+.2f} (p={s['pearson_p']:.3f})",
                     fontsize=9)
        ax.set_xlabel("1 - cos(vec_i, vec_j)", fontsize=8)
        ax.set_ylabel("AUROC (probe_i on ctx_j)", fontsize=8)
    for k in range(n, rows_n * cols):
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out / "scatter_per_trait.png", dpi=150)
    plt.close(fig)

    # --- aggregate scatter ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trait_names)))
    for trait, c in zip(trait_names, colors):
        mask = [p["trait"] == trait for p in all_points]
        ax.scatter(all_x[mask], all_y[mask], color=c, label=trait, alpha=0.5, s=15)
    coef = np.polyfit(all_x, all_y, 1)
    xl = np.linspace(all_x.min(), all_x.max(), 100)
    ax.plot(xl, coef[0] * xl + coef[1], "k--",
            label=f"r={overall_pr.statistic:+.2f} (p={overall_pr.pvalue:.3f})")
    ax.set_xlabel("1 - cos(vec_i, vec_j)  (pairwise vector distance)")
    ax.set_ylabel("AUROC (probe trained on i, evaluated on j)")
    ax.set_title("Vector dissimilarity vs probe transfer")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "scatter_aggregate.png", dpi=150)
    plt.close(fig)

    log_images(out, prefix="x6_correlation")
    finish_run()

    print("\n=== AGGREGATE ===")
    print(f"n={summary['n']}  r={summary['pearson_r']:+.3f} "
          f"(p={summary['pearson_p']:.3f})  "
          f"rho={summary['spearman_r']:+.3f} (p={summary['spearman_p']:.3f})")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
