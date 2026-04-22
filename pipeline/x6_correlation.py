#!/usr/bin/env python3
"""X6: correlate representational deviation with probe transfer gap.

Consumes iv_transfer_full.json (from x2) and/or x5's cross-transfer matrices.

Per (trait, context):
  x = 1 - cos(caa_vector[ctx], caa_vector[null])     # how far ctx vector is from null
  y = within[ctx] - A[ctx]                             # how much within-probe beats null-probe

Hypothesis: positive correlation — contexts with more deviant representations
benefit more from a context-specific probe vs. the null probe.

Usage:
    python pipeline/x6_correlation.py \
        --transfer-json outputs/gemma-2-27b-it/v2/caa_probes/iv_transfer_full.json \
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

from persona_steering.utils import derive_model_short_from_path
from persona_steering.wandb_utils import (
    finish_run, init_run, log_images, log_metrics, log_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--transfer-json", type=str, required=True)
    p.add_argument("--vectors-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--reference", type=str, default="null",
                   help="Context to measure deviation from (default: null)")
    return p.parse_args()


def load_vector(vectors_dir: Path, persona: str, trait: str, layer: int) -> np.ndarray:
    path = vectors_dir / f"{persona}_{trait}.pt"
    obj = torch.load(path, map_location="cpu", weights_only=True)
    vec = obj["vector"] if isinstance(obj, dict) and "vector" in obj else obj
    vec = vec[layer] if vec.ndim == 2 else vec
    return vec.float().numpy()


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    args = parse_args()
    transfer = json.loads(Path(args.transfer_json).read_text())
    vectors_dir = Path(args.vectors_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_short = derive_model_short_from_path(vectors_dir)
    init_run("x6_correlation", model_short, config=vars(args), method="caa")

    rows = []
    for trait, regimes in transfer.items():
        if "A" not in regimes or "within" not in regimes:
            continue
        try:
            ref_vec = load_vector(vectors_dir, args.reference, trait, args.layer)
        except FileNotFoundError:
            print(f"skip {trait}: no {args.reference} vector")
            continue

        for ctx in regimes["A"]:
            if ctx == args.reference:
                continue
            try:
                ctx_vec = load_vector(vectors_dir, ctx, trait, args.layer)
            except FileNotFoundError:
                continue
            x = 1.0 - cos(ctx_vec, ref_vec)
            y = regimes["within"][ctx] - regimes["A"][ctx]
            rows.append({"trait": trait, "context": ctx,
                         "vector_dist": x, "probe_gap": y,
                         "within": regimes["within"][ctx], "A": regimes["A"][ctx]})

    (out / "points.json").write_text(json.dumps(rows, indent=2))

    xs = np.array([r["vector_dist"] for r in rows])
    ys = np.array([r["probe_gap"] for r in rows])
    overall_pearson = pearsonr(xs, ys)
    overall_spearman = spearmanr(xs, ys)

    summary = {
        "n": len(rows),
        "pearson_r": float(overall_pearson.statistic),
        "pearson_p": float(overall_pearson.pvalue),
        "spearman_r": float(overall_spearman.statistic),
        "spearman_p": float(overall_spearman.pvalue),
        "per_trait": {},
    }

    traits = sorted({r["trait"] for r in rows})
    for trait in traits:
        tx = np.array([r["vector_dist"] for r in rows if r["trait"] == trait])
        ty = np.array([r["probe_gap"] for r in rows if r["trait"] == trait])
        if len(tx) < 3:
            continue
        pr = pearsonr(tx, ty)
        sr = spearmanr(tx, ty)
        summary["per_trait"][trait] = {
            "n": len(tx),
            "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue),
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    log_summary({
        "overall/pearson_r": summary["pearson_r"],
        "overall/pearson_p": summary["pearson_p"],
        "overall/spearman_r": summary["spearman_r"],
        "overall/spearman_p": summary["spearman_p"],
        "overall/n": summary["n"],
    })
    for trait, stats in summary["per_trait"].items():
        log_metrics({
            f"trait/{trait}/pearson_r": stats["pearson_r"],
            f"trait/{trait}/spearman_r": stats["spearman_r"],
        })

    # --- figure 1: aggregate scatter ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(traits)))
    for trait, c in zip(traits, colors):
        mask = [r["trait"] == trait for r in rows]
        ax.scatter(xs[mask], ys[mask], color=c, label=trait, alpha=0.7, s=40)
    # least-squares line
    if len(xs) >= 2:
        coef = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xline, coef[0] * xline + coef[1], "k--", alpha=0.5,
                label=f"r={overall_pearson.statistic:.2f} (p={overall_pearson.pvalue:.3f})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(f"1 - cos(context, {args.reference}) vector distance")
    ax.set_ylabel("within[ctx] - A[ctx]  (probe-gap, positive = within beats null)")
    ax.set_title("Vector deviation vs. probe transfer gap")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out / "scatter_aggregate.png", dpi=150)
    plt.close(fig)

    # --- figure 2: per-trait grid ---
    n = len(traits)
    cols = 4
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n), squeeze=False)
    for i, trait in enumerate(traits):
        ax = axes[i // cols][i % cols]
        tx = np.array([r["vector_dist"] for r in rows if r["trait"] == trait])
        ty = np.array([r["probe_gap"] for r in rows if r["trait"] == trait])
        ax.scatter(tx, ty, alpha=0.7)
        if len(tx) >= 2:
            coef = np.polyfit(tx, ty, 1)
            xline = np.linspace(tx.min(), tx.max(), 50)
            ax.plot(xline, coef[0] * xline + coef[1], "k--", alpha=0.5)
        ax.axhline(0, color="gray", lw=0.5)
        s = summary["per_trait"].get(trait, {})
        ax.set_title(f"{trait}\nr={s.get('pearson_r', float('nan')):.2f} "
                     f"(p={s.get('pearson_p', float('nan')):.3f})", fontsize=9)
        ax.set_xlabel("vec dist", fontsize=8)
        ax.set_ylabel("probe gap", fontsize=8)
    # hide unused
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out / "scatter_per_trait.png", dpi=150)
    plt.close(fig)

    log_images(out, prefix="x6_correlation")
    finish_run()

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
