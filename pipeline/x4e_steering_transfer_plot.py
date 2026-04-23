#!/usr/bin/env python3
"""X4e: Steering-transfer overview from the first steering experiment.

Reads steering_transfer_scores.json (Claude-judge scores on a 0–1 scale)
from one or two alpha runs and produces:

  steering_overview.pdf           — 4 panels: baseline vs self vs cross bar,
                                     self-vs-cross scatter, delta-over-
                                     baseline bar, cross/self ratio.
  steering_heatmaps_per_trait.pdf — 2x4 grid of 10x10 source→target
                                     matrices, one per trait.
  steering_stats.json             — per-trait summary numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIT_ORDER = ["empathy", "warmth", "honesty", "confidence",
               "assertiveness", "impulsivity", "risk_taking", "deference"]

PERSONA_ORDER = [
    "farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
    "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True,
                   help="Path to steering_transfer_scores.json")
    p.add_argument("--label", default="alpha=?",
                   help="Label for this run (e.g. 'alpha=2' or 'alpha=4')")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def build_matrices(scores: dict) -> dict[str, np.ndarray]:
    """{trait: 10x10 source→target score matrix} with NaN for missing."""
    mats = {}
    for trait, entries in scores.items():
        M = np.full((len(PERSONA_ORDER), len(PERSONA_ORDER)), np.nan)
        for key, entry in entries.items():
            if "->" not in key:
                continue
            src, tgt = key.split("->", 1)
            if src not in PERSONA_ORDER or tgt not in PERSONA_ORDER:
                continue
            i = PERSONA_ORDER.index(src)
            j = PERSONA_ORDER.index(tgt)
            M[i, j] = entry["mean_score"]
        mats[trait] = M
    return mats


def build_baselines(scores: dict) -> dict[str, np.ndarray]:
    """{trait: length-10 baseline score per target persona}."""
    base = {}
    for trait, entries in scores.items():
        b = np.full(len(PERSONA_ORDER), np.nan)
        for key, entry in entries.items():
            if not key.startswith("baseline_"):
                continue
            tgt = key.removeprefix("baseline_")
            if tgt not in PERSONA_ORDER:
                continue
            b[PERSONA_ORDER.index(tgt)] = entry["mean_score"]
        base[trait] = b
    return base


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scores = json.loads(Path(args.scores).read_text())
    mats = build_matrices(scores)
    baselines = build_baselines(scores)

    stats = {}
    for t in TRAIT_ORDER:
        M = mats[t]
        diag = np.diag(M)
        off = M.copy()
        np.fill_diagonal(off, np.nan)
        stats[t] = {
            "baseline_mean": float(np.nanmean(baselines[t])),
            "self_mean": float(np.nanmean(diag)),
            "cross_mean": float(np.nanmean(off)),
            "delta_self": float(np.nanmean(diag) - np.nanmean(baselines[t])),
            "delta_cross": float(np.nanmean(off) - np.nanmean(baselines[t])),
        }
    with open(out / "steering_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ---------- Figure 1: overview (4 panels)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    x = np.arange(len(TRAIT_ORDER))
    w = 0.27

    # Panel 1: baseline vs self vs cross
    ax = axes[0]
    b = [stats[t]["baseline_mean"] for t in TRAIT_ORDER]
    s = [stats[t]["self_mean"] for t in TRAIT_ORDER]
    c = [stats[t]["cross_mean"] for t in TRAIT_ORDER]
    ax.bar(x - w, b, w, label="baseline (no steer)", color="#aaaaaa")
    ax.bar(x,     s, w, label="self-steered",       color="#1f4e79")
    ax.bar(x + w, c, w, label="cross-steered",      color="#a23a3a")
    ax.set_xticks(x)
    ax.set_xticklabels(TRAIT_ORDER, rotation=30, ha="right")
    ax.set_ylabel("LLM-judge trait score (0–1)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Per-trait: baseline vs self vs cross  ({args.label})")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: self vs cross scatter per (source,target)
    ax = axes[1]
    all_self, all_cross = [], []
    for t in TRAIT_ORDER:
        M = mats[t]
        diag = np.diag(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if i == j:
                    continue
                if not (np.isnan(M[i, i]) or np.isnan(M[i, j])):
                    all_self.append(M[i, i])
                    all_cross.append(M[i, j])
    ax.scatter(all_self, all_cross, s=18, alpha=0.5, color="#4682b4")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("self-steer score  v_{src}→src")
    ax.set_ylabel("cross-steer score v_{src}→tgt (tgt≠src)")
    ax.set_title("Each source × target pair — diagonal = parity")
    ax.grid(alpha=0.3)

    # Panel 3: delta (self − baseline) vs delta (cross − baseline)
    ax = axes[2]
    ds = [stats[t]["delta_self"] for t in TRAIT_ORDER]
    dc = [stats[t]["delta_cross"] for t in TRAIT_ORDER]
    ax.bar(x - w / 2, ds, w, label="Δ self", color="#1f4e79")
    ax.bar(x + w / 2, dc, w, label="Δ cross", color="#a23a3a")
    ax.set_xticks(x)
    ax.set_xticklabels(TRAIT_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Score − baseline")
    ax.set_ylim(-0.05, 0.55)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Steering gain over baseline")
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: cross / self ratio
    ax = axes[3]
    ratio = [stats[t]["cross_mean"] / stats[t]["self_mean"]
             if stats[t]["self_mean"] > 0 else np.nan
             for t in TRAIT_ORDER]
    ax.bar(x, ratio, color="#a23a3a")
    ax.axhline(1.0, color="black", ls="--", lw=0.8, label="parity")
    ax.set_xticks(x)
    ax.set_xticklabels(TRAIT_ORDER, rotation=30, ha="right")
    ax.set_ylim(0.6, 1.2)
    ax.set_ylabel("cross_mean / self_mean")
    ax.set_title("Does cross-steering equal self-steering? (ratio)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Steering-transfer overview — source × target × trait  "
                 f"({args.label})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "steering_overview.pdf")
    fig.savefig(out / "steering_overview.png", dpi=150)
    plt.close(fig)
    print("Wrote steering_overview.pdf")

    # ---------- Figure 2: per-trait 10x10 transfer heatmaps
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for ax, t in zip(axes.ravel(), TRAIT_ORDER):
        M = mats[t]
        im = ax.imshow(M, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(len(PERSONA_ORDER)))
        ax.set_yticks(range(len(PERSONA_ORDER)))
        ax.set_xticklabels(PERSONA_ORDER, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(PERSONA_ORDER, fontsize=7)
        ax.set_xlabel("target persona (system prompt)")
        ax.set_ylabel("source persona (steering vector)")
        # outline diagonal
        for k in range(len(PERSONA_ORDER)):
            ax.add_patch(plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=False, edgecolor="black", linewidth=1.1))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="black" if 0.2 < v < 0.75 else "white")
        ax.set_title(f"{t}   baseline={stats[t]['baseline_mean']:.2f}  "
                     f"self={stats[t]['self_mean']:.2f}  "
                     f"cross={stats[t]['cross_mean']:.2f}",
                     fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    fig.suptitle(f"Steering transfer — each cell: LLM-judge trait score "
                 f"when source's v_t is applied to target    ({args.label})",
                 fontsize=13)
    fig.savefig(out / "steering_heatmaps_per_trait.pdf", bbox_inches="tight")
    fig.savefig(out / "steering_heatmaps_per_trait.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Wrote steering_heatmaps_per_trait.pdf")

    # ---------- print summary
    print(f"\nSummary ({args.label}):")
    print(f"{'trait':<14} {'base':>6} {'self':>6} {'cross':>7} "
          f"{'Δself':>7} {'Δcross':>7}")
    for t in TRAIT_ORDER:
        print(f"{t:<14} {stats[t]['baseline_mean']:>6.2f} "
              f"{stats[t]['self_mean']:>6.2f} {stats[t]['cross_mean']:>7.2f} "
              f"{stats[t]['delta_self']:>+7.2f} "
              f"{stats[t]['delta_cross']:>+7.2f}")


if __name__ == "__main__":
    main()
