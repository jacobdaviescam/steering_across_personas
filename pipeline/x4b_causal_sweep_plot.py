#!/usr/bin/env python3
"""X4b: Causal sweep summary plot.

Reads X3c's sweep_results.json and produces a multi-panel figure showing
how each of three metrics changes with the steering coefficient α:
  * P(context | output)           — behavioural drift (rises)
  * AUROC of null-trained probe   — degrades with α
  * AUROC of within-context probe — degrades with α

Figures:
  causal_sweep_summary.pdf         — mean per trait, 3 panels
  causal_sweep_small_multiples.pdf — per-trait 2x4 grid, lines per context
  causal_sweep_phase.pdf           — P(context) vs AUROC scatter
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PERSONA_COLORS = {
    "farmer": "#6b8e23",
    "politician": "#b22222",
    "therapist": "#4682b4",
    "drill_sergeant": "#2f4f4f",
    "street_hustler": "#daa520",
    "professor": "#556b2f",
    "tech_ceo": "#8a2be2",
    "kindergarten_teacher": "#ff69b4",
    "surgeon": "#8b0000",
    "con_artist": "#ff8c00",
}

TRAIT_ORDER = ["empathy", "warmth", "honesty", "confidence",
               "assertiveness", "impulsivity", "risk_taking", "deference"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-results", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.sweep_results).read_text())["results"]
    # Filter: keep only "main" condition (the orthogonalised u_C steering)
    data = [r for r in data if r.get("condition", "main") == "main"]

    by_curve = defaultdict(list)
    for r in data:
        by_curve[(r["trait"], r["context"])].append(r)
    for rows in by_curve.values():
        rows.sort(key=lambda r: r["alpha"])

    traits = [t for t in TRAIT_ORDER
              if any(k[0] == t for k in by_curve)]
    contexts = sorted({k[1] for k in by_curve},
                      key=lambda c: list(PERSONA_COLORS).index(c)
                      if c in PERSONA_COLORS else 999)
    alphas = sorted({r["alpha"] for r in data})
    print(f"Traits: {traits}")
    print(f"Contexts: {contexts}")
    print(f"Alphas: {alphas}")

    def get(rows, key):
        ys = []
        for r in rows:
            v = r.get(key)
            ys.append(np.nan if v is None else float(v))
        return np.array(ys)

    # ---------- Figure 1: summary (mean across contexts, lines per trait)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    metrics = [
        ("p_context", "P(context | output)", axes[0]),
        ("auroc_null", "AUROC — null-trained probe", axes[1]),
        ("auroc_within", "AUROC — within-context probe", axes[2]),
    ]

    trait_palette = plt.cm.tab10(np.linspace(0, 1, len(traits)))
    for metric, title, ax in metrics:
        for i, t in enumerate(traits):
            xs = alphas
            ys = []
            for a in alphas:
                pts = [r[metric] for r in data
                       if r["trait"] == t and r["alpha"] == a
                       and r.get(metric) is not None]
                ys.append(np.mean(pts) if pts else np.nan)
            ax.plot(xs, ys, "-o", color=trait_palette[i],
                    label=t, linewidth=2, markersize=5)
        ax.set_xlabel("α (steering coefficient)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if metric.startswith("auroc"):
            ax.axhline(0.5, color="grey", ls=":", lw=0.6)
            ax.set_ylim(0.4, 1.02)
        else:
            ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("mean across contexts")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                    fontsize=8, frameon=False)
    fig.suptitle(
        "Causal sweep: steering α·u_C^⊥ shifts behaviour toward C "
        "while both probes degrade",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "causal_sweep_summary.pdf")
    fig.savefig(out / "causal_sweep_summary.png", dpi=150)
    plt.close(fig)
    print("Wrote", out / "causal_sweep_summary.pdf")

    # ---------- Figure 2: per-trait small multiples
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    for ax, t in zip(axes.ravel(), traits):
        ax2 = ax.twinx()
        for c in contexts:
            rows = by_curve.get((t, c))
            if not rows:
                continue
            xs = [r["alpha"] for r in rows]
            p_ctx = get(rows, "p_context")
            a_null = get(rows, "auroc_null")
            a_within = get(rows, "auroc_within")
            col = PERSONA_COLORS.get(c, "gray")
            ax.plot(xs, p_ctx, "-o", color=col, alpha=0.8,
                    markersize=4, linewidth=1.5, label=c)
            ax2.plot(xs, a_null, "--s", color=col, alpha=0.5, markersize=3)
            ax2.plot(xs, a_within, ":^", color=col, alpha=0.5, markersize=3)
        ax.set_ylim(0, 1.02)
        ax2.set_ylim(0.4, 1.02)
        ax2.axhline(0.5, color="grey", ls=":", lw=0.6)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("α")
        ax.set_ylabel("P(context)", color="#222")
        ax2.set_ylabel("AUROC", color="#777")
        ax.grid(alpha=0.25)

    # single legend for the whole figure
    persona_handles = [plt.Line2D([0], [0], marker="o", linestyle="-",
                                  color=PERSONA_COLORS[c], label=c)
                       for c in contexts if c in PERSONA_COLORS]
    style_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-",  marker="o",
                   label="P(context)"),
        plt.Line2D([0], [0], color="black", linestyle="--", marker="s",
                   label="AUROC null"),
        plt.Line2D([0], [0], color="black", linestyle=":",  marker="^",
                   label="AUROC within"),
    ]
    fig.legend(handles=persona_handles + style_handles,
               loc="lower center", ncol=7, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Causal sweep per trait — solid=behavioural P(C), dashed=null probe, "
        "dotted=within probe",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out / "causal_sweep_small_multiples.pdf")
    fig.savefig(out / "causal_sweep_small_multiples.png", dpi=150)
    plt.close(fig)
    print("Wrote", out / "causal_sweep_small_multiples.pdf")

    # ---------- Figure 3: phase portrait — P(context) vs AUROC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for (t, c), rows in by_curve.items():
        xs = [r["p_context"] for r in rows]
        ys_null = get(rows, "auroc_null")
        ys_within = get(rows, "auroc_within")
        col = PERSONA_COLORS.get(c, "gray")
        ax1.plot(xs, ys_null, "-o", color=col, alpha=0.45, markersize=3)
        ax2.plot(xs, ys_within, "-o", color=col, alpha=0.45, markersize=3)
        # mark the α=8 endpoint
        if len(rows) >= 2:
            ax1.scatter(xs[-1], ys_null[-1], color=col, s=70,
                        edgecolor="black", linewidth=0.8, zorder=5)
            ax2.scatter(xs[-1], ys_within[-1], color=col, s=70,
                        edgecolor="black", linewidth=0.8, zorder=5)
    for ax, title in ((ax1, "Null-trained probe"),
                      (ax2, "Within-context probe")):
        ax.set_xlabel("Classifier P(context | output)")
        ax.set_title(title)
        ax.axhline(0.5, color="grey", ls=":", lw=0.6)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
    ax1.set_ylabel("Probe AUROC")
    fig.suptitle("Phase portrait: behavioural drift (x) vs probe decay (y) — "
                 "big markers = α=max")
    fig.tight_layout()
    fig.savefig(out / "causal_sweep_phase.pdf")
    fig.savefig(out / "causal_sweep_phase.png", dpi=150)
    plt.close(fig)
    print("Wrote", out / "causal_sweep_phase.pdf")

    # ---------- print a quick numeric summary
    print("\nMean values per α:")
    print(f"{'α':>6} {'P(ctx)':>8} {'AUROC null':>12} {'AUROC within':>14}")
    for a in alphas:
        pc = [r["p_context"] for r in data if r["alpha"] == a]
        an = [r["auroc_null"] for r in data
              if r["alpha"] == a and r.get("auroc_null") is not None]
        aw = [r["auroc_within"] for r in data
              if r["alpha"] == a and r.get("auroc_within") is not None]
        print(f"{a:>6.1f} {np.mean(pc):>8.3f} {np.mean(an):>12.3f} "
              f"{np.mean(aw):>14.3f}")


if __name__ == "__main__":
    main()
