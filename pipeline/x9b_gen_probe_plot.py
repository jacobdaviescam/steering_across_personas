#!/usr/bin/env python3
"""X9b: Plots for steer-generate-then-probe (x9) + judge (x9c).

Reads:
    {gen_probe_dir}/summary.json          (activation-space cos + AUROC)
    {gen_probe_dir}/judge_summary.json    (per-condition judge mean/stderr)

Writes (PNG) into {gen_probe_dir}/:
    x9_divergence.png        cos vs α_trait | judge vs α_trait, per pair
    x9_probe_vs_judge.png    scatter: cos_v_null vs mean judge score
    x9_judge_heatmap.png     α_ctx × α_trait judge mean, per pair
    x9_cos_heatmap.png       α_ctx × α_trait cos_v_null / cos_v_persona,
                             per pair

Usage:
    python pipeline/x9b_gen_probe_plot.py \\
        --gen-probe-dir outputs/gemma-2-27b-it/v2/gen_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-probe-dir", required=True)
    p.add_argument("--layer", type=int, default=22,
                   help="Layer to read cos values from")
    return p.parse_args()


def load_summaries(gp: Path) -> tuple[dict, dict | None]:
    summary = json.loads((gp / "summary.json").read_text())
    judge_path = gp / "judge_summary.json"
    judge = json.loads(judge_path.read_text()) if judge_path.exists() else None
    return summary, judge


def cos_records_at_layer(pair_block: dict, layer: int) -> list[dict]:
    return [r for r in pair_block["records"] if r["layer"] == layer]


def judge_conditions(judge_block: dict) -> dict[tuple[float, float], dict]:
    out = {}
    for c in judge_block["conditions"]:
        out[(c["alpha_ctx"], c["alpha_trait"])] = c
    return out


# ---------------------------------------------------------------------------
def plot_divergence(summary: dict, judge: dict | None, layer: int,
                    out_path: Path) -> None:
    pairs = list(summary["pairs"].keys())
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n), squeeze=False)

    for pi, pair_name in enumerate(pairs):
        block = summary["pairs"][pair_name]
        recs = cos_records_at_layer(block, layer)
        has_persona = block.get("has_persona_probe", False)

        ctx_vals = sorted({r["alpha_ctx"] for r in recs})
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.15, 0.85, max(len(ctx_vals), 2)))
        color_by_ctx = {a: colors[i] for i, a in enumerate(ctx_vals)}

        ax_cos = axes[pi, 0]
        for a_ctx in ctx_vals:
            sub = sorted([r for r in recs if r["alpha_ctx"] == a_ctx],
                         key=lambda r: r["alpha_trait"])
            xs = [r["alpha_trait"] for r in sub]
            ax_cos.plot(xs, [r["cos_trait_null_mean"] for r in sub],
                        marker="o", color=color_by_ctx[a_ctx],
                        label=f"v_null, α_ctx={a_ctx:g}", linestyle="-")
            if has_persona:
                ax_cos.plot(xs, [r["cos_trait_persona_mean"] for r in sub],
                            marker="s", color=color_by_ctx[a_ctx],
                            label=f"v_C,    α_ctx={a_ctx:g}", linestyle="--",
                            alpha=0.85)
        ax_cos.set_xlabel("α_trait")
        ax_cos.set_ylabel(f"cos(h_gen, ·) at L{layer}")
        ax_cos.set_title(f"{pair_name} — activation-space probe")
        ax_cos.axhline(0, color="grey", lw=0.5)
        ax_cos.grid(True, alpha=0.25)
        if pi == 0:
            ax_cos.legend(fontsize=7, ncol=2, loc="best")

        ax_j = axes[pi, 1]
        if judge and pair_name in judge.get("pairs", {}):
            conds = judge_conditions(judge["pairs"][pair_name])
            for a_ctx in ctx_vals:
                xs, means, ses = [], [], []
                for (ac, at), stats in sorted(conds.items()):
                    if ac != a_ctx:
                        continue
                    xs.append(at)
                    means.append(stats["mean_score"])
                    ses.append(stats.get("stderr", 0.0) or 0.0)
                ax_j.errorbar(xs, means, yerr=ses, marker="o",
                              color=color_by_ctx[a_ctx],
                              label=f"α_ctx={a_ctx:g}", capsize=3)
            ax_j.set_ylim(0, 1)
            ax_j.axhline(0.5, color="grey", lw=0.5, linestyle=":")
            ax_j.set_title(f"{pair_name} — Claude-judge trait score")
            if pi == 0:
                ax_j.legend(fontsize=7, loc="best")
        else:
            ax_j.text(0.5, 0.5, "judge_summary.json not found",
                      ha="center", va="center", transform=ax_j.transAxes,
                      color="grey")
            ax_j.set_title(f"{pair_name} — judge (missing)")
        ax_j.set_xlabel("α_trait")
        ax_j.set_ylabel("mean judge score")
        ax_j.grid(True, alpha=0.25)

    fig.suptitle("x9: activation probe vs behavioural judge", y=1.0,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_probe_vs_judge(summary: dict, judge: dict | None, layer: int,
                        out_path: Path) -> None:
    if judge is None:
        return
    pairs = list(summary["pairs"].keys())
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)

    for pi, pair_name in enumerate(pairs):
        ax = axes[0, pi]
        block = summary["pairs"][pair_name]
        has_persona = block.get("has_persona_probe", False)
        recs = {(r["alpha_ctx"], r["alpha_trait"]): r
                for r in cos_records_at_layer(block, layer)}
        jconds = (judge_conditions(judge["pairs"].get(pair_name, {}))
                  if pair_name in judge.get("pairs", {}) else {})

        xs_null, xs_persona, ys = [], [], []
        labels = []
        for key, stats in jconds.items():
            if key not in recs:
                continue
            r = recs[key]
            xs_null.append(r["cos_trait_null_mean"])
            if has_persona:
                xs_persona.append(r["cos_trait_persona_mean"])
            ys.append(stats["mean_score"])
            labels.append(f"({key[0]:g},{key[1]:g})")

        if not xs_null:
            ax.text(0.5, 0.5, "no overlap", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_title(pair_name)
            continue

        ax.scatter(xs_null, ys, marker="o", color="tab:green",
                   label="v_null probe", s=45)
        if has_persona:
            ax.scatter(xs_persona, ys, marker="s", color="tab:purple",
                       label="v_C probe", s=45, alpha=0.85)
        for x, y, lab in zip(xs_null, ys, labels):
            ax.annotate(lab, (x, y), fontsize=6, xytext=(3, 3),
                        textcoords="offset points", color="tab:green")

        ax.set_xlabel(f"cos(h_gen, v_trait) at L{layer}")
        ax.set_ylabel("mean judge score")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="grey", lw=0.5, linestyle=":")
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(pair_name)
        ax.grid(True, alpha=0.25)
        if pi == 0:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("x9: linear probe vs behavioural judge — per condition",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _heatmap(ax, rows, cols, Z, title, vmin=None, vmax=None, cmap="viridis",
             fmt="{:.2f}"):
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c:g}" for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r:g}" for r in rows])
    ax.set_xlabel("α_trait")
    ax.set_ylabel("α_ctx")
    ax.set_title(title)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            val = Z[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    fontsize=7,
                    color="white" if vmax is None or val < 0.6 * vmax
                    else "black")
    return im


def plot_judge_heatmap(judge: dict | None, out_path: Path) -> None:
    if judge is None:
        return
    pairs = list(judge.get("pairs", {}).keys())
    if not pairs:
        return
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.6), squeeze=False)

    for pi, pair_name in enumerate(pairs):
        conds = judge["pairs"][pair_name]["conditions"]
        ctx_vals = sorted({c["alpha_ctx"] for c in conds})
        tr_vals = sorted({c["alpha_trait"] for c in conds})
        Z = np.full((len(ctx_vals), len(tr_vals)), np.nan)
        for c in conds:
            i = ctx_vals.index(c["alpha_ctx"])
            j = tr_vals.index(c["alpha_trait"])
            Z[i, j] = c["mean_score"]
        ax = axes[0, pi]
        im = _heatmap(ax, ctx_vals, tr_vals, Z,
                      f"{pair_name}\nmean judge score",
                      vmin=0.0, vmax=1.0, cmap="RdYlBu_r",
                      fmt="{:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cos_heatmap(summary: dict, layer: int, out_path: Path) -> None:
    pairs = list(summary["pairs"].keys())
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(3.8 * n, 6.4), squeeze=False)

    for pi, pair_name in enumerate(pairs):
        block = summary["pairs"][pair_name]
        has_persona = block.get("has_persona_probe", False)
        recs = cos_records_at_layer(block, layer)
        ctx_vals = sorted({r["alpha_ctx"] for r in recs})
        tr_vals = sorted({r["alpha_trait"] for r in recs})
        Z_null = np.full((len(ctx_vals), len(tr_vals)), np.nan)
        Z_pers = np.full((len(ctx_vals), len(tr_vals)), np.nan)
        for r in recs:
            i = ctx_vals.index(r["alpha_ctx"])
            j = tr_vals.index(r["alpha_trait"])
            Z_null[i, j] = r["cos_trait_null_mean"]
            if has_persona:
                Z_pers[i, j] = r["cos_trait_persona_mean"]

        vmax = max(abs(np.nanmin(Z_null)), abs(np.nanmax(Z_null)),
                   abs(np.nanmin(Z_pers)) if has_persona else 0.0,
                   abs(np.nanmax(Z_pers)) if has_persona else 0.0)
        vmin = -vmax

        ax0 = axes[0, pi]
        im = _heatmap(ax0, ctx_vals, tr_vals, Z_null,
                      f"{pair_name}\ncos(h_gen, v_null) L{layer}",
                      vmin=vmin, vmax=vmax, cmap="RdBu_r",
                      fmt="{:+.2f}")
        fig.colorbar(im, ax=ax0, fraction=0.04, pad=0.04)

        ax1 = axes[1, pi]
        if has_persona:
            im = _heatmap(ax1, ctx_vals, tr_vals, Z_pers,
                          f"{pair_name}\ncos(h_gen, v_C) L{layer}",
                          vmin=vmin, vmax=vmax, cmap="RdBu_r",
                          fmt="{:+.2f}")
            fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.04)
        else:
            ax1.text(0.5, 0.5, "no persona probe",
                     ha="center", va="center", transform=ax1.transAxes,
                     color="grey")
            ax1.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    gp = Path(args.gen_probe_dir)
    summary, judge = load_summaries(gp)

    plot_divergence(summary, judge, args.layer, gp / "x9_divergence.png")
    plot_probe_vs_judge(summary, judge, args.layer,
                        gp / "x9_probe_vs_judge.png")
    plot_judge_heatmap(judge, gp / "x9_judge_heatmap.png")
    plot_cos_heatmap(summary, args.layer, gp / "x9_cos_heatmap.png")
    print(f"Wrote PNGs to {gp}/")


if __name__ == "__main__":
    main()
