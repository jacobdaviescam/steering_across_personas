#!/usr/bin/env python3
"""X8b: Plot X8 steer-then-probe results.

Produces four figures from x8's summary.json:
  x8_single_sweep.png       -- cos(probe) vs alpha at L_steer, per pair, per probe
  x8_mix_grid.png           -- 2D heatmaps over (alpha_ctx, alpha_trait) at L_steer
  x8_layer_propagation.png  -- cos(probe) vs extraction layer at alpha=1
  x8_null_vs_persona.png    -- direct comparison of v_null and v_persona probe
                               AUROC vs alpha, per pair, per steering condition
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COND_COLORS = {"ctx": "#1f77b4", "trait": "#d62728", "random": "#7f7f7f"}
COND_LABEL = {"ctx": r"steer $u_C$", "trait": r"steer $v_{T,\mathrm{null}}$",
              "random": "random"}

PROBE_LABEL = {
    "ctx": r"cos($h$, $\hat u_C$)",
    "trait_null": r"cos($h$, $\hat v_{T,\mathrm{null}}$)",
    "trait_persona": r"cos($h$, $\hat v_{T,C}$)",
}
AUROC_LABEL = {
    "ctx": r"AUROC$(\hat u_C)$",
    "trait_null": r"AUROC$(\hat v_{T,\mathrm{null}})$",
    "trait_persona": r"AUROC$(\hat v_{T,C})$",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def _probes_for(pd):
    probes = ["ctx", "trait_null"]
    if pd.get("has_persona_probe", False):
        probes.append("trait_persona")
    return probes


def plot_single_sweep(summary, out_path):
    pairs = list(summary["pairs"].items())
    L_steer = summary["config"]["layer_steer"]
    # max probes across pairs (determines column count)
    n_cols = max(len(_probes_for(pd)) for _, pd in pairs)
    fig, axes = plt.subplots(len(pairs), n_cols,
                             figsize=(3.6 * n_cols, 2.8 * len(pairs)),
                             sharex=True, squeeze=False)

    for row, (pair_key, pd) in enumerate(pairs):
        probes = _probes_for(pd)
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(probes):
                ax.axis("off")
                continue
            probe = probes[col]
            key = f"cos_{probe}_mean"
            for cond in ["ctx", "trait", "random"]:
                pts = [(r["alpha"], r[key])
                       for r in pd["single"]
                       if r["condition"] == cond and r["layer"] == L_steer]
                pts.sort()
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=COND_COLORS[cond],
                        label=COND_LABEL[cond], linewidth=1.6, markersize=4)
            ax.axhline(0, color="k", lw=0.5, alpha=0.3)
            ax.set_ylabel(PROBE_LABEL[probe])
            if row == len(pairs) - 1:
                ax.set_xlabel(r"steering $\alpha$")
            if row == 0 and col == n_cols - 1:
                ax.legend(loc="best", fontsize=7)
            ax.grid(alpha=0.3)
            title = pair_key.replace("_", " ")
            if col == 0:
                title += (f"  cos(u,v_null)={pd['cos_u_v_null']:+.2f}"
                          f"  cos(u,v_C)={pd['cos_u_v_persona']:+.2f}"
                          f"  cos(v_null,v_C)={pd['cos_v_null_v_persona']:+.2f}")
            ax.set_title(title, fontsize=8)

    fig.suptitle(f"Single-direction sweeps at L{L_steer}", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_mix_grid(summary, out_path):
    pairs = list(summary["pairs"].items())
    L_steer = summary["config"]["layer_steer"]
    n_cols = max(len(_probes_for(pd)) for _, pd in pairs)
    fig, axes = plt.subplots(len(pairs), n_cols,
                             figsize=(3.6 * n_cols, 3.2 * len(pairs)),
                             squeeze=False)

    # symmetric vabs per probe across pairs
    def collect(probe):
        key = f"cos_{probe}_mean"
        vals = [r[key]
                for _, pd in pairs
                for r in pd["mix"]
                if r["layer"] == L_steer and key in r]
        return max(abs(min(vals)), abs(max(vals)))

    probe_vabs = {p: collect(p) for p in ["ctx", "trait_null", "trait_persona"]
                  if any(p in _probes_for(pd) for _, pd in pairs)}

    for row, (pair_key, pd) in enumerate(pairs):
        probes = _probes_for(pd)
        alphas = sorted({r["alpha_ctx"] for r in pd["mix"]})
        n = len(alphas)
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(probes):
                ax.axis("off")
                continue
            probe = probes[col]
            key = f"cos_{probe}_mean"
            vabs = probe_vabs[probe]
            grid = np.zeros((n, n))
            for r in pd["mix"]:
                if r["layer"] != L_steer or key not in r:
                    continue
                i = alphas.index(r["alpha_ctx"])
                j = alphas.index(r["alpha_trait"])
                grid[i, j] = r[key]
            im = ax.imshow(grid, origin="lower", cmap="RdBu_r",
                           vmin=-vabs, vmax=vabs, aspect="auto")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([f"{a:g}" for a in alphas])
            ax.set_yticklabels([f"{a:g}" for a in alphas])
            ax.set_xlabel(r"$\alpha_{\mathrm{trait}}$")
            ax.set_ylabel(r"$\alpha_{\mathrm{ctx}}$")
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{grid[i, j]:+.2f}",
                            ha="center", va="center", fontsize=6,
                            color=("white" if abs(grid[i, j]) > 0.55 * vabs
                                   else "black"))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=PROBE_LABEL[probe])
            title = pair_key.replace("_", " ")
            if col == 0:
                title += f"  (cos(u,v_null)={pd['cos_u_v_null']:+.2f})"
            ax.set_title(title, fontsize=8)

    fig.suptitle(f"2D mix grid at L{L_steer}", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_layer_propagation(summary, out_path):
    pairs = list(summary["pairs"].items())
    L_steer = summary["config"]["layer_steer"]
    n_cols = max(len(_probes_for(pd)) for _, pd in pairs)
    fig, axes = plt.subplots(len(pairs), n_cols,
                             figsize=(3.6 * n_cols, 2.8 * len(pairs)),
                             sharex=True, squeeze=False)

    for row, (pair_key, pd) in enumerate(pairs):
        probes = _probes_for(pd)
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(probes):
                ax.axis("off")
                continue
            probe = probes[col]
            key = f"cos_{probe}_mean"
            base = {r["layer"]: r[key]
                    for r in pd["single"]
                    if r["condition"] == "ctx" and r["alpha"] == 0.0}
            for cond in ["ctx", "trait"]:
                pts = [(r["layer"], r[key])
                       for r in pd["single"]
                       if r["condition"] == cond and r["alpha"] == 1.0]
                pts.sort()
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=COND_COLORS[cond],
                        label=f"{COND_LABEL[cond]} (α=1)",
                        linewidth=1.8, markersize=5)
            bx = sorted(base)
            ax.plot(bx, [base[l] for l in bx], "s--", color="#555",
                    label="baseline (α=0)", linewidth=1.2, markersize=4)
            ax.axvline(L_steer, color="k", linestyle=":", alpha=0.4)
            ax.set_ylabel(PROBE_LABEL[probe])
            if row == len(pairs) - 1:
                ax.set_xlabel("extraction layer")
            if row == 0 and col == n_cols - 1:
                ax.legend(loc="best", fontsize=6)
            ax.grid(alpha=0.3)
            title = pair_key.replace("_", " ")
            ax.set_title(title, fontsize=8)

    fig.suptitle("Layer propagation of steering effect (α=1)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_null_vs_persona(summary, out_path):
    """Direct comparison: does v_persona probe fire more strongly than v_null
    probe on the same steered activations?

    Layout: one row per pair, one column per (metric, steering condition).
    Each panel shows only two lines -- v_null (green) vs v_persona (purple)
    -- so the key is trivial.
    """
    pairs = [(k, pd) for k, pd in summary["pairs"].items()
             if pd.get("has_persona_probe", False)]
    if not pairs:
        print(f"skip {out_path} — no persona probe available")
        return
    L_steer = summary["config"]["layer_steer"]

    # columns: (metric_prefix, steering_condition, ylabel, panel_title)
    cols = [
        ("cos",   "ctx",   r"cos($h$, probe)",  r"steer $u_C$"),
        ("cos",   "trait", r"cos($h$, probe)",  r"steer $v_{T,\mathrm{null}}$"),
        ("auroc", "ctx",   "AUROC vs baseline", r"steer $u_C$"),
        ("auroc", "trait", "AUROC vs baseline", r"steer $v_{T,\mathrm{null}}$"),
    ]

    fig, axes = plt.subplots(len(pairs), len(cols),
                             figsize=(3.2 * len(cols), 2.7 * len(pairs)),
                             sharex=True, squeeze=False)

    probe_colors = {"trait_null": "#2ca02c", "trait_persona": "#9467bd"}
    probe_labels = {
        "trait_null": r"probe $\hat v_{T,\mathrm{null}}$",
        "trait_persona": r"probe $\hat v_{T,C}$",
    }

    for row, (pair_key, pd) in enumerate(pairs):
        for col, (mpref, cond, ylabel, panel_title) in enumerate(cols):
            ax = axes[row, col]
            for probe in ["trait_null", "trait_persona"]:
                key = (f"cos_{probe}_mean" if mpref == "cos"
                       else f"auroc_{probe}_vs_base")
                pts = [(r["alpha"], r[key])
                       for r in pd["single"]
                       if r["condition"] == cond and r["layer"] == L_steer]
                pts.sort()
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", color=probe_colors[probe],
                        label=probe_labels[probe],
                        linewidth=1.8, markersize=5)
            if mpref == "auroc":
                ax.axhline(0.5, color="k", lw=0.5, alpha=0.3)
                ax.set_ylim(-0.02, 1.05)
            else:
                ax.axhline(0, color="k", lw=0.5, alpha=0.3)
            if col == 0:
                ax.set_ylabel(
                    f"{pair_key.replace('_', ' ')}\n"
                    f"cos(v_null,v_C)={pd['cos_v_null_v_persona']:+.2f}\n\n"
                    f"{ylabel}",
                    fontsize=8)
            elif col in (0, 2):
                ax.set_ylabel(ylabel)
            if row == len(pairs) - 1:
                ax.set_xlabel(r"steering $\alpha$")
            if row == 0:
                ax.set_title(
                    f"{'cos' if mpref == 'cos' else 'AUROC'}  |  "
                    f"{panel_title}", fontsize=9)
            if row == 0 and col == len(cols) - 1:
                ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.3)

    fig.suptitle(r"$\hat v_{T,\mathrm{null}}$ vs $\hat v_{T,C}$ probe at "
                 f"L{L_steer}", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    args = parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    out_dir = Path(args.output_dir) if args.output_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_single_sweep(summary, out_dir / "x8_single_sweep.png")
    plot_mix_grid(summary, out_dir / "x8_mix_grid.png")
    plot_layer_propagation(summary, out_dir / "x8_layer_propagation.png")
    plot_null_vs_persona(summary, out_dir / "x8_null_vs_persona.png")


if __name__ == "__main__":
    main()
