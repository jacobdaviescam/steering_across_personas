#!/usr/bin/env python3
"""X4: Paper figures (Fig 1, Fig 2, Fig 3) for the causal-figures branch.

Fig 1 — Behavioural context-sensitivity per trait (from X1):
        bar chart of per-trait classifier accuracy + confusion-matrix appendix.

Fig 2 — Probe AUROC vs trait context-sensitivity (from X1 + X2):
        scatter with three series (within-context, null-trained, multi-context).

Fig 3 — Causal phase portrait (from X3c):
        P(C | output) on x, null-probe AUROC on y, parameterised by alpha.

Usage:
    python pipeline/x4_figures.py \\
        --classifier-dir outputs/gemma-2-27b-it/v2/classifier \\
        --probes-dir outputs/gemma-2-27b-it/v2/probes \\
        --sweep-results outputs/gemma-2-27b-it/v2/causal/metrics/sweep_results.json \\
        --output-dir outputs/gemma-2-27b-it/v2/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from persona_steering.utils import derive_model_short_from_path, log, save_fig
from persona_steering.wandb_utils import (
    finish_run, init_run, log_artifact, log_images,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--classifier-dir", type=str, required=True)
    p.add_argument("--probes-dir", type=str, required=True)
    p.add_argument("--sweep-results", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--no-fig3", action="store_true",
                   help="Skip Fig 3 (sweep results may not exist yet)")
    return p.parse_args()


def fig1(classifier_dir: Path, out: Path) -> dict[str, float]:
    metrics = json.loads((classifier_dir / "metrics.json").read_text())
    per_trait = metrics["per_trait_accuracy"]
    chance = metrics["chance"]

    traits = sorted(per_trait.keys())
    accs = [per_trait[t] for t in traits]
    order = np.argsort(accs)
    traits = [traits[i] for i in order]
    accs = [accs[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(traits, accs, color="#3a7ca5")
    ax.axvline(chance, color="grey", ls="--", label=f"chance = {chance:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Classifier accuracy on held-out questions")
    ax.set_title("Fig 1 — Behavioural context-sensitivity per trait")
    ax.legend(loc="lower right")
    save_fig(fig, out / "fig1_per_trait_accuracy.pdf")

    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["context_labels"]
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im = ax2.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
    ax2.set_title("Fig 1 (appendix) — Confusion matrix")
    plt.colorbar(im, ax=ax2, fraction=0.046)
    save_fig(fig2, out / "fig1_confusion_matrix.pdf")

    return per_trait


def fig2(classifier_dir: Path, probes_dir: Path, out: Path) -> None:
    cls_metrics = json.loads((classifier_dir / "metrics.json").read_text())
    sensitivity = cls_metrics["per_trait_accuracy"]
    probe_metrics = json.loads((probes_dir / "metrics.json").read_text())["results"]

    traits = sorted(set(sensitivity) & set(probe_metrics))
    if not traits:
        log.warning("Fig 2: no overlapping traits, skipping")
        return

    rows = []
    for t in traits:
        x = sensitivity[t]
        a = probe_metrics[t]["A"]
        b = probe_metrics[t]["B"]
        c = probe_metrics[t].get("C", {})

        a_vals = [v for v in a.values() if v is not None]
        b_vals = [v for v in b.values() if v is not None]

        within_ceiling = None
        if c:
            diag = []
            for tr_ctx, row in c.items():
                v = row.get(tr_ctx)
                if v is not None:
                    diag.append(v)
            if diag:
                within_ceiling = float(np.mean(diag))

        rows.append({
            "trait": t,
            "x": x,
            "null_trained": float(np.mean(a_vals)) if a_vals else np.nan,
            "multi_context": float(np.mean(b_vals)) if b_vals else np.nan,
            "within": within_ceiling if within_ceiling is not None else np.nan,
        })

    xs = np.array([r["x"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5))

    series = [
        ("within", "Within-context (ceiling)", "#444"),
        ("multi_context", "Multi-context trained, held-out", "#1f7a1f"),
        ("null_trained", "Null-trained, cross-context", "#c0392b"),
    ]
    for key, label, color in series:
        ys = np.array([r[key] for r in rows])
        ax.scatter(xs, ys, label=label, color=color, s=60, edgecolor="white")
        mask = ~np.isnan(ys)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(xs[mask], ys[mask], 1)
            xfit = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xfit, slope * xfit + intercept, color=color, ls="--", alpha=0.5)

    for r in rows:
        ax.annotate(r["trait"], (r["x"], r.get("null_trained", np.nan)),
                    fontsize=8, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Behavioural context-sensitivity (Fig 1)")
    ax.set_ylabel("Probe AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="grey", ls=":", lw=0.5)
    ax.legend(loc="lower left")
    ax.set_title("Fig 2 — Cross-context probe degradation tracks behavioural drift")
    save_fig(fig, out / "fig2_probe_vs_sensitivity.pdf")


def fig3(sweep_results_path: Path, out: Path) -> None:
    data = json.loads(sweep_results_path.read_text())["results"]
    if not data:
        log.warning("Fig 3: empty sweep results")
        return

    by_curve: dict[tuple[str, str, str], list[dict]] = {}
    for row in data:
        key = (row["trait"], row["context"], row["condition"])
        by_curve.setdefault(key, []).append(row)
    for k, rows in by_curve.items():
        rows.sort(key=lambda r: r["alpha"])

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {"main": "#1f4e79", "rand": "#999999", "trait": "#a23a3a"}
    labelled: set[str] = set()

    for (trait, ctx, cond), rows in by_curve.items():
        xs = [r["p_context"] for r in rows]
        ys = [r["auroc"] if r["auroc"] is not None else np.nan for r in rows]
        ls = "-" if cond == "main" else "--"
        label = cond if cond not in labelled else None
        labelled.add(cond)
        ax.plot(xs, ys, ls=ls, marker="o", color=palette.get(cond, "#333"),
                alpha=0.8 if cond == "main" else 0.45, label=label)
        ax.annotate(f"{trait[:3]}/{ctx[:5]}", (xs[-1], ys[-1]),
                    fontsize=6, alpha=0.5)

    ax.set_xlabel("Classifier P(context | output)")
    ax.set_ylabel("Null-trained probe AUROC")
    ax.set_xlim(0, 1.0); ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="grey", ls=":", lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left", title="Condition")
    ax.set_title("Fig 3 — Phase portrait: behaviour rises, probe decays")
    save_fig(fig, out / "fig3_phase_portrait.pdf")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_short = derive_model_short_from_path(args.classifier_dir)
    init_run("x4_figures", model_short, config=vars(args), method="causal-figures")

    log.info("Building Fig 1...")
    fig1(Path(args.classifier_dir), out)

    log.info("Building Fig 2...")
    fig2(Path(args.classifier_dir), Path(args.probes_dir), out)

    if not args.no_fig3 and args.sweep_results:
        log.info("Building Fig 3...")
        fig3(Path(args.sweep_results), out)

    log.info("Figures saved to %s", out)
    log_images(out, prefix="figures")
    log_artifact(f"{model_short}-x4-figures", "figures", out, glob_pattern="*.pdf")
    finish_run()


if __name__ == "__main__":
    main()
