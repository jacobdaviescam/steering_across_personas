#!/usr/bin/env python3
"""X4d: Text-classifier accuracy per PERSONA (not per trait).

Reads X1's predictions.jsonl (one row per held-out test response),
computes per-persona recall (= accuracy restricted to that persona's
responses), precision, and confusion.

Outputs:
  per_persona_accuracy.pdf/png          — bar chart, recall per persona
  per_persona_precision_recall.pdf/png  — dual-bar precision/recall
  confusion_matrix.pdf/png              — confusion heatmap (row-normalised)
  per_persona_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier-dir", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cdir = Path(args.classifier_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((cdir / "metrics.json").read_text())
    contexts = metrics["config"]["contexts"]
    chance = metrics["chance"]
    ctx_to_idx = {c: i for i, c in enumerate(contexts)}

    preds = []
    with open(cdir / "predictions.jsonl") as f:
        for line in f:
            preds.append(json.loads(line))
    print(f"Loaded {len(preds)} predictions over {len(contexts)} contexts.")

    # Per-persona recall and precision
    truth_counts = Counter(p["context"] for p in preds)
    correct = Counter()
    pred_counts = Counter()
    confusion = np.zeros((len(contexts), len(contexts)), dtype=int)
    for p in preds:
        y_true = p["context"]
        y_pred = p["predicted"]
        pred_counts[y_pred] += 1
        if y_true == y_pred:
            correct[y_true] += 1
        if y_true in ctx_to_idx and y_pred in ctx_to_idx:
            confusion[ctx_to_idx[y_true], ctx_to_idx[y_pred]] += 1

    recall = {c: correct[c] / truth_counts[c] if truth_counts[c] else 0.0
              for c in contexts}
    precision = {}
    for c in contexts:
        tp = correct[c]
        fp = pred_counts[c] - tp
        denom = tp + fp
        precision[c] = tp / denom if denom else 0.0
    f1 = {c: (2 * precision[c] * recall[c] / (precision[c] + recall[c])
              if (precision[c] + recall[c]) else 0.0)
          for c in contexts}

    stats = {
        "contexts": contexts,
        "chance": chance,
        "overall_accuracy": metrics["overall_accuracy"],
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "truth_counts": dict(truth_counts),
        "pred_counts": dict(pred_counts),
    }
    with open(out / "per_persona_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ---- Figure 1: recall bar chart, sorted
    order = sorted(contexts, key=lambda c: recall[c], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [recall[c] for c in order]
    colors = ["#3a7ca5" if c not in ("null", "nonsense")
              else ("#daa520" if c == "null" else "#888") for c in order]
    bars = ax.bar(order, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(chance, color="grey", ls="--", label=f"chance = {chance:.3f}")
    ax.set_ylabel("Recall  (P(predicted = persona | true = persona))")
    ax.set_xlabel("Persona (true label)")
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(order, rotation=45, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                f"{v:.2f}", ha="center", fontsize=8)
    ax.set_title(f"Text classifier: per-persona recall "
                 f"(overall={metrics['overall_accuracy']:.2f}, "
                 f"chance={chance:.2f})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "per_persona_accuracy.pdf")
    fig.savefig(out / "per_persona_accuracy.png", dpi=150)
    plt.close(fig)
    print("Wrote per_persona_accuracy.pdf")

    # ---- Figure 2: precision + recall dual-bar
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, [recall[c] for c in order], w,
           label="Recall", color="#3a7ca5")
    ax.bar(x + w / 2, [precision[c] for c in order], w,
           label="Precision", color="#a23a3a")
    ax.axhline(chance, color="grey", ls="--", lw=0.7,
               label=f"chance = {chance:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Text classifier: precision and recall per persona")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "per_persona_precision_recall.pdf")
    fig.savefig(out / "per_persona_precision_recall.png", dpi=150)
    plt.close(fig)
    print("Wrote per_persona_precision_recall.pdf")

    # ---- Figure 3: confusion matrix (row-normalised)
    row_sum = confusion.sum(axis=1, keepdims=True).clip(min=1)
    conf_norm = confusion / row_sum
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(conf_norm, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(contexts)))
    ax.set_yticks(range(len(contexts)))
    ax.set_xticklabels(contexts, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(contexts, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(contexts)):
        for j in range(len(contexts)):
            v = conf_norm[i, j]
            if v > 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Confusion matrix (row-normalised) — text classifier")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.pdf")
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Wrote confusion_matrix.pdf")

    # ---- print summary
    print("\nPer-persona recall (sorted):")
    for c in order:
        print(f"  {c:<22} recall={recall[c]:.3f}  "
              f"precision={precision[c]:.3f}  "
              f"f1={f1[c]:.3f}  n_true={truth_counts[c]}")


if __name__ == "__main__":
    main()
