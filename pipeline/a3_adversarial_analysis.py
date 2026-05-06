#!/usr/bin/env python3
"""A3: Adversarial vs naturalistic null-probe comparison.

Re-uses n3 (Claude judge via OpenRouter) and the activation extractor on the
adversarial responses, then reports for each (persona, trait) cell:

  - mean judge score on naturalistic vs adversarial responses
    (the persona's natural answer should pull the trait toward LOW)
  - null-probe AUROC vs (judge > 0.5) on naturalistic vs adversarial
    (we expect AUROC to drop more on adversarial — null probe should
    miscalibrate more severely when persona-natural answers contradict the
    trait label)

Headline figure is a paired scatter:
  x = AUROC on naturalistic
  y = AUROC on adversarial
  one point per (persona, trait) cell. Points below the diagonal mean the
  null probe degrades on adversarial relative to naturalistic.

Prereqs:
    pipeline/a1_generate_adversarial_questions.py  -> data/prompts/adversarial/
    pipeline/a2_adversarial_generate.py            -> v2/adversarial/responses/
    pipeline/2_activations.py                       -> v2/adversarial/activations/
    pipeline/n3_naturalistic_judge.py (point at v2/adversarial/responses/)
        -> v2/adversarial/judged/
    same set of files under v2/naturalistic/

Usage:
    python pipeline/a3_adversarial_analysis.py \
        --naturalistic-judged outputs/gemma-2-27b-it/v2/naturalistic/judged \
        --naturalistic-acts   outputs/gemma-2-27b-it/v2/naturalistic/activations \
        --adversarial-judged  outputs/gemma-2-27b-it/v2/adversarial/judged \
        --adversarial-acts    outputs/gemma-2-27b-it/v2/adversarial/activations \
        --probes-dir          outputs/gemma-2-27b-it/v2/caa_probes/probes_pkl \
        --vectors-dir         outputs/gemma-2-27b-it/v2/caa_vectors \
        --output-dir          outputs/gemma-2-27b-it/v2/adversarial/figures
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


PERSONAS = [
    "farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
    "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist",
]
TRAITS = [
    "assertiveness", "empathy", "risk_taking", "honesty",
    "confidence", "deference", "warmth", "impulsivity",
]
TRAIT_LABEL = {t: t.replace("_", " ").title() for t in TRAITS}
LAYER = 22


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--naturalistic-judged", required=True)
    p.add_argument("--naturalistic-acts", required=True)
    p.add_argument("--adversarial-judged", required=True)
    p.add_argument("--adversarial-acts", required=True)
    p.add_argument("--probes-dir", required=True)
    p.add_argument("--vectors-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, default=LAYER)
    return p.parse_args()


def load_acts(act_path: Path, layer: int):
    if not act_path.exists():
        return {}
    data = torch.load(act_path, map_location="cpu", weights_only=True)
    out = {}
    for k, v in data.items():
        if not (k.startswith("v") and "_q" in k):
            continue
        try:
            parts = k.split("_")
            vi = int(parts[0][1:]); qi = int(parts[1][1:])
        except (ValueError, IndexError):
            continue
        a = v[layer].float()
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        out[(vi, qi)] = a.numpy()
    return out


def cell_metrics(judged_path: Path, act_path: Path, probe, scaler, layer: int):
    if not judged_path.exists() or not act_path.exists():
        return None
    acts = load_acts(act_path, layer)
    if not acts:
        return None
    judges, X_rows = [], []
    with open(judged_path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("judge_score") is None:
                continue
            key = (int(e["variant_index"]), int(e["question_index"]))
            if key not in acts:
                continue
            judges.append(float(e["judge_score"]))
            X_rows.append(acts[key])
    if len(judges) < 8:
        return None
    X_s = scaler.transform(np.stack(X_rows))
    # sklearn version-mismatch on probe.multi_class; use sigmoid(w·x + b)
    # directly. Identical to predict_proba()[:, 1] for binary classes [0, 1].
    logits = X_s @ probe.coef_.reshape(-1) + probe.intercept_.item()
    probe_score = 1.0 / (1.0 + np.exp(-logits))
    j = np.array(judges)
    binary = (j > 0.5).astype(int)
    if len(set(binary)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(binary, probe_score))
    return {
        "n": int(len(j)),
        "mean_judge": float(j.mean()),
        "mean_probe": float(probe_score.mean()),
        "auroc": auroc,
        "frac_high": float(binary.mean()),
    }


def load_vec(vec_dir: Path, slug: str, trait: str, layer: int) -> np.ndarray:
    obj = torch.load(vec_dir / f"{slug}_{trait}.pt", map_location="cpu",
                     weights_only=True)
    v = obj["vector"] if isinstance(obj, dict) and "vector" in obj else obj
    return v[layer].float().numpy() if v.ndim == 2 else v.float().numpy()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nj = Path(args.naturalistic_judged); na = Path(args.naturalistic_acts)
    aj = Path(args.adversarial_judged);  aa = Path(args.adversarial_acts)
    probes_dir = Path(args.probes_dir);  vec_dir = Path(args.vectors_dir)

    null_vecs = {t: load_vec(vec_dir, "null", t, args.layer) for t in TRAITS}

    rows = []
    for trait in TRAITS:
        pkl = probes_dir / f"{trait}_A_null.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            pkg = pickle.load(f)
        probe, scaler = pkg["probe"], pkg["scaler"]

        for persona in PERSONAS:
            n_metrics = cell_metrics(nj / f"{persona}_{trait}_judged.jsonl",
                                     na / f"{persona}_{trait}.pt",
                                     probe, scaler, args.layer)
            a_metrics = cell_metrics(aj / f"{persona}_{trait}_judged.jsonl",
                                     aa / f"{persona}_{trait}.pt",
                                     probe, scaler, args.layer)
            if n_metrics is None or a_metrics is None:
                continue
            v_persona = load_vec(vec_dir, persona, trait, args.layer)
            distance = 1.0 - cos(v_persona, null_vecs[trait])
            rows.append({
                "trait": trait, "persona": persona,
                "x_distance": distance,
                "naturalistic": n_metrics,
                "adversarial": a_metrics,
                "auroc_drop": (n_metrics["auroc"] - a_metrics["auroc"])
                              if not (np.isnan(n_metrics["auroc"]) or np.isnan(a_metrics["auroc"])) else None,
            })

    if not rows:
        print("No cells with both naturalistic and adversarial data.")
        return
    (out_dir / "a3_per_cell.json").write_text(json.dumps(rows, indent=2))

    # ----- paired-AUROC scatter -----
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    cmap = plt.get_cmap("tab10")
    color = {t: cmap(i % 10) for i, t in enumerate(TRAITS)}
    nx, ax_y = [], []
    for r in rows:
        n_au = r["naturalistic"]["auroc"]; a_au = r["adversarial"]["auroc"]
        if np.isnan(n_au) or np.isnan(a_au):
            continue
        nx.append(n_au); ax_y.append(a_au)
        ax.scatter(n_au, a_au, color=color[r["trait"]], s=42,
                   alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.6)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("AUROC on naturalistic responses")
    ax.set_ylabel("AUROC on adversarial responses")
    ax.set_title("Null-probe AUROC: adversarial vs naturalistic per cell\n"
                 "points below diagonal = null probe degrades when persona\n"
                 "natural answer disagrees with the trait label")
    ax.grid(alpha=0.25, ls=":")

    if nx:
        from scipy.stats import wilcoxon
        try:
            wstat, wp = wilcoxon(nx, ax_y, alternative="greater")
            ax.text(0.03, 0.95,
                    f"n = {len(nx)}\nmean drop = {np.mean(np.array(nx) - np.array(ax_y)):+.3f}\n"
                    f"Wilcoxon p (nat > adv) = {wp:.1e}",
                    transform=ax.transAxes, fontsize=9, va="top")
        except ValueError:
            pass

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="", color=color[t],
                      markersize=6, label=TRAIT_LABEL[t])
               for t in TRAITS if any(r["trait"] == t for r in rows)]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False,
              ncol=2, title="Trait")

    fig.tight_layout()
    fig.savefig(out_dir / "a3_paired_auroc.pdf")
    fig.savefig(out_dir / "a3_paired_auroc.png", dpi=180)
    plt.close(fig)
    print(f"Saved {out_dir/'a3_paired_auroc.pdf'} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
