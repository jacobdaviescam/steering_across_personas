#!/usr/bin/env python3
"""N4: Naturalistic-response Fig 3 v2.

Joins:
  * judged responses from n3_naturalistic_judge.py
  * activation tensors from 2_activations.py (run on n1 outputs)
  * null-trained probes from x2_probe_regimes.py (the {trait}_A_null.pkl files)
  * CAA vectors from caa_vectors/ (for the 1 - cos(v_T,c, v_T,null) x-axis)

For each (persona, trait) cell:
  - Apply the null probe to the response activations -> probe_score per resp
  - Pearson r between probe_score and judge_score (the "measurement quality"
    of the null probe at that cell)
  - AUROC of probe_score against (judge_score > 0.5) as a binary label

Plot:
  x = 1 - cos(v_T,c, v_T,null)
  y = |Pearson r|  (panel a)  /  AUROC  (panel b)
  one point per (persona, trait) cell, colour by trait.

This is the experiment Fig 3 in the paper *should* be reporting -- the
deployment-monitoring quantity, not contrastive-pair AUROC.

Usage:
    python pipeline/n4_naturalistic_eval.py \
        --judged-dir       outputs/gemma-2-27b-it/v2/naturalistic/judged \
        --activations-dir  outputs/gemma-2-27b-it/v2/naturalistic/activations \
        --probes-dir       outputs/gemma-2-27b-it/v2/caa_probes/probes_pkl \
        --vectors-dir      outputs/gemma-2-27b-it/v2/caa_vectors \
        --output-dir       outputs/gemma-2-27b-it/v2/naturalistic/figures
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--judged-dir", required=True)
    p.add_argument("--activations-dir", required=True)
    p.add_argument("--probes-dir", required=True)
    p.add_argument("--vectors-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, default=LAYER)
    return p.parse_args()


def load_vec(vec_dir: Path, slug: str, trait: str, layer: int) -> np.ndarray:
    obj = torch.load(vec_dir / f"{slug}_{trait}.pt", map_location="cpu",
                     weights_only=True)
    v = obj["vector"] if isinstance(obj, dict) and "vector" in obj else obj
    return v[layer].float().numpy() if v.ndim == 2 else v.float().numpy()


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_probe(probes_dir: Path, trait: str):
    pkl_path = probes_dir / f"{trait}_A_null.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_activations(act_path: Path, layer: int) -> dict[tuple[int, int], np.ndarray]:
    """Returns {(variant_index, question_index): activation_vector}."""
    if not act_path.exists():
        return {}
    data = torch.load(act_path, map_location="cpu", weights_only=True)
    out: dict[tuple[int, int], np.ndarray] = {}
    for k, v in data.items():
        # 2_activations.py uses 'v{vi}_q{qi}'
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


def perm_pearson(x: np.ndarray, y: np.ndarray, n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    r0 = pearsonr(x, y).statistic
    rs = np.empty(n)
    for i in range(n):
        rs[i] = pearsonr(x, rng.permutation(y)).statistic
    return float(r0), float((np.abs(rs) >= abs(r0)).mean())


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judged_dir = Path(args.judged_dir)
    act_dir = Path(args.activations_dir)
    probes_dir = Path(args.probes_dir)
    vec_dir = Path(args.vectors_dir)

    # cache vectors
    null_vecs = {t: load_vec(vec_dir, "null", t, args.layer) for t in TRAITS}

    rows = []
    for trait in TRAITS:
        probe_pkg = load_probe(probes_dir, trait)
        if probe_pkg is None:
            continue
        probe = probe_pkg["probe"]
        scaler = probe_pkg["scaler"]

        for persona in PERSONAS:
            judged_path = judged_dir / f"{persona}_{trait}_judged.jsonl"
            act_path = act_dir / f"{persona}_{trait}.pt"
            if not judged_path.exists() or not act_path.exists():
                continue

            acts = load_activations(act_path, args.layer)
            if not acts:
                continue

            judges = []
            X_rows = []
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

            if len(judges) < 8:  # need enough for stable correlation
                continue

            X = np.stack(X_rows)
            X_s = scaler.transform(X)
            # Bypass probe.predict_proba (sklearn version-mismatch on
            # multi_class attribute). Compute sigmoid(w·x + b) directly;
            # this is identical to predict_proba for a binary
            # LogisticRegression(CV) classifier with classes [0, 1].
            logits = X_s @ probe.coef_.reshape(-1) + probe.intercept_.item()
            probe_score = 1.0 / (1.0 + np.exp(-logits))

            j_arr = np.array(judges)
            try:
                pr = pearsonr(probe_score, j_arr)
                r_val = float(pr.statistic)
            except Exception:
                r_val = float("nan")
            try:
                auroc = float(roc_auc_score((j_arr > 0.5).astype(int), probe_score))
            except ValueError:
                auroc = float("nan")  # all labels one class

            v_persona = load_vec(vec_dir, persona, trait, args.layer)
            distance = 1.0 - cos(v_persona, null_vecs[trait])

            rows.append({
                "trait": trait, "persona": persona,
                "n": int(len(j_arr)),
                "x_distance": distance,
                "pearson_r": r_val,
                "abs_r": abs(r_val) if r_val == r_val else float("nan"),
                "auroc": auroc,
                "mean_judge": float(j_arr.mean()),
                "mean_probe": float(probe_score.mean()),
            })

    if not rows:
        print("No (persona, trait) cells produced — check that judged/activations/probes/vectors all line up.")
        return

    (out_dir / "n4_per_cell.json").write_text(json.dumps(rows, indent=2))

    # ----- plot -----
    fig, (ax_r, ax_au) = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("tab10")
    color = {t: cmap(i % 10) for i, t in enumerate(TRAITS)}

    def _panel(ax, ykey, ylabel, ylim=None):
        x = np.array([r["x_distance"] for r in rows])
        y = np.array([r[ykey] for r in rows])
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        for r in rows:
            if not np.isfinite(r[ykey]):
                continue
            ax.scatter(r["x_distance"], r[ykey], color=color[r["trait"]],
                       s=42, alpha=0.85, edgecolor="white", linewidth=0.5)
        if len(x) >= 3:
            r0, p_perm = perm_pearson(x, y)
            coef = np.polyfit(x, y, 1)
            xl = np.linspace(x.min(), x.max(), 100)
            ax.plot(xl, coef[0] * xl + coef[1], "k--", lw=1.4)
            ax.text(0.03, 0.05,
                    f"r = {r0:+.2f}   $p_{{\\mathrm{{perm}}}}$ = {p_perm:.1e}\n"
                    f"slope = {coef[0]:+.2f}   n = {len(x)}",
                    transform=ax.transAxes, fontsize=8, va="bottom")
        ax.set_xlabel(r"$1 - \cos(\mathbf{v}_{T,c},\,\mathbf{v}_{T,\mathrm{null}})$",
                      fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, ls=":")

    _panel(ax_r, "abs_r",
           r"$|r|$  null-probe vs Claude judge", (0, 1))
    _panel(ax_au, "auroc",
           r"AUROC  null-probe vs (judge $> 0.5$)", (0.4, 1))

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="", color=color[t],
                      markersize=6, label=TRAIT_LABEL[t])
               for t in TRAITS if t in {r["trait"] for r in rows}]
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Naturalistic Fig 3: null-probe measurement quality vs distance from null",
                 y=1.08, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "n4_naturalistic_fig3.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "n4_naturalistic_fig3.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir/'n4_naturalistic_fig3.pdf'} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
