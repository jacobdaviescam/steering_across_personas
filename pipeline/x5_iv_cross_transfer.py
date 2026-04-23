#!/usr/bin/env python3
"""X5: cross-context probe transfer on IV responses.

Loads each within-context CAA probe, evaluates it on every other context's
IV activations. Produces a 12x12 train_ctx x eval_ctx AUROC matrix per trait.

Answers: does a farmer-trained probe detect honest-vs-dishonest on politician
IV responses? (Off-diagonal cells.)

Feeds into x6 (correlation analysis).

Usage:
    python pipeline/x5_iv_cross_transfer.py \
        --iv-activations-dir outputs/gemma-2-27b-it/v2/activations \
        --probes-dir outputs/gemma-2-27b-it/v2/caa_probes/probes_pkl \
        --output-dir outputs/gemma-2-27b-it/v2/caa_probes \
        --layer 22
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

from persona_steering.config import PERSONA_SLUGS, Trait
from persona_steering.utils import derive_model_short_from_path
from persona_steering.wandb_utils import (
    finish_run, init_run, log_images, log_metrics, log_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--iv-activations-dir", type=str, required=True)
    p.add_argument("--probes-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--contexts", nargs="+", default=list(PERSONA_SLUGS))
    p.add_argument("--traits", nargs="+", default=None)
    return p.parse_args()


def load_iv_activations(act_dir: Path, ctx: str, trait: str, layer: int):
    """Return X (n_samples, hidden), y (0/1) for one context+trait."""
    X_list, y_list = [], []
    for direction, label in [("pos", 1), ("neg", 0)]:
        path = act_dir / f"{ctx}_{trait}_{direction}.pt"
        if not path.exists():
            return None, None
        data = torch.load(path, map_location="cpu", weights_only=True)
        for _, tensor in data.items():
            vec = tensor[layer].float().numpy()
            vec = np.nan_to_num(vec)
            X_list.append(vec)
            y_list.append(label)
    return np.array(X_list), np.array(y_list)


def main() -> None:
    args = parse_args()
    act_dir = Path(args.iv_activations_dir)
    probes_dir = Path(args.probes_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    traits = args.traits or [t.value for t in Trait]
    contexts = sorted(args.contexts)

    model_short = derive_model_short_from_path(act_dir)
    init_run("x5_iv_cross_transfer", model_short, config=vars(args), method="caa")

    iv_cache = {}
    for ctx in contexts:
        for trait in traits:
            X, y = load_iv_activations(act_dir, ctx, trait, args.layer)
            if X is not None:
                iv_cache[(ctx, trait)] = (X, y)

    all_results = {}
    for trait in traits:
        print(f"\n=== {trait} ===")
        mat = np.full((len(contexts), len(contexts)), np.nan)
        cell_details = {}

        for i, train_ctx in enumerate(contexts):
            pkl_path = probes_dir / f"{trait}_within_{train_ctx}.pkl"
            if not pkl_path.exists():
                print(f"  missing probe: {pkl_path.name}")
                continue
            with open(pkl_path, "rb") as f:
                bundle = pickle.load(f)
            probe = bundle["probe"]
            scaler = bundle["scaler"]

            for j, eval_ctx in enumerate(contexts):
                if (eval_ctx, trait) not in iv_cache:
                    continue
                X, y = iv_cache[(eval_ctx, trait)]
                if len(set(y)) < 2:
                    continue
                # Use decision_function to avoid sklearn version mismatch in
                # predict_proba; roc_auc_score accepts any monotonic score.
                scores = probe.decision_function(scaler.transform(X))
                auroc = float(roc_auc_score(y, scores))
                mat[i, j] = auroc
                cell_details[f"{train_ctx}->{eval_ctx}"] = auroc

            row = mat[i]
            diag = mat[i, i]
            offd = np.nanmean([mat[i, j] for j in range(len(contexts)) if j != i])
            print(f"  train={train_ctx:22s}  diag={diag:.3f}  mean_off_diag={offd:.3f}")

        np.save(out / f"iv_cross_transfer_{trait}.npy", mat)
        with open(out / f"iv_cross_transfer_{trait}_contexts.json", "w") as f:
            json.dump({"contexts": contexts, "cells": cell_details}, f, indent=2)
        mean_diag = float(np.nanmean(np.diag(mat)))
        mean_off_diag = float(np.nanmean(mat[~np.eye(len(contexts), dtype=bool)]))
        all_results[trait] = {"mean_diag": mean_diag, "mean_off_diag": mean_off_diag}
        log_metrics({
            f"trait/{trait}/mean_diag": mean_diag,
            f"trait/{trait}/mean_off_diag": mean_off_diag,
        })

        # --- per-trait heatmap ---
        fig, ax = plt.subplots(figsize=(9, 7.5))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(contexts)))
        ax.set_yticks(range(len(contexts)))
        ax.set_xticklabels(contexts, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(contexts, fontsize=8)
        ax.set_xlabel("Eval context (IV responses)")
        ax.set_ylabel("Train context (CAA probe)")
        ax.set_title(f"Probe transfer — {trait}\n"
                     f"within={mean_diag:.3f}  cross={mean_off_diag:.3f}  "
                     f"drop={mean_diag - mean_off_diag:+.3f}")
        for i in range(len(contexts)):
            for j in range(len(contexts)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="black")
        fig.colorbar(im, ax=ax, label="AUROC")
        fig.tight_layout()
        fig.savefig(out / f"iv_cross_transfer_{trait}.png", dpi=120)
        plt.close(fig)

    with open(out / "iv_cross_transfer_summary.json", "w") as f:
        json.dump({"contexts": contexts, "per_trait": all_results}, f, indent=2)

    log_summary({
        "overall/mean_diag": float(np.nanmean([r["mean_diag"] for r in all_results.values()])),
        "overall/mean_off_diag": float(np.nanmean([r["mean_off_diag"] for r in all_results.values()])),
    })
    log_images(out, prefix="x5_iv_cross_transfer")
    finish_run()

    print("\n=== SUMMARY ===")
    for trait, r in all_results.items():
        drop = r["mean_diag"] - r["mean_off_diag"]
        print(f"  {trait:15s}  diag={r['mean_diag']:.3f}  "
              f"off_diag={r['mean_off_diag']:.3f}  drop={drop:+.3f}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
