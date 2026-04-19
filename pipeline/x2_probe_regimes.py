#!/usr/bin/env python3
"""X2: Linear trait probes — null vs context-specific training regimes.

Per trait, trains probes under three regimes and measures cross-context
generalisation. Used for Fig 2.

  Regime A (null-only): train on null-context activations only,
                        eval on each context's held-out test set
  Regime B (multi-context, leave-one-out): for each held-out context C,
                        train on pooled activations from the other 11
  Regime B-parity:      subsample B's train set to ~A's size for fair compare
  Regime C (full):      12x12 train-context x eval-context AUROC matrix (appendix)

Test split by question matches X1's split if --classifier-splits is given,
so behavioural sensitivity (Fig 1) and probe AUROC (Fig 2) align.

Usage:
    python pipeline/x2_probe_regimes.py \\
        --activations-dir outputs/gemma-2-27b-it/v2/activations \\
        --output-dir outputs/gemma-2-27b-it/v2/probes \\
        --classifier-splits outputs/gemma-2-27b-it/v2/classifier/splits.json \\
        --layer 22
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from persona_steering.config import PERSONA_SLUGS, TARGET_LAYER, Trait
from persona_steering.utils import derive_model_short_from_path, log, save_json
from persona_steering.wandb_utils import (
    finish_run, init_run, log_artifact, log_metrics, log_summary,
)


DEFAULT_CONTEXTS = list(PERSONA_SLUGS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--activations-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--classifier-splits", type=str, default=None,
                   help="splits.json from X1 to align held-out questions")
    p.add_argument("--n-held-out-questions", type=int, default=20)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--contexts", nargs="+", default=DEFAULT_CONTEXTS)
    p.add_argument("--traits", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parity-train-size", type=int, default=160)
    p.add_argument("--skip-regime-c", action="store_true")
    return p.parse_args()


def load_activations(path: Path, layer: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Load .pt activations dict, return (n, dim) at layer + (variant_idx, question_idx)."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    keys, vecs = [], []
    for k, v in data.items():
        try:
            parts = k.split("_")
            vi = int(parts[0][1:])
            qi = int(parts[1][1:])
        except (ValueError, IndexError):
            continue
        a = v[layer].float()
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        vecs.append(a.numpy()); keys.append((vi, qi))
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.stack(vecs).astype(np.float32), keys


def collect(activations_dir: Path, contexts, traits, layer):
    out = {}
    for ctx in contexts:
        for tr in traits:
            pos_path = activations_dir / f"{ctx}_{tr}_pos.pt"
            neg_path = activations_dir / f"{ctx}_{tr}_neg.pt"
            if not (pos_path.exists() and neg_path.exists()):
                log.warning("Missing %s/%s", ctx, tr)
                continue
            X_pos, k_pos = load_activations(pos_path, layer)
            X_neg, k_neg = load_activations(neg_path, layer)
            if len(X_pos) == 0 or len(X_neg) == 0:
                continue
            out[(ctx, tr)] = {"X_pos": X_pos, "keys_pos": k_pos,
                              "X_neg": X_neg, "keys_neg": k_neg}
    return out


def held_out_per_trait(data, n_held_out, seed, classifier_splits):
    if classifier_splits and Path(classifier_splits).exists():
        log.info("Aligning splits with %s", classifier_splits)
        ho = json.loads(Path(classifier_splits).read_text())["held_out"]
        per: dict[str, set[int]] = {}
        for t, q in ho:
            per.setdefault(t, set()).add(int(q))
        return per
    rng = random.Random(seed)
    qs_per_trait: dict[str, set[int]] = {}
    for (_, tr), payload in data.items():
        for vi, qi in payload["keys_pos"] + payload["keys_neg"]:
            qs_per_trait.setdefault(tr, set()).add(qi)
    return {t: set(rng.sample(sorted(qs), min(n_held_out, len(qs))))
            for t, qs in qs_per_trait.items()}


def split_train_test(payload, held_qs):
    X_pos, k_pos = payload["X_pos"], payload["keys_pos"]
    X_neg, k_neg = payload["X_neg"], payload["keys_neg"]

    def split(X, keys, label):
        tr_X, tr_y, te_X, te_y = [], [], [], []
        for x, (_, qi) in zip(X, keys):
            if qi in held_qs:
                te_X.append(x); te_y.append(label)
            else:
                tr_X.append(x); tr_y.append(label)
        return tr_X, tr_y, te_X, te_y

    p_trX, p_try, p_teX, p_tey = split(X_pos, k_pos, 1)
    n_trX, n_try, n_teX, n_tey = split(X_neg, k_neg, 0)
    dim = X_pos.shape[1]
    X_train = (np.array(p_trX + n_trX) if (p_trX or n_trX) else np.zeros((0, dim)))
    y_train = np.array(p_try + n_try)
    X_test = (np.array(p_teX + n_teX) if (p_teX or n_teX) else np.zeros((0, dim)))
    y_test = np.array(p_tey + n_tey)
    return X_train, y_train, X_test, y_test


def fit_probe(X_train, y_train, seed):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    probe = LogisticRegressionCV(
        cv=min(5, max(2, len(set(y_train)))),
        max_iter=1000, class_weight="balanced",
        scoring="roc_auc", random_state=seed,
    )
    probe.fit(X_s, y_train)
    return probe, scaler


def auroc_eval(probe, scaler, X_test, y_test):
    if len(set(y_test)) < 2 or len(X_test) == 0:
        return None
    p = probe.predict_proba(scaler.transform(X_test))[:, 1]
    return float(roc_auc_score(y_test, p))


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    probes_dir = out / "probes_pkl"
    probes_dir.mkdir(exist_ok=True)
    model_short = derive_model_short_from_path(args.activations_dir)
    init_run("x2_probes", model_short, config=vars(args), method="causal-figures")

    contexts = list(args.contexts)
    traits = args.traits or [t.value for t in Trait]

    log.info("Loading activations from %s (layer %d)", args.activations_dir, args.layer)
    data = collect(Path(args.activations_dir), contexts, traits, args.layer)
    if not data:
        log.error("No activations loaded.")
        return

    held_qs_per_trait = held_out_per_trait(
        data, args.n_held_out_questions, args.seed, args.classifier_splits
    )

    results = {}
    for trait in traits:
        log.info("=== %s ===", trait)
        held_qs = held_qs_per_trait.get(trait, set())
        results[trait] = {"A": {}, "B": {}, "B_parity": {}, "C": {}}

        per_ctx_splits = {}
        for ctx in contexts:
            payload = data.get((ctx, trait))
            if payload is None:
                continue
            per_ctx_splits[ctx] = split_train_test(payload, held_qs)

        # --- Regime A: null only ---
        if "null" in per_ctx_splits:
            X_tr, y_tr, _, _ = per_ctx_splits["null"]
            if len(X_tr) >= 4 and len(set(y_tr)) == 2:
                probe_A, scaler_A = fit_probe(X_tr, y_tr, args.seed)
                with open(probes_dir / f"{trait}_A_null.pkl", "wb") as f:
                    pickle.dump({"probe": probe_A, "scaler": scaler_A,
                                 "trained_on": "null"}, f)
                for ctx, (_, _, X_te, y_te) in per_ctx_splits.items():
                    results[trait]["A"][ctx] = auroc_eval(probe_A, scaler_A, X_te, y_te)

        # --- Regime B: leave-one-out ---
        for held_ctx in contexts:
            train_X, train_y = [], []
            for ctx, (X_tr, y_tr, _, _) in per_ctx_splits.items():
                if ctx == held_ctx:
                    continue
                train_X.append(X_tr); train_y.append(y_tr)
            if not train_X or held_ctx not in per_ctx_splits:
                continue
            X_pool = np.vstack(train_X)
            y_pool = np.concatenate(train_y)
            if len(X_pool) < 4 or len(set(y_pool)) < 2:
                continue
            probe_B, scaler_B = fit_probe(X_pool, y_pool, args.seed)
            X_te, y_te = per_ctx_splits[held_ctx][2], per_ctx_splits[held_ctx][3]
            results[trait]["B"][held_ctx] = auroc_eval(probe_B, scaler_B, X_te, y_te)

            # Parity control
            if len(X_pool) > args.parity_train_size:
                rng = np.random.default_rng(args.seed)
                idx = rng.choice(len(X_pool), args.parity_train_size, replace=False)
                if len(set(y_pool[idx])) == 2:
                    probe_Bp, scaler_Bp = fit_probe(X_pool[idx], y_pool[idx], args.seed)
                    results[trait]["B_parity"][held_ctx] = auroc_eval(
                        probe_Bp, scaler_Bp, X_te, y_te
                    )

        # --- Regime C: full matrix ---
        if not args.skip_regime_c:
            ctx_idx = {c: i for i, c in enumerate(sorted(per_ctx_splits.keys()))}
            n = len(ctx_idx)
            mat = np.full((n, n), np.nan)
            for tr_ctx, (X_tr, y_tr, _, _) in per_ctx_splits.items():
                if len(X_tr) < 4 or len(set(y_tr)) < 2:
                    continue
                probe_C, scaler_C = fit_probe(X_tr, y_tr, args.seed)
                results[trait]["C"][tr_ctx] = {}
                for te_ctx, (_, _, X_te, y_te) in per_ctx_splits.items():
                    a = auroc_eval(probe_C, scaler_C, X_te, y_te)
                    results[trait]["C"][tr_ctx][te_ctx] = a
                    if a is not None:
                        mat[ctx_idx[tr_ctx], ctx_idx[te_ctx]] = a
            np.save(out / f"auroc_matrix_{trait}.npy", mat)
            save_json({"contexts": sorted(per_ctx_splits.keys())},
                      out / f"auroc_matrix_{trait}_contexts.json")

    save_json({
        "results": results,
        "config": vars(args),
        "contexts": contexts,
        "traits": traits,
    }, out / "metrics.json")
    save_json({
        "held_out_per_trait": {t: sorted(qs) for t, qs in held_qs_per_trait.items()},
        "seed": args.seed,
    }, out / "splits.json")

    summary_metrics: dict[str, float] = {}
    for trait in traits:
        if trait not in results:
            continue
        a = [v for v in results[trait]["A"].values() if v is not None]
        b = [v for v in results[trait]["B"].values() if v is not None]
        bp = [v for v in results[trait]["B_parity"].values() if v is not None]
        a_mean = float(np.mean(a)) if a else float("nan")
        b_mean = float(np.mean(b)) if b else float("nan")
        bp_mean = float(np.mean(bp)) if bp else float("nan")
        log.info("%s  A_mean=%.3f  B_mean=%.3f  Bparity_mean=%.3f",
                 trait, a_mean, b_mean, bp_mean)
        summary_metrics[f"trait/{trait}/A_mean"] = a_mean
        summary_metrics[f"trait/{trait}/B_mean"] = b_mean
        summary_metrics[f"trait/{trait}/Bparity_mean"] = bp_mean

    log_summary(summary_metrics)
    log_artifact(f"{model_short}-x2-probes", "probes", out, glob_pattern="**/*")
    finish_run()


if __name__ == "__main__":
    main()
