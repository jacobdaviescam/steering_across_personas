#!/usr/bin/env python3
"""E4: Probe transfer across contexts.

Trains a logistic regression probe to detect trait presence (pos vs neg)
within each context, then evaluates cross-context transfer.

This is the key safety experiment: if a probe trained in one context fails
in another, safety monitors built on single-context representations are unreliable.

Usage:
    python pipeline/e4_probe_transfer.py --activations-dir outputs/gemma-2-27b-it/activations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from persona_steering.config import Trait, TARGET_LAYER, PERSONA_SLUGS
from persona_steering.utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe transfer across contexts")
    parser.add_argument("--activations-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def discover_pairs(activations_dir: Path) -> dict[str, dict[str, tuple[Path, Path]]]:
    """Find pos/neg activation file pairs."""
    trait_values = {t.value for t in Trait}
    files = {f.stem: f for f in activations_dir.glob("*.pt")}

    pairs: dict[str, dict[str, tuple[Path, Path]]] = {}
    seen = set()

    for stem in files:
        for direction in ("_pos", "_neg"):
            if not stem.endswith(direction):
                continue
            base = stem[: -len(direction)]

            persona_slug = None
            trait_name = None
            for tv in trait_values:
                if base.endswith(f"_{tv}"):
                    persona_slug = base[: -(len(tv) + 1)]
                    trait_name = tv
                    break

            if persona_slug is None or trait_name is None or base in seen:
                continue
            seen.add(base)

            pos_path = activations_dir / f"{base}_pos.pt"
            neg_path = activations_dir / f"{base}_neg.pt"
            if pos_path.exists() and neg_path.exists():
                pairs.setdefault(persona_slug, {})[trait_name] = (pos_path, neg_path)

    return pairs


def load_activations(path: Path, layer: int) -> np.ndarray:
    """Load activation file and return (n_samples, hidden_dim) numpy array."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    _clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    vecs = [_clean(v[layer].float()).numpy() for v in data.values()]
    return np.stack(vecs)


def main() -> None:
    args = parse_args()
    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir) if args.output_dir else activations_dir.parent / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    layer = args.layer
    seed = args.seed

    pairs = discover_pairs(activations_dir)
    personas = sorted(pairs.keys())
    traits = sorted({t for p in pairs.values() for t in p.keys()})

    log.info("Found %d personas, %d traits", len(personas), len(traits))

    # Pre-load all activations
    log.info("Loading activation files...")
    act_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for persona in personas:
        act_data[persona] = {}
        for trait in traits:
            if trait not in pairs[persona]:
                continue
            pos_path, neg_path = pairs[persona][trait]
            pos_acts = load_activations(pos_path, layer)
            neg_acts = load_activations(neg_path, layer)
            act_data[persona][trait] = (pos_acts, neg_acts)
    log.info("Loaded all activations")

    results = {}

    for trait in traits:
        log.info("Processing trait: %s", trait)
        trait_personas = [p for p in personas if trait in act_data[p]]
        n = len(trait_personas)

        # Build per-context datasets
        context_data = {}
        for persona in trait_personas:
            pos, neg = act_data[persona][trait]
            X = np.vstack([pos, neg])
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            context_data[persona] = (X, y)

        # Train per-context probes and evaluate cross-context
        accuracy_matrix = np.zeros((n, n))  # (train_ctx, test_ctx)
        cv_scores = {}

        for i, train_persona in enumerate(trait_personas):
            X_train, y_train = context_data[train_persona]
            probe = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            probe.fit(X_train, y_train)

            # Cross-validation score on own data
            cv = cross_val_score(
                LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
                X_train, y_train, cv=5, scoring="accuracy",
            )
            cv_scores[train_persona] = float(np.mean(cv))

            for j, test_persona in enumerate(trait_personas):
                X_test, y_test = context_data[test_persona]
                y_pred = probe.predict(X_test)
                accuracy_matrix[i, j] = accuracy_score(y_test, y_pred)

        # Train universal probe (all contexts pooled)
        X_all = np.vstack([context_data[p][0] for p in trait_personas])
        y_all = np.concatenate([context_data[p][1] for p in trait_personas])
        universal_probe = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        universal_probe.fit(X_all, y_all)

        universal_per_context = {}
        for persona in trait_personas:
            X_test, y_test = context_data[persona]
            y_pred = universal_probe.predict(X_test)
            universal_per_context[persona] = float(accuracy_score(y_test, y_pred))

        universal_cv = float(np.mean(cross_val_score(
            LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
            X_all, y_all, cv=5, scoring="accuracy",
        )))

        # Compute summary metrics
        self_acc = float(np.mean(np.diag(accuracy_matrix)))
        cross_acc = float(np.mean(accuracy_matrix[~np.eye(n, dtype=bool)]))
        transfer_gap = self_acc - cross_acc

        results[trait] = {
            "personas": trait_personas,
            "accuracy_matrix": accuracy_matrix.tolist(),
            "cv_scores": cv_scores,
            "self_accuracy": self_acc,
            "cross_accuracy": cross_acc,
            "transfer_gap": transfer_gap,
            "universal_probe": {
                "cv_accuracy": universal_cv,
                "per_context": universal_per_context,
                "mean_per_context": float(np.mean(list(universal_per_context.values()))),
            },
        }

    # Print summary
    print(f"\n{'Trait':<16} {'Self':>8} {'Cross':>8} {'Gap':>8} {'Universal':>10}")
    print("-" * 56)
    for trait in traits:
        r = results[trait]
        print(
            f"{trait:<16} "
            f"{r['self_accuracy']:>8.3f} "
            f"{r['cross_accuracy']:>8.3f} "
            f"{r['transfer_gap']:>+8.3f} "
            f"{r['universal_probe']['cv_accuracy']:>10.3f}"
        )

    # Save
    output_path = output_dir / "probe_transfer.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Generate figure: transfer matrices
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_traits = len(traits)
    cols = 4
    rows = (n_traits + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten() if n_traits > 1 else [axes]

    for idx, trait in enumerate(traits):
        ax = axes[idx]
        r = results[trait]
        matrix = np.array(r["accuracy_matrix"])
        personas_short = [p[:8] for p in r["personas"]]

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(personas_short)))
        ax.set_xticklabels(personas_short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(personas_short)))
        ax.set_yticklabels(personas_short, fontsize=7)
        ax.set_title(f"{trait.replace('_', ' ')}\ngap={r['transfer_gap']:+.3f}", fontsize=9)
        ax.set_xlabel("Test context", fontsize=8)
        ax.set_ylabel("Train context", fontsize=8)

    # Remove empty subplots
    for idx in range(len(traits), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Probe Transfer Matrices (Train Row → Test Col)", fontsize=13, y=1.02)
    fig.tight_layout()

    fig_path = output_dir / "probe_transfer.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
