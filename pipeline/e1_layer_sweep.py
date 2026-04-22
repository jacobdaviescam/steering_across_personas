#!/usr/bin/env python3
"""E1: Layer-sweep shared variance profile.

Collates decomposition results across layers to show how context dependence
of trait representations varies through the network.

Reads existing decomposition.json files from analysis_layer_* directories
(produced by pipeline step 4 at multiple layers).

Usage:
    python pipeline/e1_layer_sweep.py --model-dir outputs/gemma-2-27b-it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-sweep shared variance analysis")
    parser.add_argument(
        "--model-dir", type=str, default="outputs/gemma-2-27b-it",
        help="Model output directory containing analysis_layer_* dirs",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: {model-dir}/experiments)",
    )
    return parser.parse_args()


def load_decomposition(path: Path) -> dict[str, float]:
    """Load decomposition.json and return trait -> variance_explained mapping."""
    with open(path) as f:
        data = json.load(f)
    return {trait: info["variance_explained"] for trait, info in data.items()}


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect layer -> trait -> rho_t from analysis_layer_* dirs
    results: dict[int, dict[str, float]] = {}

    # Main analysis dir (layer 22 by default)
    main_decomp = model_dir / "analysis" / "decomposition.json"
    if main_decomp.exists():
        results[22] = load_decomposition(main_decomp)

    # Layer-specific dirs
    for d in sorted(model_dir.glob("analysis_layer_*")):
        layer_str = d.name.replace("analysis_layer_", "")
        try:
            layer = int(layer_str)
        except ValueError:
            continue
        decomp_file = d / "decomposition.json"
        if decomp_file.exists():
            results[layer] = load_decomposition(decomp_file)

    # Also load CAA at layer 22 if available
    caa_results: dict[int, dict[str, float]] = {}
    caa_decomp = model_dir / "caa_analysis" / "decomposition.json"
    if caa_decomp.exists():
        caa_results[22] = load_decomposition(caa_decomp)

    if not results:
        print("ERROR: No decomposition files found")
        return

    layers = sorted(results.keys())
    traits = sorted(results[layers[0]].keys())

    print(f"Found {len(layers)} layers: {layers}")
    print(f"Traits: {traits}")
    print()

    # Build output data
    output = {
        "iv": {
            "layers": layers,
            "traits": {t: [results[l][t] for l in layers] for t in traits},
        },
    }
    if caa_results:
        output["caa"] = {
            "layers": list(caa_results.keys()),
            "traits": {t: [caa_results[l][t] for l in caa_results] for t in traits},
        }

    # Print summary table
    print(f"{'Layer':>6}", end="")
    for t in traits:
        print(f"  {t[:8]:>8}", end="")
    print(f"  {'mean':>8}")
    print("-" * (6 + (len(traits) + 1) * 10))

    for l in layers:
        vals = [results[l][t] for t in traits]
        print(f"{l:>6}", end="")
        for v in vals:
            print(f"  {v:>8.3f}", end="")
        print(f"  {np.mean(vals):>8.3f}")

    if caa_results:
        print("\nCAA at layer 22:")
        for t in traits:
            print(f"  {t}: {caa_results[22][t]:.3f} (IV: {results[22][t]:.3f}, gap: {results[22][t] - caa_results[22][t]:.3f})")

    # Save JSON
    output_path = output_dir / "layer_sweep.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Generate figure
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(traits)))
    for i, t in enumerate(traits):
        vals = [results[l][t] for l in layers]
        ax.plot(layers, vals, "o-", color=colors[i], label=t.replace("_", " "), linewidth=2, markersize=5)

        # Add CAA point at layer 22 if available
        if caa_results and t in caa_results.get(22, {}):
            ax.plot(22, caa_results[22][t], "x", color=colors[i], markersize=10, markeredgewidth=2)

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Shared Variance Ratio (ρ)", fontsize=13)
    ax.set_title("Context Dependence Across Layers", fontsize=14)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="_")
    ax.grid(True, alpha=0.3)

    if caa_results:
        # Add annotation for CAA points
        ax.annotate("× = CAA (layer 22)", xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", fontsize=9, color="gray")

    fig_path = output_dir / "layer_sweep.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
