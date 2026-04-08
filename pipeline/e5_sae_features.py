#!/usr/bin/env python3
"""E5: SAE feature divergence across contexts.

Loads a Gemma Scope SAE (JumpReLU, 131K features, layer 22) and encodes
trait activations to identify which SAE features correspond to each trait
in each context. Compares feature sets across contexts.

If the same SAE features fire for a trait regardless of context, the trait
representation is context-independent at the feature level. If different
features fire, the model uses genuinely different internal mechanisms.

Usage:
    python pipeline/e5_sae_features.py \
        --activations-dir outputs/gemma-2-27b-it/activations \
        --sae-repo google/gemma-scope-27b-pt-res \
        --sae-folder layer_22/width_131k/average_l0_82

    # Or with a pre-downloaded params.npz:
    python pipeline/e5_sae_features.py \
        --activations-dir outputs/gemma-2-27b-it/activations \
        --sae-path /path/to/params.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from persona_steering.config import Trait, TARGET_LAYER, PERSONA_SLUGS
from persona_steering.utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAE feature divergence analysis")
    parser.add_argument("--activations-dir", type=str, required=True)
    parser.add_argument("--sae-path", type=str, default=None,
                        help="Path to pre-downloaded params.npz")
    parser.add_argument("--sae-repo", type=str, default="google/gemma-scope-27b-pt-res",
                        help="HuggingFace repo ID for Gemma Scope SAE")
    parser.add_argument("--sae-folder", type=str,
                        default="layer_22/width_131k/average_l0_82",
                        help="Subfolder within the HF repo")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of top features to consider per trait x context")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


class JumpReLUSAE:
    """Minimal JumpReLU SAE for encoding activations into sparse features.

    Architecture: features = z * (z > threshold)
    where z = W_enc @ x + b_enc
    Decode: x_hat = W_dec @ features + b_dec
    """

    def __init__(self, params_path: str, device: str = "cpu"):
        log.info("Loading SAE from %s", params_path)
        with np.load(params_path) as data:
            keys = list(data.keys())
            log.info("SAE keys: %s", keys)

            self.W_enc = torch.tensor(data["w_enc"], dtype=torch.float32, device=device)
            self.W_dec = torch.tensor(data["w_dec"], dtype=torch.float32, device=device)
            self.b_enc = torch.tensor(data["b_enc"], dtype=torch.float32, device=device)
            self.b_dec = torch.tensor(data["b_dec"], dtype=torch.float32, device=device)

            # JumpReLU threshold (may be stored as "threshold" or absent)
            if "threshold" in data:
                self.threshold = torch.tensor(
                    data["threshold"], dtype=torch.float32, device=device
                )
            else:
                # Default to 0 (standard ReLU)
                self.threshold = torch.zeros(
                    self.W_enc.shape[0], dtype=torch.float32, device=device
                )

        self.d_in = self.W_enc.shape[1]
        self.d_sae = self.W_enc.shape[0]
        log.info("SAE: d_in=%d, d_sae=%d", self.d_in, self.d_sae)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse feature activations.

        Args:
            x: (batch, d_in) or (d_in,)

        Returns:
            features: same leading dims, (d_sae,)
        """
        z = x @ self.W_enc.T + self.b_enc  # (..., d_sae)
        # JumpReLU: z * (z > threshold)
        features = z * (z > self.threshold).float()
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        return features @ self.W_dec + self.b_dec

    def reconstruction_error(self, x: torch.Tensor) -> float:
        """Mean squared reconstruction error."""
        features = self.encode(x)
        x_hat = self.decode(features)
        return ((x - x_hat) ** 2).mean().item()


def discover_pairs(activations_dir: Path) -> dict[str, dict[str, tuple[Path, Path]]]:
    """Find pos/neg activation file pairs."""
    trait_values = {t.value for t in Trait}
    pairs: dict[str, dict[str, tuple[Path, Path]]] = {}
    seen = set()

    for f in sorted(activations_dir.glob("*.pt")):
        stem = f.stem
        for direction in ("_pos", "_neg"):
            if not stem.endswith(direction):
                continue
            base = stem[: -len(direction)]
            for tv in trait_values:
                if base.endswith(f"_{tv}"):
                    persona = base[: -(len(tv) + 1)]
                    if base not in seen:
                        seen.add(base)
                        pos_path = activations_dir / f"{base}_pos.pt"
                        neg_path = activations_dir / f"{base}_neg.pt"
                        if pos_path.exists() and neg_path.exists():
                            pairs.setdefault(persona, {})[tv] = (pos_path, neg_path)
                    break

    return pairs


def load_mean_activation(path: Path, layer: int) -> torch.Tensor:
    """Load activation file and return mean activation at given layer."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    _clean = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    vecs = [_clean(v[layer].float()) for v in data.values()]
    return torch.stack(vecs).mean(dim=0)


def main() -> None:
    args = parse_args()
    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir) if args.output_dir else activations_dir.parent / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = args.layer
    top_k = args.top_k

    # Load SAE
    if args.sae_path:
        sae_path = args.sae_path
    else:
        from huggingface_hub import hf_hub_download
        log.info("Downloading SAE from %s/%s", args.sae_repo, args.sae_folder)
        sae_path = hf_hub_download(
            repo_id=args.sae_repo,
            filename="params.npz",
            subfolder=args.sae_folder,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE(sae_path, device=device)

    # Discover activation pairs
    pairs = discover_pairs(activations_dir)
    personas = sorted(pairs.keys())
    traits = sorted({t for p in pairs.values() for t in p.keys()})
    log.info("Found %d personas, %d traits", len(personas), len(traits))

    # Compute reconstruction error on a sample
    sample_pos_path = list(list(pairs.values())[0].values())[0][0]
    sample_act = load_mean_activation(sample_pos_path, layer).to(device)
    recon_err = sae.reconstruction_error(sample_act.unsqueeze(0))
    log.info("Sample reconstruction MSE: %.4f (relative: %.4f)",
             recon_err, recon_err / (sample_act ** 2).mean().item())

    results = {}

    for trait in traits:
        log.info("Processing trait: %s", trait)
        trait_personas = [p for p in personas if trait in pairs[p]]

        # For each context, compute trait-discriminative features
        # = features with largest |activation_pos - activation_neg|
        context_feature_sets = {}  # persona -> set of top-K feature indices
        context_feature_scores = {}  # persona -> array of per-feature scores

        for persona in trait_personas:
            pos_path, neg_path = pairs[persona][trait]
            pos_mean = load_mean_activation(pos_path, layer).to(device)
            neg_mean = load_mean_activation(neg_path, layer).to(device)

            pos_features = sae.encode(pos_mean.unsqueeze(0)).squeeze(0)
            neg_features = sae.encode(neg_mean.unsqueeze(0)).squeeze(0)

            # Feature importance: absolute difference in activation
            diff = (pos_features - neg_features).abs().cpu().numpy()
            context_feature_scores[persona] = diff

            # Top-K features
            top_indices = np.argsort(diff)[-top_k:]
            context_feature_sets[persona] = set(top_indices.tolist())

        # Compute Jaccard similarity matrix
        n = len(trait_personas)
        jaccard_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                si = context_feature_sets[trait_personas[i]]
                sj = context_feature_sets[trait_personas[j]]
                jaccard_matrix[i, j] = len(si & sj) / len(si | sj) if si | sj else 0

        # Shared features: present in ALL contexts
        all_sets = [context_feature_sets[p] for p in trait_personas]
        shared_features = set.intersection(*all_sets) if all_sets else set()

        # Features unique to each context
        unique_per_context = {}
        for p in trait_personas:
            others = set.union(*[context_feature_sets[q] for q in trait_personas if q != p])
            unique_per_context[p] = len(context_feature_sets[p] - others)

        results[trait] = {
            "personas": trait_personas,
            "jaccard_matrix": jaccard_matrix.tolist(),
            "mean_jaccard": float(jaccard_matrix[~np.eye(n, dtype=bool)].mean()),
            "min_jaccard": float(jaccard_matrix[~np.eye(n, dtype=bool)].min()),
            "n_shared_all": len(shared_features),
            "shared_features": sorted(shared_features)[:50],  # save up to 50
            "unique_per_context": unique_per_context,
            "mean_unique": float(np.mean(list(unique_per_context.values()))),
            "top_k": top_k,
        }

    # Print summary
    print(f"\n{'Trait':<16} {'Jaccard':>8} {'Shared':>8} {'Unique':>8}")
    print("-" * 46)
    for trait in traits:
        r = results[trait]
        print(
            f"{trait:<16} "
            f"{r['mean_jaccard']:>8.3f} "
            f"{r['n_shared_all']:>8d} "
            f"{r['mean_unique']:>8.1f}"
        )

    print(f"\nReconstruction MSE: {recon_err:.4f}")
    print(f"Top-K: {top_k}")
    print(f"\nInterpretation:")
    print(f"  Jaccard: fraction of top-{top_k} features shared between context pairs")
    print(f"  Shared: features in ALL contexts' top-{top_k}")
    print(f"  Unique: mean features per context not in any other context's top-{top_k}")

    # Save
    results["_meta"] = {
        "sae_repo": args.sae_repo,
        "sae_folder": args.sae_folder,
        "layer": layer,
        "top_k": top_k,
        "reconstruction_mse": recon_err,
        "device": device,
    }
    output_path = output_dir / "sae_features.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Generate Jaccard heatmaps
    n_traits = len(traits)
    cols = 4
    rows = (n_traits + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten() if n_traits > 1 else [axes]

    for idx, trait in enumerate(traits):
        ax = axes[idx]
        r = results[trait]
        matrix = np.array(r["jaccard_matrix"])
        personas_short = [p[:8] for p in r["personas"]]

        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(personas_short)))
        ax.set_xticklabels(personas_short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(personas_short)))
        ax.set_yticklabels(personas_short, fontsize=7)
        ax.set_title(f"{trait.replace('_', ' ')}\nJ={r['mean_jaccard']:.2f}", fontsize=9)

    for idx in range(len(traits), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"SAE Feature Overlap (Jaccard, top-{top_k})", fontsize=13, y=1.02)
    fig.tight_layout()

    fig_path = output_dir / "sae_features.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
