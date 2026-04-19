#!/usr/bin/env python3
"""X3b: Extract context directions u_C from neutral-response activations,
then orthogonalize per trait against the null-context CAA trait vector.

For each context C in {10 personas}:
    u_C = mean(activations | C, neutral prompts) - mean(activations | null)

For each (C, trait T):
    v_T_null = pre-existing null-context trait vector (from pipeline/3_vectors.py)
    u_C_T_orth = u_C - <u_C, v_T_null>/||v_T_null||^2 * v_T_null

If ||u_C_T_orth|| / ||u_C|| < 0.5 the (C, T) pair is flagged as
"trait-entangled" — kept but reported separately downstream.

Assumes activations were extracted by pipeline/2_activations.py
(or an equivalent that produces {variant_question -> (n_layers, hidden_dim)} .pt).

Usage:
    # Step A: extract activations from neutral responses (reuse pipeline/2)
    python pipeline/2_activations.py \\
        --model google/gemma-2-27b-it \\
        --responses-dir outputs/gemma-2-27b-it/v2/neutral_responses \\
        --output-dir outputs/gemma-2-27b-it/v2/neutral_activations

    # Step B: this script
    python pipeline/x3b_context_directions.py \\
        --neutral-activations-dir outputs/gemma-2-27b-it/v2/neutral_activations \\
        --vectors-dir outputs/gemma-2-27b-it/v2/vectors \\
        --output-dir outputs/gemma-2-27b-it/v2/causal/directions \\
        --layer 22
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from persona_steering.config import PERSONA_SLUGS, TARGET_LAYER, Trait
from persona_steering.utils import log, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--neutral-activations-dir", type=str, required=True,
                   help="Dir of {context}.pt files (one per persona)")
    p.add_argument("--vectors-dir", type=str, required=True,
                   help="Dir of CAA/IV trait vectors {persona}_{trait}.pt; "
                        "we use null_{trait}.pt as the trait basis for ortho")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--null-context", type=str, default="null")
    p.add_argument("--contexts", nargs="+", default=None,
                   help="Personas to compute u_C for (default: PERSONA_SLUGS minus null/nonsense)")
    p.add_argument("--traits", nargs="+", default=None)
    p.add_argument("--entanglement-threshold", type=float, default=0.5,
                   help="Flag (C,T) if ||u_orth||/||u_C|| < threshold")
    return p.parse_args()


def load_layer_mean(pt_path: Path, layer: int) -> torch.Tensor:
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    vecs = []
    for v in data.values():
        a = v[layer].float()
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        vecs.append(a)
    if not vecs:
        raise ValueError(f"No vectors in {pt_path}")
    return torch.stack(vecs).mean(0)


def load_trait_vector(vectors_dir: Path, persona: str, trait: str, layer: int) -> torch.Tensor | None:
    path = vectors_dir / f"{persona}_{trait}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    full = data["vector"].float()
    if layer >= full.shape[0]:
        return None
    return full[layer]


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    contexts = args.contexts or [c for c in PERSONA_SLUGS if c not in {"null", "nonsense"}]
    traits = args.traits or [t.value for t in Trait]
    act_dir = Path(args.neutral_activations_dir)
    vec_dir = Path(args.vectors_dir)

    null_path = act_dir / f"{args.null_context}.pt"
    if not null_path.exists():
        log.error("Null activations missing: %s", null_path)
        return
    log.info("Loading null mean activation from %s", null_path)
    mean_null = load_layer_mean(null_path, args.layer)

    summary = {"contexts": contexts, "traits": traits, "layer": args.layer,
               "entangled_pairs": [], "norms": {}}

    for ctx in contexts:
        ctx_path = act_dir / f"{ctx}.pt"
        if not ctx_path.exists():
            log.warning("Missing context activations: %s", ctx_path)
            continue
        mean_ctx = load_layer_mean(ctx_path, args.layer)
        u_C = (mean_ctx - mean_null).contiguous()
        norm_uC = float(u_C.norm())

        torch.save({"vector": u_C, "context": ctx, "layer": args.layer,
                    "norm": norm_uC},
                   out / f"u_{ctx}.pt")
        log.info("u_%s norm=%.4f", ctx, norm_uC)

        for trait in traits:
            v_T_null = load_trait_vector(vec_dir, args.null_context, trait, args.layer)
            if v_T_null is None:
                log.warning("No null trait vector for %s — skipping ortho for (%s,%s)",
                            trait, ctx, trait)
                continue
            v_norm_sq = (v_T_null * v_T_null).sum().clamp_min(1e-12)
            proj_coef = (u_C * v_T_null).sum() / v_norm_sq
            u_orth = u_C - proj_coef * v_T_null
            ratio = float(u_orth.norm() / max(norm_uC, 1e-12))
            entangled = ratio < args.entanglement_threshold
            if entangled:
                summary["entangled_pairs"].append([ctx, trait, ratio])
                log.warning("Entangled: (%s, %s) ratio=%.3f", ctx, trait, ratio)

            torch.save({
                "vector": u_orth.contiguous(),
                "context": ctx, "trait": trait,
                "layer": args.layer,
                "raw_context_direction_norm": norm_uC,
                "orth_norm": float(u_orth.norm()),
                "ratio_orth_over_raw": ratio,
                "entangled": entangled,
                "trait_basis_persona": args.null_context,
            }, out / f"u_{ctx}_{trait}_orth.pt")
            summary["norms"][f"{ctx}_{trait}"] = {
                "raw": norm_uC, "orth": float(u_orth.norm()), "ratio": ratio,
            }

    save_json(summary, out / "directions_summary.json")
    log.info("Saved %d context directions and orthogonalised variants to %s",
             len(contexts), out)
    log.info("Entangled pairs flagged: %d", len(summary["entangled_pairs"]))


if __name__ == "__main__":
    main()
