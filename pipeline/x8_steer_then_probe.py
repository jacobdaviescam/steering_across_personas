#!/usr/bin/env python3
"""X8: Steer-then-probe — does steering along direction d raise the
projection of the resulting activation onto d?

For each (context C, trait T) pair we run three single-direction sweeps
(α·u_C, α·v_T, α·random_matched_norm) and a 2D mix grid over
(α_ctx, α_trait).  No generation — we run one forward pass per prompt,
under ActivationSteering at layer L_steer, and read the activation at
the assistant-turn-start position at several downstream layers.

The "probe" is a cosine/dot projection onto u_C / v_T.  We also report
AUROC of steered-vs-unsteered projections as a binary-classification
score — that's what a mean-difference probe achieves.

Usage (single-GPU, no generation):
    python pipeline/x8_steer_then_probe.py \\
        --model google/gemma-2-27b-it \\
        --directions-dir outputs/gemma-2-27b-it/v2/causal_pilot/directions \\
        --vectors-dir outputs/gemma-2-27b-it/v2/caa_vectors \\
        --output-dir outputs/gemma-2-27b-it/v2/steer_probe \\
        --pairs therapist:empathy drill_sergeant:assertiveness con_artist:honesty \\
        --n-prompts 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "assistant-axis-ref"))

from persona_steering.config import PROMPTS_DIR, TARGET_LAYER
from persona_steering.utils import get_device, log, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--directions-dir", required=True,
                   help="X3b output dir with u_{context}.pt files")
    p.add_argument("--vectors-dir", required=True,
                   help="CAA vectors dir — uses null_{trait}.pt as v_T")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pairs", nargs="+",
                   default=["therapist:empathy",
                            "drill_sergeant:assertiveness",
                            "con_artist:honesty"],
                   help="(context:trait) pairs to test")
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--prompts-file", type=str,
                   default=str(PROMPTS_DIR / "neutral.json"))
    p.add_argument("--layer-steer", type=int, default=TARGET_LAYER)
    p.add_argument("--layers-extract", type=int, nargs="+",
                   default=[15, 20, 22, 25, 30, 35, 40])
    p.add_argument("--alphas-single", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--alphas-mix", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_direction_ctx(dirs_dir: Path, context: str) -> torch.Tensor:
    data = torch.load(dirs_dir / f"u_{context}.pt", map_location="cpu",
                      weights_only=False)
    return data["vector"].float()


def load_direction_trait(vec_dir: Path, trait: str, layer: int,
                         persona: str = "null") -> torch.Tensor:
    data = torch.load(vec_dir / f"{persona}_{trait}.pt",
                      map_location="cpu", weights_only=False)
    v = data["vector"].float()
    return v[layer]


def _register_probes(pd_entry, probe_names, H, baseline_projs, layers_extract,
                     auroc_fn, alpha=None, condition=None,
                     alpha_ctx=None, alpha_trait=None):
    """Build one record per layer with cos/AUROC for each probe direction."""
    records_per_layer = [dict() for _ in layers_extract]
    norms = H.norm(dim=-1).clamp_min(1e-12)  # (N, L)
    for name, d_hat in probe_names.items():
        proj = torch.stack([h @ d_hat for h in H])  # (N, L)
        cos = proj / norms
        base = baseline_projs[name]
        for li, _ in enumerate(layers_extract):
            records_per_layer[li][f"proj_{name}_mean"] = float(proj[:, li].mean())
            records_per_layer[li][f"cos_{name}_mean"] = float(cos[:, li].mean())
            records_per_layer[li][f"auroc_{name}_vs_base"] = auroc_fn(
                proj[:, li].numpy(), base[:, li].numpy())
    for li, L in enumerate(layers_extract):
        rec = records_per_layer[li]
        rec["layer"] = L
        if condition is not None:
            rec["condition"] = condition
            rec["alpha"] = alpha
        else:
            rec["alpha_ctx"] = alpha_ctx
            rec["alpha_trait"] = alpha_trait
    return records_per_layer


def main():
    args = parse_args()
    out = Path(args.output_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from assistant_axis.internals.model import ProbingModel
    from assistant_axis.internals import (ActivationExtractor,
                                          ConversationEncoder)
    from assistant_axis.steering import ActivationSteering

    log.info("Loading model %s …", args.model)
    pm = ProbingModel(args.model, device=str(get_device()))
    encoder = ConversationEncoder(pm.tokenizer, model_name=args.model)
    extractor = ActivationExtractor(pm, encoder)
    hidden = pm.model.config.hidden_size
    log.info("Model loaded. hidden=%d, layer_steer=%d",
             hidden, args.layer_steer)

    # Load prompts
    prompts_data = json.loads(Path(args.prompts_file).read_text())
    all_prompts = prompts_data["questions"] if "questions" in prompts_data \
        else prompts_data
    prompts = random.sample(all_prompts, min(args.n_prompts, len(all_prompts)))
    log.info("Using %d prompts from %s", len(prompts), args.prompts_file)

    dirs_dir = Path(args.directions_dir)
    vec_dir = Path(args.vectors_dir)

    # Parse pair list
    pairs = []
    for spec in args.pairs:
        ctx, trait = spec.split(":")
        pairs.append((ctx.strip(), trait.strip()))
    log.info("Pairs: %s", pairs)

    # ------------------------------------------------------------------
    # Helper: probe an activation against a library of directions.
    def project(h: torch.Tensor, d_hat: torch.Tensor) -> float:
        return float(torch.dot(h.to(d_hat.dtype), d_hat).item())

    # Helper: forward-pass the user prompt WITH add_generation_prompt=True
    # and read activations at the last token (just before generation).
    model_layers = pm.get_layers()

    def get_h(prompt: str, steer_vec=None, alpha: float = 0.0):
        conv = [{"role": "user", "content": prompt}]
        ids = encoder.token_ids(conv, add_generation_prompt=True)
        input_ids = torch.tensor([ids], device=pm.model.device)

        captured = {}

        def mk_hook(li: int):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                captured[li] = t[0, -1, :].detach().cpu().float()
            return hook

        def run_forward():
            # Register extraction hooks AFTER entering any steering context
            # so they run after steering modifies the layer output.
            handles = [model_layers[li].register_forward_hook(mk_hook(li))
                       for li in args.layers_extract]
            try:
                with torch.inference_mode():
                    _ = pm.model(input_ids)
            finally:
                for h in handles:
                    h.remove()

        if steer_vec is None or alpha == 0.0:
            run_forward()
        else:
            with ActivationSteering(
                pm.model,
                steering_vectors=[steer_vec],
                coefficients=[alpha],
                layer_indices=[args.layer_steer],
                positions="all",
            ):
                run_forward()

        return torch.stack([captured[li] for li in args.layers_extract], 0)

    # AUROC helper for two 1-D score samples (steered vs baseline).
    def auroc(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
        # class 1 = a, class 0 = b; AUROC of a > b
        all_scores = np.concatenate([scores_a, scores_b])
        labels = np.concatenate([
            np.ones(len(scores_a)), np.zeros(len(scores_b)),
        ])
        order = np.argsort(all_scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(all_scores) + 1)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
                     / (n_pos * n_neg))

    # ------------------------------------------------------------------
    # Main loop
    summary = {
        "config": vars(args),
        "pairs": {},
    }

    for ctx, trait in pairs:
        log.info("=== (%s, %s) ===", ctx, trait)
        u_C_full = load_direction_ctx(dirs_dir, ctx)   # (hidden,)
        v_null_full = load_direction_trait(
            vec_dir, trait, args.layer_steer, persona="null")
        try:
            v_persona_full = load_direction_trait(
                vec_dir, trait, args.layer_steer, persona=ctx)
            has_persona = True
        except FileNotFoundError:
            log.warning("No persona-conditioned trait vector for %s:%s — "
                        "skipping v_persona probe", ctx, trait)
            v_persona_full = v_null_full  # placeholder, ignored below
            has_persona = False

        n_u = float(u_C_full.norm())
        n_v_null = float(v_null_full.norm())
        n_v_persona = float(v_persona_full.norm())
        u_hat = u_C_full / u_C_full.norm()
        v_null_hat = v_null_full / v_null_full.norm()
        v_persona_hat = v_persona_full / v_persona_full.norm()

        cos_u_vnull = float(torch.dot(u_hat, v_null_hat).item())
        cos_u_vpersona = float(torch.dot(u_hat, v_persona_hat).item())
        cos_vnull_vpersona = float(torch.dot(v_null_hat, v_persona_hat).item())
        # Keep backward-compatible alias
        v_T_full = v_null_full
        v_hat = v_null_hat
        cos_uv = cos_u_vnull
        n_v = n_v_null
        log.info("‖u‖=%.1f  ‖v_null‖=%.1f  ‖v_persona‖=%.1f", n_u,
                 n_v_null, n_v_persona)
        log.info("cos(u, v_null)=%.4f  cos(u, v_persona)=%.4f  "
                 "cos(v_null, v_persona)=%.4f",
                 cos_u_vnull, cos_u_vpersona, cos_vnull_vpersona)

        probe_names = {"ctx": u_hat, "trait_null": v_null_hat}
        if has_persona:
            probe_names["trait_persona"] = v_persona_hat

        # Random direction at unit norm, independent seed per pair.
        r_vec = torch.randn_like(u_C_full)
        r_hat = r_vec / r_vec.norm()

        # We'll steer with α * d_raw so α=1 == one natural vector norm.
        def steer_vec(alpha_ctx, alpha_trait, random_alpha=0.0):
            v = torch.zeros_like(u_C_full)
            if alpha_ctx:
                v = v + alpha_ctx * u_C_full
            if alpha_trait:
                v = v + alpha_trait * v_T_full
            if random_alpha:
                v = v + random_alpha * n_u * r_hat
            return v.to(pm.model.dtype).to(pm.model.device)

        # ---------- Single-direction sweeps
        single_records = []
        conditions = [
            ("ctx",    lambda a: steer_vec(alpha_ctx=a, alpha_trait=0)),
            ("trait",  lambda a: steer_vec(alpha_ctx=0, alpha_trait=a)),
            ("random", lambda a: steer_vec(alpha_ctx=0, alpha_trait=0,
                                            random_alpha=a)),
        ]

        # Precompute baseline (α=0) activations once per prompt
        log.info("  baseline forward passes…")
        baseline_H = []  # list of (n_layers, hidden) per prompt
        for prompt in prompts:
            baseline_H.append(get_h(prompt, steer_vec=None, alpha=0.0))
        baseline_H = torch.stack(baseline_H, 0)  # (N, n_layers, hidden)
        baseline_projs = {
            name: torch.stack([h @ d_hat for h in baseline_H])  # (N, L)
            for name, d_hat in probe_names.items()
        }
        L_steer_idx = args.layers_extract.index(args.layer_steer)

        for cond_name, mk_vec in conditions:
            for alpha in args.alphas_single:
                if alpha == 0.0:
                    H = baseline_H
                else:
                    Hs = []
                    for prompt in prompts:
                        vec = mk_vec(alpha)
                        Hs.append(get_h(prompt, steer_vec=vec, alpha=1.0))
                    H = torch.stack(Hs, 0)

                recs = _register_probes(None, probe_names, H, baseline_projs,
                                         args.layers_extract, auroc,
                                         alpha=alpha, condition=cond_name)
                single_records.extend(recs)
                at_L = recs[L_steer_idx]
                log.info("    %-6s α=%5.2f   L%d  cos_u=%+.3f  "
                         "cos_v_null=%+.3f  cos_v_persona=%s",
                         cond_name, alpha, args.layer_steer,
                         at_L["cos_ctx_mean"], at_L["cos_trait_null_mean"],
                         (f"{at_L['cos_trait_persona_mean']:+.3f}"
                          if has_persona else " n/a"))

        # ---------- 2D mix grid
        mix_records = []
        for a_ctx in args.alphas_mix:
            for a_trait in args.alphas_mix:
                if a_ctx == 0.0 and a_trait == 0.0:
                    H = baseline_H
                else:
                    Hs = []
                    for prompt in prompts:
                        vec = steer_vec(a_ctx, a_trait)
                        Hs.append(get_h(prompt, steer_vec=vec, alpha=1.0))
                    H = torch.stack(Hs, 0)

                recs = _register_probes(None, probe_names, H, baseline_projs,
                                         args.layers_extract, auroc,
                                         alpha_ctx=a_ctx, alpha_trait=a_trait)
                mix_records.extend(recs)
                at_L = recs[L_steer_idx]
                log.info("  mix α_ctx=%.2f α_tr=%.2f    L%d  cos_u=%+.3f  "
                         "cos_v_null=%+.3f  cos_v_persona=%s",
                         a_ctx, a_trait, args.layer_steer,
                         at_L["cos_ctx_mean"], at_L["cos_trait_null_mean"],
                         (f"{at_L['cos_trait_persona_mean']:+.3f}"
                          if has_persona else " n/a"))

        summary["pairs"][f"{ctx}:{trait}"] = {
            "context": ctx,
            "trait": trait,
            "u_norm": n_u,
            "v_null_norm": n_v_null,
            "v_persona_norm": n_v_persona if has_persona else None,
            "cos_u_v_null": cos_u_vnull,
            "cos_u_v_persona": cos_u_vpersona if has_persona else None,
            "cos_v_null_v_persona": cos_vnull_vpersona if has_persona else None,
            "has_persona_probe": has_persona,
            # Back-compat aliases
            "v_norm": n_v_null,
            "cos_u_v": cos_u_vnull,
            "n_prompts": len(prompts),
            "single": single_records,
            "mix": mix_records,
        }

        save_json(summary["pairs"][f"{ctx}:{trait}"],
                  out / "raw" / f"{ctx}_{trait}.json")

    save_json(summary, out / "summary.json")
    log.info("Wrote %s/summary.json", out)


if __name__ == "__main__":
    main()
