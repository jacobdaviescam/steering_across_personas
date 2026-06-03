#!/usr/bin/env python3
"""X9: Steer-and-generate, then probe the generated activations.

Extension of x8. Whereas x8 runs a single forward pass at the
assistant-turn-start token under steering, x9 keeps the steering hook
active for the whole autoregressive decode. We then re-encode
``prompt ++ generated_tokens`` in a second, unsteered forward pass and
read the mean of the assistant-turn activations at a set of probe
layers. This tests whether steering persists through generation and
whether the generated text itself carries the trait direction.

Per (context, trait) pair, per (alpha_ctx, alpha_trait) condition, per
prompt, we:
  1. Build a neutral conversation ``[user=prompt]`` with
     ``add_generation_prompt=True``.
  2. Enter an ``ActivationSteering`` context at ``layer_steer`` with
     steer vector = ``alpha_ctx * u_C + alpha_trait * v_trait``. The
     hook fires on every forward pass during generation so every new
     token's residual is pushed along the steer direction.
  3. ``pm.model.generate(...)`` (greedy) to produce up to
     ``max_new_tokens`` tokens.
  4. Exit the steering context and run one more forward pass on the
     full sequence, capturing layer outputs at the generated-token
     positions. Mean over those positions gives ``h_gen``.
  5. Probe ``cos(h_gen, d_hat)`` for d_hat in {u_C, v_trait_null,
     v_trait_persona}. Aggregate across prompts and report AUROC of
     steered vs alpha=0 baseline.

Generations are persisted to ``generations.jsonl`` so a later
Claude-as-judge pass can be run without re-generating.

Usage:
    python pipeline/x9_steer_generate_probe.py \\
        --model google/gemma-2-27b-it \\
        --directions-dir outputs/gemma-2-27b-it/v2/causal_pilot/directions \\
        --vectors-dir outputs/gemma-2-27b-it/v2/caa_vectors \\
        --output-dir outputs/gemma-2-27b-it/v2/gen_probe \\
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
                   help="CAA vectors dir — uses null_{trait}.pt as v_null "
                        "and {context}_{trait}.pt as v_persona")
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
    p.add_argument("--alphas-ctx", type=float, nargs="+",
                   default=[0.0, 1.0, 2.0, 4.0])
    p.add_argument("--alphas-trait", type=float, nargs="+",
                   default=[0.0, 1.0, 2.0, 4.0])
    p.add_argument("--steer-trait-with",
                   choices=["null", "persona"], default="null",
                   help="Which trait vector to steer with. Probes always "
                        "include both null and persona-conditioned v.")
    p.add_argument("--max-new-tokens", type=int, default=256)
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


def auroc(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """AUROC of scores_a > scores_b (class 1 = a, class 0 = b)."""
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


def main():
    args = parse_args()
    out = Path(args.output_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "generations").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from assistant_axis.internals.model import ProbingModel
    from assistant_axis.internals import ConversationEncoder
    from assistant_axis.steering import ActivationSteering

    log.info("Loading model %s …", args.model)
    pm = ProbingModel(args.model, device=str(get_device()))
    encoder = ConversationEncoder(pm.tokenizer, model_name=args.model)
    hidden = pm.model.config.hidden_size
    model_layers = pm.get_layers()
    log.info("Model loaded. hidden=%d, layer_steer=%d",
             hidden, args.layer_steer)

    prompts_data = json.loads(Path(args.prompts_file).read_text())
    all_prompts = (prompts_data["questions"]
                   if "questions" in prompts_data else prompts_data)
    prompts = random.sample(all_prompts,
                            min(args.n_prompts, len(all_prompts)))
    log.info("Using %d prompts from %s", len(prompts), args.prompts_file)

    dirs_dir = Path(args.directions_dir)
    vec_dir = Path(args.vectors_dir)

    pairs = []
    for spec in args.pairs:
        ctx, trait = spec.split(":")
        pairs.append((ctx.strip(), trait.strip()))
    log.info("Pairs: %s", pairs)

    # ------------------------------------------------------------------
    # Core: generate under steering, then probe the generated activations.
    def generate_and_probe(prompt: str, steer_vec,
                           probe_dirs: dict) -> dict:
        """Returns {gen_text, cos_{name}_at_layer: {...}} plus per-layer
        mean activation vectors keyed by layer index (to stash for AUROC)."""
        conv = [{"role": "user", "content": prompt}]
        prompt_ids = encoder.token_ids(conv, add_generation_prompt=True)
        input_ids = torch.tensor([prompt_ids], device=pm.model.device)
        prompt_len = input_ids.shape[1]

        # Generate with steering hook active
        if steer_vec is None:
            with torch.inference_mode():
                out_ids = pm.model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pm.tokenizer.eos_token_id,
                )
        else:
            with ActivationSteering(
                pm.model,
                steering_vectors=[steer_vec],
                coefficients=[1.0],
                layer_indices=[args.layer_steer],
                positions="all",
            ):
                with torch.inference_mode():
                    out_ids = pm.model.generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=pm.tokenizer.eos_token_id,
                    )

        full_ids = out_ids[0]  # (prompt_len + gen_len,)
        gen_ids = full_ids[prompt_len:]
        gen_text = pm.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Strip trailing end-of-turn tokens etc. Keep all positions that
        # the model actually attended to under steering for the probe.
        gen_len = gen_ids.shape[0]
        if gen_len == 0:
            return {"gen_text": gen_text, "gen_len": 0, "h_gen": None}

        # Re-probe: unsteered forward pass on full sequence, mean over
        # the generated positions at each extract layer.
        captured: dict[int, torch.Tensor] = {}

        def mk_hook(li: int):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                captured[li] = (t[0, prompt_len:prompt_len + gen_len, :]
                                .detach().cpu().float().mean(0))
            return hook

        handles = [model_layers[li].register_forward_hook(mk_hook(li))
                   for li in args.layers_extract]
        try:
            with torch.inference_mode():
                _ = pm.model(full_ids.unsqueeze(0))
        finally:
            for h in handles:
                h.remove()

        h_gen = torch.stack([captured[li] for li in args.layers_extract], 0)
        return {"gen_text": gen_text, "gen_len": int(gen_len),
                "h_gen": h_gen}

    # ------------------------------------------------------------------
    summary = {"config": vars(args), "pairs": {}}

    for ctx, trait in pairs:
        log.info("=== (%s, %s) ===", ctx, trait)
        u_C_full = load_direction_ctx(dirs_dir, ctx)
        v_null_full = load_direction_trait(
            vec_dir, trait, args.layer_steer, persona="null")
        try:
            v_persona_full = load_direction_trait(
                vec_dir, trait, args.layer_steer, persona=ctx)
            has_persona = True
        except FileNotFoundError:
            log.warning("No persona-conditioned trait vector for %s:%s",
                        ctx, trait)
            v_persona_full = v_null_full
            has_persona = False

        n_u = float(u_C_full.norm())
        n_v_null = float(v_null_full.norm())
        n_v_persona = float(v_persona_full.norm())
        u_hat = u_C_full / u_C_full.norm()
        v_null_hat = v_null_full / v_null_full.norm()
        v_persona_hat = v_persona_full / v_persona_full.norm()
        log.info("‖u‖=%.1f  ‖v_null‖=%.1f  ‖v_persona‖=%.1f",
                 n_u, n_v_null, n_v_persona)

        probe_names = {"ctx": u_hat, "trait_null": v_null_hat}
        if has_persona:
            probe_names["trait_persona"] = v_persona_hat

        v_trait_steer = (v_persona_full
                         if args.steer_trait_with == "persona" and has_persona
                         else v_null_full)

        def make_steer_vec(a_ctx: float, a_trait: float):
            if a_ctx == 0.0 and a_trait == 0.0:
                return None
            v = a_ctx * u_C_full + a_trait * v_trait_steer
            return v.to(pm.model.dtype).to(pm.model.device)

        records = []  # one per (alpha_ctx, alpha_trait)
        gen_rows = []  # one per (alpha_ctx, alpha_trait, prompt)
        # We cache per-prompt H arrays keyed by (a_ctx, a_trait).
        H_by_cond: dict[tuple, torch.Tensor] = {}
        L_steer_idx = args.layers_extract.index(args.layer_steer)

        # Iterate conditions. (0,0) is the baseline; run it first.
        conds = [(0.0, 0.0)]
        for a_ctx in args.alphas_ctx:
            for a_trait in args.alphas_trait:
                if (a_ctx, a_trait) == (0.0, 0.0):
                    continue
                conds.append((a_ctx, a_trait))

        for a_ctx, a_trait in conds:
            log.info("  cond α_ctx=%.2f α_tr=%.2f — generating %d prompts…",
                     a_ctx, a_trait, len(prompts))
            steer_vec = make_steer_vec(a_ctx, a_trait)
            Hs = []  # list of (n_layers, hidden)
            for pi, prompt in enumerate(prompts):
                res = generate_and_probe(prompt, steer_vec, probe_names)
                if res["h_gen"] is None:
                    # Model produced no tokens — skip this prompt.
                    continue
                Hs.append(res["h_gen"])
                gen_rows.append({
                    "alpha_ctx": a_ctx,
                    "alpha_trait": a_trait,
                    "prompt_idx": pi,
                    "prompt": prompt,
                    "gen_text": res["gen_text"],
                    "gen_len": res["gen_len"],
                })
            if not Hs:
                log.warning("   no successful generations for (%s,%s) "
                            "at α_ctx=%.2f α_tr=%.2f",
                            ctx, trait, a_ctx, a_trait)
                continue
            H = torch.stack(Hs, 0)  # (N, L, D)
            H_by_cond[(a_ctx, a_trait)] = H

        # Baseline projections
        if (0.0, 0.0) not in H_by_cond:
            log.error("No baseline activations for (%s,%s) — skipping pair",
                      ctx, trait)
            continue
        H_base = H_by_cond[(0.0, 0.0)]
        baseline_projs = {
            name: torch.stack([h @ d_hat for h in H_base])  # (N, L)
            for name, d_hat in probe_names.items()
        }

        # Build per-condition records (mean cos over prompts, AUROC vs base).
        for a_ctx, a_trait in conds:
            if (a_ctx, a_trait) not in H_by_cond:
                continue
            H = H_by_cond[(a_ctx, a_trait)]
            norms = H.norm(dim=-1).clamp_min(1e-12)
            for li, L in enumerate(args.layers_extract):
                rec = {"alpha_ctx": a_ctx, "alpha_trait": a_trait,
                       "layer": L, "n_prompts": int(H.shape[0])}
                for name, d_hat in probe_names.items():
                    proj = torch.stack([h @ d_hat for h in H])  # (N, L)
                    cos = proj / norms
                    rec[f"proj_{name}_mean"] = float(proj[:, li].mean())
                    rec[f"cos_{name}_mean"] = float(cos[:, li].mean())
                    rec[f"auroc_{name}_vs_base"] = auroc(
                        proj[:, li].numpy(),
                        baseline_projs[name][:, li].numpy())
                records.append(rec)
            at_L = [r for r in records
                    if r["alpha_ctx"] == a_ctx
                    and r["alpha_trait"] == a_trait
                    and r["layer"] == args.layer_steer][0]
            log.info("    α_ctx=%.2f α_tr=%.2f  L%d  "
                     "cos_u=%+.3f  cos_v_null=%+.3f  cos_v_persona=%s",
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
            "has_persona_probe": has_persona,
            "steer_trait_with": args.steer_trait_with,
            "n_prompts_target": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "records": records,
        }
        save_json(summary["pairs"][f"{ctx}:{trait}"],
                  out / "raw" / f"{ctx}_{trait}.json")
        with (out / "generations" / f"{ctx}_{trait}.jsonl").open("w") as f:
            for row in gen_rows:
                f.write(json.dumps(row) + "\n")

    save_json(summary, out / "summary.json")
    log.info("Wrote %s/summary.json", out)


if __name__ == "__main__":
    main()
