#!/usr/bin/env python3
"""X3c: Causal alpha sweep — show that steering along context direction
causally degrades null-trained probe AUROC while shifting outputs toward
the target context.

For each (trait T, context C, alpha):
  * Generate steered responses for 20 (high, low) trait-eliciting prompt pairs
  * Compute classifier P(context = C | output)
  * Compute null-trained probe AUROC over (high, low) activations
  * Compute LLM-judge coherence

Three conditions per (T, C):
  main:  steer along orthogonalised u_{C,T}^perp
  rand:  steer along random direction at matched norm
  trait: steer along null-trait vector v_{T,null} at matched norm

Pilot mode (--pilot) runs 1 trait x 1 context x 5 alphas x 5 pairs.

Usage (pilot):
    python pipeline/x3c_causal_sweep.py \\
        --model google/gemma-2-27b-it \\
        --directions-dir outputs/gemma-2-27b-it/v2/causal/directions \\
        --null-trait-vectors-dir outputs/gemma-2-27b-it/v2/vectors \\
        --classifier-dir outputs/gemma-2-27b-it/v2/classifier \\
        --probes-dir outputs/gemma-2-27b-it/v2/probes/probes_pkl \\
        --output-dir outputs/gemma-2-27b-it/v2/causal \\
        --pilot --traits honesty --contexts therapist
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from persona_steering.config import (PERSONA_SLUGS, PROMPTS_DIR, TARGET_LAYER, Trait)
from persona_steering.personas import load_all_personas
from persona_steering.utils import get_device, log, model_short_name, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--directions-dir", type=str, required=True,
                   help="Output of x3b: u_{C}_{T}_orth.pt files")
    p.add_argument("--null-trait-vectors-dir", type=str, required=True,
                   help="Dir with null_{trait}.pt CAA/IV trait vectors")
    p.add_argument("--classifier-dir", type=str, required=True,
                   help="Output of x1: head.pt + metrics.json")
    p.add_argument("--probes-dir", type=str, required=True,
                   help="Output of x2 probes_pkl/: trait_A_null.pkl files")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--eliciting-pairs", type=str,
                   default=str(PROMPTS_DIR / "eliciting_pairs.json"))
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--traits", nargs="+", default=None)
    p.add_argument("--contexts", nargs="+", default=None)
    p.add_argument("--conditions", nargs="+",
                   default=["main", "rand", "trait"])
    p.add_argument("--target-persona-for-system-prompt", type=str, default="null",
                   help="System prompt context to use for steered generations")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-model", type=str, default="claude-sonnet-4-20250514")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip Claude judge for coherence (saves API cost)")
    p.add_argument("--pilot", action="store_true",
                   help="Pilot: 1 trait x 1 context x first 5 alphas x first 5 pairs")
    return p.parse_args()


def load_classifier(classifier_dir: Path, sbert_model: str | None = None):
    head_data = torch.load(classifier_dir / "head.pt", map_location="cpu", weights_only=False)
    contexts = head_data["contexts"]
    in_dim = head_data["in_dim"]
    n_classes = head_data["n_classes"]
    sbert_name = sbert_model or head_data.get("sbert_model", "all-mpnet-base-v2")

    head = torch.nn.Linear(in_dim, n_classes)
    head.load_state_dict(head_data["state_dict"])
    head.eval()

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(sbert_name)
    return head, sbert, contexts


def encode_and_predict(texts: list[str], head, sbert, ctx_to_idx, target_ctx: str) -> np.ndarray:
    X = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    with torch.no_grad():
        logits = head(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).numpy()
    target_idx = ctx_to_idx[target_ctx]
    return probs[:, target_idx]


def load_null_probe(probes_dir: Path, trait: str):
    path = probes_dir / f"{trait}_A_null.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def auroc_from_probe(probe_pkg, X_pos: np.ndarray, X_neg: np.ndarray) -> float | None:
    if X_pos.size == 0 or X_neg.size == 0:
        return None
    from sklearn.metrics import roc_auc_score
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    p = probe_pkg["probe"].predict_proba(probe_pkg["scaler"].transform(X))[:, 1]
    return float(roc_auc_score(y, p))


def coherence_judge(texts: list[str], judge_model: str) -> list[float]:
    """Claude LLM judge: 0 (incoherent) to 1 (coherent), per text."""
    import anthropic
    client = anthropic.Anthropic()
    scores = []
    sys_prompt = (
        "Rate the coherence of the following AI response on a 0.0–1.0 scale. "
        "1.0 = fluent, on-topic, intelligible. "
        "0.0 = degenerate, repetitive, nonsensical. "
        "Output ONLY a single float between 0 and 1, nothing else."
    )
    for text in texts:
        try:
            resp = client.messages.create(
                model=judge_model, max_tokens=8,
                system=sys_prompt,
                messages=[{"role": "user", "content": text[:1500]}],
            )
            s = float(resp.content[0].text.strip())
            scores.append(max(0.0, min(1.0, s)))
        except Exception as e:  # noqa: BLE001
            log.warning("Judge failed: %s", e)
            scores.append(float("nan"))
    return scores


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    (out / "generations").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    rng_np = np.random.default_rng(args.seed)
    random.seed(args.seed)

    with open(args.eliciting_pairs) as f:
        elic = json.load(f)["traits"]

    traits = args.traits or list(elic.keys())
    contexts = args.contexts or [c for c in PERSONA_SLUGS if c not in {"null", "nonsense"}]
    alphas = list(args.alphas)
    n_pairs = args.n_pairs

    if args.pilot:
        traits = traits[:1]
        contexts = contexts[:1]
        alphas = alphas[:5]
        n_pairs = min(5, n_pairs)
        log.info("PILOT: %s x %s, alphas=%s, pairs=%d",
                 traits, contexts, alphas, n_pairs)

    log.info("Loading classifier from %s", args.classifier_dir)
    head, sbert, ctx_labels = load_classifier(Path(args.classifier_dir))
    ctx_to_idx = {c: i for i, c in enumerate(ctx_labels)}

    # Load model + steering infra
    from assistant_axis.internals.model import ProbingModel
    from assistant_axis.steering import ActivationSteering
    from assistant_axis.generation import generate_response, format_conversation
    from assistant_axis.internals import (ConversationEncoder, ActivationExtractor,
                                          SpanMapper)
    log.info("Loading model %s ...", args.model)
    pm = ProbingModel(args.model, device=str(get_device()))
    encoder = ConversationEncoder(pm.tokenizer, model_name=args.model)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)

    # Load target persona's system prompt (default: null)
    personas = {p.slug: p for p in load_all_personas()}
    target_persona = personas[args.target_persona_for_system_prompt]
    sys_prompt = (target_persona.system_prompt_variants[0]
                  if target_persona.system_prompt_variants else "")

    sweep_results = []  # one row per (trait, ctx, alpha, condition)

    for trait in traits:
        log.info("=== TRAIT %s ===", trait)
        pairs = elic[trait][:n_pairs]
        probe_pkg = load_null_probe(Path(args.probes_dir), trait)
        if probe_pkg is None:
            log.warning("No null probe for %s — AUROC will be None", trait)

        v_T_null_path = Path(args.null_trait_vectors_dir) / f"null_{trait}.pt"
        v_T_null = None
        if v_T_null_path.exists():
            d = torch.load(v_T_null_path, map_location="cpu", weights_only=False)
            v_T_null = d["vector"].float()[args.layer]

        for ctx in contexts:
            log.info("--- CONTEXT %s ---", ctx)
            orth_path = Path(args.directions_dir) / f"u_{ctx}_{trait}_orth.pt"
            if not orth_path.exists():
                log.warning("Missing orth direction %s — skipping", orth_path)
                continue
            orth_data = torch.load(orth_path, map_location="cpu", weights_only=False)
            u_orth = orth_data["vector"].float()
            target_norm = float(u_orth.norm())

            # Build vectors per condition
            condition_vecs: dict[str, torch.Tensor] = {}
            if "main" in args.conditions:
                condition_vecs["main"] = u_orth
            if "rand" in args.conditions:
                rnd = torch.tensor(rng_np.standard_normal(u_orth.shape[0]),
                                   dtype=torch.float32)
                rnd = rnd / rnd.norm() * target_norm
                condition_vecs["rand"] = rnd
            if "trait" in args.conditions and v_T_null is not None:
                vt = v_T_null / max(v_T_null.norm().item(), 1e-12) * target_norm
                condition_vecs["trait"] = vt

            for cond_name, vec in condition_vecs.items():
                for alpha in alphas:
                    log.info("alpha=%.2f cond=%s", alpha, cond_name)
                    pos_texts, neg_texts = [], []

                    for pi, pair in enumerate(pairs):
                        for direction, prompt in (("pos", pair["pos"]),
                                                  ("neg", pair["neg"])):
                            conv = format_conversation(sys_prompt, prompt, pm.tokenizer)
                            if alpha == 0.0:
                                resp = generate_response(
                                    pm.model, pm.tokenizer, conv,
                                    max_new_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                )
                            else:
                                with ActivationSteering(
                                    pm.model,
                                    steering_vectors=[vec],
                                    coefficients=[alpha],
                                    layer_indices=[args.layer],
                                ):
                                    resp = generate_response(
                                        pm.model, pm.tokenizer, conv,
                                        max_new_tokens=args.max_tokens,
                                        temperature=args.temperature,
                                    )
                            (pos_texts if direction == "pos" else neg_texts).append({
                                "pair_index": pi,
                                "prompt": prompt,
                                "response": resp,
                            })

                    # Persist generations
                    gen_file = (out / "generations"
                                / f"{trait}_{ctx}_{cond_name}_a{alpha:g}.jsonl")
                    with open(gen_file, "w") as f:
                        for entry in pos_texts:
                            f.write(json.dumps({**entry, "direction": "pos",
                                                "trait": trait, "context": ctx,
                                                "condition": cond_name,
                                                "alpha": alpha}) + "\n")
                        for entry in neg_texts:
                            f.write(json.dumps({**entry, "direction": "neg",
                                                "trait": trait, "context": ctx,
                                                "condition": cond_name,
                                                "alpha": alpha}) + "\n")

                    # Classifier P(C | output) averaged over pos+neg
                    all_texts = [e["response"] for e in pos_texts + neg_texts]
                    p_ctx = encode_and_predict(all_texts, head, sbert, ctx_to_idx, ctx)
                    mean_p = float(np.mean(p_ctx))

                    # Null probe AUROC on activations of the same generations
                    auroc_val = None
                    if probe_pkg is not None:
                        # Re-run extractor over the new conversations
                        all_convs = []
                        for e in pos_texts + neg_texts:
                            conv_full = format_conversation(sys_prompt, e["prompt"], pm.tokenizer)
                            conv_full = conv_full + [{"role": "assistant",
                                                      "content": e["response"]}]
                            all_convs.append(conv_full)
                        try:
                            batch_acts, batch_meta = extractor.batch_conversations(
                                all_convs, layer=None, max_length=2048
                            )
                            _, batch_spans, span_meta = encoder.build_batch_turn_spans(all_convs)
                            per_conv = span_mapper.map_spans(
                                batch_acts, batch_spans, {**batch_meta, **span_meta}
                            )
                            X = []
                            for ca in per_conv:
                                if ca.numel() == 0:
                                    continue
                                X.append(ca[-1][args.layer].float().cpu().numpy())
                            n_pos = len(pos_texts)
                            X = np.array(X)
                            if len(X) >= 2:
                                X_pos_arr = X[:n_pos]
                                X_neg_arr = X[n_pos:]
                                auroc_val = auroc_from_probe(probe_pkg, X_pos_arr, X_neg_arr)
                        except Exception as e:  # noqa: BLE001
                            log.warning("AUROC eval failed for %s/%s a=%.2f: %s",
                                        trait, ctx, alpha, e)

                    coherence = None
                    if not args.skip_judge:
                        coh_scores = coherence_judge(all_texts, args.judge_model)
                        coherence = float(np.nanmean(coh_scores)) if coh_scores else None

                    sweep_results.append({
                        "trait": trait, "context": ctx,
                        "condition": cond_name, "alpha": alpha,
                        "p_context": mean_p,
                        "auroc": auroc_val,
                        "coherence": coherence,
                        "n_pos": len(pos_texts), "n_neg": len(neg_texts),
                    })
                    log.info("  P(%s)=%.3f  AUROC=%s  coh=%s",
                             ctx, mean_p,
                             f"{auroc_val:.3f}" if auroc_val is not None else "—",
                             f"{coherence:.3f}" if coherence is not None else "—")

    save_json({"results": sweep_results, "config": vars(args)},
              out / "metrics" / "sweep_results.json")

    portrait = np.array([
        [r["p_context"], r["auroc"] if r["auroc"] is not None else np.nan]
        for r in sweep_results
    ])
    np.save(out / "metrics" / "phase_portrait.npy", portrait)
    log.info("Done. %d sweep points written to %s",
             len(sweep_results), out / "metrics" / "sweep_results.json")


if __name__ == "__main__":
    main()
