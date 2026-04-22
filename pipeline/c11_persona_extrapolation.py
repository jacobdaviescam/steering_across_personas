#!/usr/bin/env python3
"""C11 — Persona-space extrapolation.

Compute persona-identity vectors ``d_{p→p'} = mu_{p'} - mu_p`` from the
persona baselines (pooled-mean activations). For each of a handful of
pair (p, p'):

* Generate baseline responses under p's system prompt (no steering).
* Generate with ``alpha * d_{p→p'}`` at a small alpha.
* Generate with a larger alpha (2–3×).

Each response is judged on three axes:

* source-persona resemblance (should decrease with alpha)
* target-persona resemblance (should increase with alpha)
* coherence / fluency (manifold: stays high; UoS: collapses)

Usage:
    python pipeline/c11_persona_extrapolation.py --model google/gemma-2-27b-it
    python pipeline/c11_persona_extrapolation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from persona_steering.config import TARGET_LAYER, Trait
from persona_steering.council import (
    all_traits,
    council_dir,
    persona_baseline,
    score_generations,
)
from persona_steering.data import load_trait_dataset
from persona_steering.personas import load_all_personas
from persona_steering.utils import log, save_fig


DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("farmer", "politician"),
    ("therapist", "drill_sergeant"),
    ("kindergarten_teacher", "con_artist"),
    ("professor", "street_hustler"),
    ("surgeon", "tech_ceo"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C11 persona-space extrapolation")
    p.add_argument("--model", type=str, default="google/gemma-2-27b-it")
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--alpha-small", type=float, default=4.0)
    p.add_argument("--alpha-large", type=float, default=10.0)
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--pairs", nargs="+", default=None,
                   help="format: src:tgt src:tgt ... (uses DEFAULT_PAIRS if omitted)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--judge-only", action="store_true")
    return p.parse_args()


def parse_pairs(raw: list[str] | None) -> list[tuple[str, str]]:
    if not raw:
        return list(DEFAULT_PAIRS)
    pairs = []
    for item in raw:
        src, tgt = item.split(":")
        pairs.append((src.strip(), tgt.strip()))
    return pairs


def sample_neutral_questions(n: int, seed: int) -> list[str]:
    """Pool questions across every trait and sample a neutral-ish subset."""
    pool = []
    for t in all_traits():
        try:
            pool.extend(load_trait_dataset(Trait(t)).questions)
        except FileNotFoundError:
            continue
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def build_direction_vector(model: str, src: str, tgt: str, layer: int) -> torch.Tensor:
    mu_src = persona_baseline(model, src, layer=layer)
    mu_tgt = persona_baseline(model, tgt, layer=layer)
    return mu_tgt - mu_src


def generate(
    model_hf_id: str,
    layer: int,
    persona,
    questions: list[str],
    vector: torch.Tensor | None,
) -> list[str]:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))
    from assistant_axis import ProbingModel, ActivationSteering  # type: ignore

    m = ProbingModel(model_hf_id)
    ctx = ActivationSteering(m, layer=layer, vector=vector) if vector is not None else None
    texts = []
    try:
        if ctx is not None:
            with ctx:
                for q in questions:
                    texts.append(m.generate(
                        system_prompt=persona.default_system_prompt,
                        user_prompt=q, max_new_tokens=400, temperature=0.7,
                    ))
        else:
            for q in questions:
                texts.append(m.generate(
                    system_prompt=persona.default_system_prompt,
                    user_prompt=q, max_new_tokens=400, temperature=0.7,
                ))
    finally:
        if hasattr(m, "cleanup"):
            m.cleanup()
    return texts


def score_axes(text: str, src_desc: str, tgt_desc: str, judge) -> dict:
    """Single-judge call per axis. Returns three 0-1 floats."""
    import json as _json

    def axis(prompt: str) -> float:
        resp = judge.client.messages.create(
            model=judge.model, max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            data = _json.loads(resp.content[0].text)
            return max(0.0, min(1.0, float(data["score"])))
        except Exception:
            return 0.5

    src_prompt = (
        "Rate 0.0-1.0 how consistent this text is with the persona below.\n"
        f"Persona:\n{src_desc}\n\nText:\n---\n{text}\n---\n"
        'Respond with ONLY {"score": <float>, "explanation": ""}'
    )
    tgt_prompt = src_prompt.replace(src_desc, tgt_desc)
    coherence_prompt = (
        "Rate 0.0-1.0 the fluency and internal coherence of this text "
        "(1.0 = perfectly fluent and coherent, 0.0 = gibberish).\n"
        f"Text:\n---\n{text}\n---\n"
        'Respond with ONLY {"score": <float>, "explanation": ""}'
    )
    return {
        "src_resemblance": axis(src_prompt),
        "tgt_resemblance": axis(tgt_prompt),
        "coherence": axis(coherence_prompt),
    }


def main() -> None:
    args = parse_args()
    out_dir = council_dir(args.model, "c11")
    pairs = parse_pairs(args.pairs)

    persona_objs = {p.slug: p for p in load_all_personas()}
    questions = sample_neutral_questions(args.n_questions, args.seed)
    gens_path = out_dir / "generations.jsonl"

    if args.dry_run:
        plan = {"pairs": pairs, "n_questions": args.n_questions,
                "alphas": [0.0, args.alpha_small, args.alpha_large]}
        json.dump(plan, open(out_dir / "plan.json", "w"), indent=2)
        log.info("c11 dry-run plan: %s", plan)
        return

    if not args.judge_only:
        with open(gens_path, "w") as f:
            for src, tgt in pairs:
                d = build_direction_vector(args.model, src, tgt, args.layer)
                persona_src = persona_objs[src]
                for alpha, label in [(0.0, "baseline"),
                                     (args.alpha_small, "small"),
                                     (args.alpha_large, "large")]:
                    v = None if alpha == 0.0 else d * (alpha / (d.norm() + 1e-12))
                    log.info("c11 generate %s->%s alpha=%s", src, tgt, label)
                    texts = generate(args.model, args.layer, persona_src, questions, v)
                    for q, text in zip(questions, texts):
                        f.write(json.dumps({
                            "src": src, "tgt": tgt, "alpha_label": label,
                            "alpha": alpha, "question": q, "text": text,
                        }) + "\n")

    # --- judge ---
    from persona_steering.evaluation import LLMJudge
    judge = LLMJudge()
    scored_path = out_dir / "scored.jsonl"
    with open(scored_path, "w") as f:
        for line in open(gens_path):
            row = json.loads(line)
            src_desc = persona_objs[row["src"]].description or persona_objs[row["src"]].default_system_prompt[:400]
            tgt_desc = persona_objs[row["tgt"]].description or persona_objs[row["tgt"]].default_system_prompt[:400]
            row.update(score_axes(row["text"], src_desc, tgt_desc, judge))
            f.write(json.dumps(row) + "\n")

    # --- summary + figure ---
    rows = [json.loads(line) for line in open(scored_path)]
    summary = summarize(rows, pairs)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    make_figure(summary, pairs, out_dir)


def summarize(rows: list[dict], pairs: list[tuple[str, str]]) -> dict:
    out: dict[str, dict] = {}
    for src, tgt in pairs:
        key = f"{src}->{tgt}"
        per_alpha: dict[str, dict] = {}
        for label in ("baseline", "small", "large"):
            subset = [r for r in rows if r["src"] == src and r["tgt"] == tgt and r["alpha_label"] == label]
            if not subset:
                continue
            per_alpha[label] = {
                axis: float(np.mean([r[axis] for r in subset]))
                for axis in ("src_resemblance", "tgt_resemblance", "coherence")
            }
            per_alpha[label]["n"] = len(subset)
        out[key] = per_alpha
    return out


def make_figure(summary: dict, pairs: list[tuple[str, str]], out_dir: Path) -> None:
    axes_labels = ("src_resemblance", "tgt_resemblance", "coherence")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for pair_idx, (src, tgt) in enumerate(pairs):
        key = f"{src}->{tgt}"
        if key not in summary:
            continue
        by_alpha = summary[key]
        x_labels = list(by_alpha.keys())
        x = np.arange(len(x_labels))
        for a, axis in enumerate(axes_labels):
            ys = [by_alpha[lab][axis] for lab in x_labels]
            axes[a].plot(x, ys, marker="o", label=key)
            axes[a].set_xticks(x)
            axes[a].set_xticklabels(x_labels)
            axes[a].set_ylim(0, 1)
            axes[a].set_title(axis)
    axes[-1].legend(fontsize=7, loc="lower left", bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()
    save_fig(fig, out_dir / "fig_e11_extrapolation.png")


if __name__ == "__main__":
    main()
