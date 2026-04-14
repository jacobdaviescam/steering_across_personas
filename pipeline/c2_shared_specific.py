#!/usr/bin/env python3
"""C2 — Shared + specific decomposition (behavioral).

For each (persona, trait):

* ``u_t = mean_p v_{p,t}`` (shared) and ``w_{p,t} = v_{p,t} - u_t`` (specific).
* Normalize each steering vector to a common L2 magnitude ``alpha``.
* Three generation conditions:
    (a) shared only — steer with ``u_t`` while prompting as persona p
    (b) full — steer with ``v_{p,t} = u_t + w_{p,t}``
    (c) mismatched — steer with ``u_t + w_{p',t}`` where p' is the furthest
        persona by baseline vector cosine

Each response is judged for trait expression AND persona coherence. Primary
contrasts: (b) − (a) on trait expression; (b) − (c) on persona coherence.

This script depends on ``assistant-axis-ref`` for steering via forward hooks.
It writes custom vectors to a temporary directory and reuses the generation
loop from :mod:`pipeline.8_steered_generation` (imported as a module) so we
don't duplicate the generation/hooking code.

Usage:
    python pipeline/c2_shared_specific.py --model google/gemma-2-27b-it \
        --pilot --traits assertiveness empathy honesty \
        --personas farmer politician therapist drill_sergeant con_artist
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from persona_steering.config import TARGET_LAYER, Trait
from persona_steering.council import (
    all_personas,
    all_traits,
    cosine,
    council_dir,
    load_trait_vector,
    score_generations,
)
from persona_steering.data import load_trait_dataset
from persona_steering.personas import load_all_personas
from persona_steering.utils import log, save_fig


CONDITIONS = ("shared", "full", "mismatched")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C2 shared+specific behavioral decomposition")
    p.add_argument("--model", type=str, default="google/gemma-2-27b-it")
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    p.add_argument("--alpha", type=float, default=4.0, help="steering magnitude")
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--personas", nargs="+", default=None)
    p.add_argument("--traits", nargs="+", default=None)
    p.add_argument("--pilot", action="store_true", help="pilot subset (3 traits × 5 personas)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="plan only, no generation")
    p.add_argument(
        "--judge-only",
        action="store_true",
        help="skip generation, re-score existing generations.jsonl (offline rerun)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Vector construction
# ---------------------------------------------------------------------------

def build_condition_vectors(
    model: str,
    personas: list[str],
    traits: list[str],
    alpha: float,
    layer: int,
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Return {(persona, trait, condition): vector_at_alpha_magnitude}."""
    vectors: dict[tuple[str, str], torch.Tensor] = {}
    for p in personas:
        for t in traits:
            vectors[(p, t)] = load_trait_vector(model, p, t, layer)

    out: dict[tuple[str, str, str], torch.Tensor] = {}
    for t in traits:
        stack = torch.stack([vectors[(p, t)] for p in personas])
        u = stack.mean(dim=0)
        for p in personas:
            v_full = vectors[(p, t)]
            w_p = v_full - u
            # pick furthest persona p' by baseline cosine for mismatched residual
            cosines = {
                q: cosine(vectors[(p, t)], vectors[(q, t)])
                for q in personas if q != p
            }
            p_far = min(cosines, key=cosines.get)
            v_mismatched = u + (vectors[(p_far, t)] - u)

            for name, v in [("shared", u), ("full", v_full), ("mismatched", v_mismatched)]:
                n = v.norm()
                scaled = v * (alpha / (n + 1e-12))
                out[(p, t, name)] = scaled
    return out


# ---------------------------------------------------------------------------
# Generation (delegated to assistant_axis ActivationSteering)
# ---------------------------------------------------------------------------

def generate_for_condition(
    model_hf_id: str,
    layer: int,
    persona,  # PersonaConfig
    trait: Trait,
    questions: list[str],
    steering_vector: torch.Tensor,
    max_new_tokens: int = 400,
    temperature: float = 0.7,
) -> list[str]:
    """Generate ``len(questions)`` responses with ``steering_vector`` applied.

    Imports ProbingModel / ActivationSteering lazily so the planning/judge-only
    paths don't require the GPU environment.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))
    from assistant_axis import ProbingModel, ActivationSteering  # type: ignore

    model = ProbingModel(model_hf_id)
    steering = ActivationSteering(model, layer=layer, vector=steering_vector)
    texts = []
    try:
        with steering:
            for q in questions:
                out = model.generate(
                    system_prompt=persona.default_system_prompt,
                    user_prompt=q,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                texts.append(out)
    finally:
        model.cleanup() if hasattr(model, "cleanup") else None
    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def persona_description(persona) -> str:
    return persona.description or persona.default_system_prompt[:400]


def main() -> None:
    args = parse_args()
    out_dir = council_dir(args.model, "c2")

    if args.pilot:
        personas = ["farmer", "politician", "therapist", "drill_sergeant", "con_artist"]
        traits = ["assertiveness", "empathy", "honesty"]
    else:
        personas = args.personas or all_personas()
        traits = args.traits or all_traits()

    persona_objs = {p.slug: p for p in load_all_personas()}
    rng = random.Random(args.seed)

    # --- question sampling ---
    questions_by_trait: dict[str, list[str]] = {}
    for t in traits:
        ds = load_trait_dataset(Trait(t))
        qs = list(ds.questions)
        rng.shuffle(qs)
        questions_by_trait[t] = qs[: args.n_questions]

    log.info(
        "c2 plan: personas=%s traits=%s n_questions=%d conditions=%s total_generations=%d",
        personas, traits, args.n_questions, CONDITIONS,
        len(personas) * len(traits) * len(CONDITIONS) * args.n_questions,
    )

    if args.dry_run:
        json.dump(
            {"personas": personas, "traits": traits, "n_questions": args.n_questions},
            open(out_dir / "plan.json", "w"),
            indent=2,
        )
        return

    gens_path = out_dir / "generations.jsonl"

    if not args.judge_only:
        # --- build steering vectors ---
        vectors = build_condition_vectors(args.model, personas, traits, args.alpha, args.layer)

        # --- generation loop ---
        with open(gens_path, "w") as f:
            for trait in traits:
                questions = questions_by_trait[trait]
                for p in personas:
                    for cond in CONDITIONS:
                        v = vectors[(p, trait, cond)]
                        log.info("c2 generate persona=%s trait=%s cond=%s", p, trait, cond)
                        texts = generate_for_condition(
                            args.model, args.layer, persona_objs[p], Trait(trait),
                            questions, v,
                        )
                        for q, text in zip(questions, texts):
                            f.write(json.dumps({
                                "persona": p, "trait": trait, "condition": cond,
                                "question": q, "text": text,
                            }) + "\n")
        log.info("c2 wrote %s", gens_path)

    # --- judging ---
    rows = [json.loads(line) for line in open(gens_path)]
    scored_path = out_dir / "scored.jsonl"
    with open(scored_path, "w") as f:
        for row in rows:
            trait = Trait(row["trait"])
            persona = persona_objs[row["persona"]]
            scored = score_generations(
                [row["text"]], trait,
                persona_description=persona_description(persona),
            )[0]
            row.update({
                "trait_score": scored.trait_score,
                "persona_coherence": scored.persona_coherence,
            })
            f.write(json.dumps(row) + "\n")

    # --- summary + figure ---
    scored_rows = [json.loads(line) for line in open(scored_path)]
    summary = summarize(scored_rows, traits, personas)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    make_figure(summary, traits, out_dir)


def summarize(rows: list[dict], traits: list[str], personas: list[str]) -> dict:
    """Aggregate mean ± 95% CI per (trait, condition) and contrasts."""
    out: dict[str, dict] = {}
    for trait in traits:
        per_cond: dict[str, dict] = {}
        for cond in CONDITIONS:
            trait_scores = [r["trait_score"] for r in rows if r["trait"] == trait and r["condition"] == cond]
            persona_scores = [r["persona_coherence"] for r in rows if r["trait"] == trait and r["condition"] == cond]
            per_cond[cond] = {
                "trait_mean": float(np.mean(trait_scores)) if trait_scores else None,
                "trait_ci95": _ci95(trait_scores),
                "persona_mean": float(np.mean(persona_scores)) if persona_scores else None,
                "persona_ci95": _ci95(persona_scores),
                "n": len(trait_scores),
            }
        contrasts = {
            "full_minus_shared_trait": per_cond["full"]["trait_mean"] - per_cond["shared"]["trait_mean"]
                if per_cond["full"]["trait_mean"] is not None and per_cond["shared"]["trait_mean"] is not None
                else None,
            "full_minus_mismatched_persona": per_cond["full"]["persona_mean"] - per_cond["mismatched"]["persona_mean"]
                if per_cond["full"]["persona_mean"] is not None and per_cond["mismatched"]["persona_mean"] is not None
                else None,
        }
        out[trait] = {"conditions": per_cond, "contrasts": contrasts}
    return out


def _ci95(xs: list[float]) -> list[float] | None:
    if len(xs) < 2:
        return None
    arr = np.asarray(xs, dtype=float)
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return [float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)]


def make_figure(summary: dict, traits: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    y = np.arange(len(traits))

    trait_c = [summary[t]["contrasts"]["full_minus_shared_trait"] or 0.0 for t in traits]
    persona_c = [summary[t]["contrasts"]["full_minus_mismatched_persona"] or 0.0 for t in traits]

    axes[0].barh(y, trait_c, color="steelblue")
    axes[0].axvline(0, color="black")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(traits)
    axes[0].set_xlabel("(full) − (shared)  on trait expression")
    axes[0].set_title("Does the specific residual add trait signal?")

    axes[1].barh(y, persona_c, color="darkorange")
    axes[1].axvline(0, color="black")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(traits)
    axes[1].set_xlabel("(full) − (mismatched)  on persona coherence")
    axes[1].set_title("Is the correct residual persona-specific?")

    fig.tight_layout()
    save_fig(fig, out_dir / "fig_e2_contrasts.png")


if __name__ == "__main__":
    main()
