#!/usr/bin/env python3
"""A1: Generate adversarial trait questions per (persona, trait).

For each (persona, trait), use Claude (via OpenRouter) to produce 10
questions that satisfy:
  - The persona's *natural* answer to the question expresses the trait
    in the OPPOSITE direction from the trait's positive label.
  - The question is open-ended and conversational, not a multiple-choice
    moral dilemma.
  - The question does not name the trait or its synonyms.

Examples:
  - con_artist + honesty: scenarios with implicit gain where the natural
    con-artist response is to deceive (i.e., the model's persona-natural
    answer is *low honesty*).
  - therapist + assertiveness: conflict situations where the natural
    therapist response is to validate and de-escalate (low assertiveness).
  - kindergarten_teacher + risk_taking: situations where the natural
    teacher response is to err on the side of caution (low risk).

These adversarial cells are where the null-trained probe is most likely to
miscalibrate, because the persona-natural answer disagrees with the
generic trait label.

Output:
    {output-dir}/{persona}_{trait}.json   (a list of 10 question strings)

Usage:
    OPENROUTER_API_KEY=sk-or-... python pipeline/a1_generate_adversarial_questions.py \
        --output-dir data/prompts/adversarial \
        --n 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_steering.config import PERSONA_SLUGS, Trait, TRAIT_CONFIGS
from persona_steering.openrouter_judge import OpenRouterJudge
from persona_steering.personas import load_all_personas
from persona_steering.utils import log


PROMPT_TEMPLATE = """You are designing an evaluation for a language model.

We want a list of {n} open-ended conversational questions, each chosen so that the speaker described below would *naturally* answer in a way that expresses LOW {trait_pos} (i.e., closer to {trait_neg}). The question itself must NOT use the words {forbidden} or any direct synonym.

Speaker:
{persona_description}

Constraints on the questions:
- Open-ended (no yes/no, no multiple choice).
- Plausibly something a real person might say in conversation.
- Set up a scenario where the speaker's role / values would push them toward a {trait_neg} answer rather than a {trait_pos} one.
- Vary topic and setting across the {n} questions; do not repeat the same scenario type.
- Do NOT mention the speaker's profession, identity, or context inside the question itself - just frame it as a question someone might ask the speaker.

Return ONLY a JSON array of {n} question strings. No commentary, no numbering."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n", type=int, default=10,
                   help="Questions per (persona, trait) cell.")
    p.add_argument("--personas", nargs="+", default=list(PERSONA_SLUGS))
    p.add_argument("--traits", nargs="+", default=None)
    p.add_argument("--model", default="anthropic/claude-sonnet-4.5")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _parse_question_list(text: str) -> list[str] | None:
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list) and all(isinstance(s, str) for s in data):
                return [s.strip() for s in data if s.strip()]
        except json.JSONDecodeError:
            return None
    return None


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    personas = {p.slug: p for p in load_all_personas()
                if p.slug in args.personas}
    traits = args.traits or [t.value for t in Trait]

    judge = OpenRouterJudge(model=args.model)
    try:
        for persona_slug, persona in personas.items():
            for trait_value in traits:
                if persona_slug in {"null", "nonsense"}:
                    continue
                out_path = out_dir / f"{persona_slug}_{trait_value}.json"
                if out_path.exists() and not args.overwrite:
                    log.info("Skipping existing %s", out_path)
                    continue
                tc = TRAIT_CONFIGS[Trait(trait_value)]
                prompt = PROMPT_TEMPLATE.format(
                    n=args.n,
                    trait_pos=tc.positive_label,
                    trait_neg=tc.negative_label,
                    forbidden=f"\"{tc.positive_label}\" or \"{tc.negative_label}\"",
                    persona_description=persona.system_prompt_variants[0],
                )
                raw = judge._chat(prompt, max_tokens=1024)
                questions = _parse_question_list(raw)
                if not questions:
                    log.warning("Failed to parse questions for %s/%s; raw=%r",
                                persona_slug, trait_value, raw[:300])
                    continue
                payload = {
                    "persona": persona_slug,
                    "trait": trait_value,
                    "trait_positive_label": tc.positive_label,
                    "trait_negative_label": tc.negative_label,
                    "questions": questions[:args.n],
                    "generator_prompt_hash": hash(prompt) & 0xFFFFFFFF,
                    "model": args.model,
                }
                out_path.write_text(json.dumps(payload, indent=2))
                log.info("Wrote %s (%d questions)", out_path, len(questions))
    finally:
        judge.close()


if __name__ == "__main__":
    main()
