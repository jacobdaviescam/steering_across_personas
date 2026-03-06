#!/usr/bin/env python3
"""Generate CAA-style A/B multiple-choice datasets via Claude API.

Each question presents a scenario with two first-person responses (A and B),
one demonstrating the positive trait, one the negative. The a_is_positive flag
is randomised ~50/50 to prevent positional bias.

Usage:
    python pipeline/0c_generate_caa_data.py --traits                        # all traits
    python pipeline/0c_generate_caa_data.py --traits assertiveness --dry-run
    python pipeline/0c_generate_caa_data.py --traits --n-questions 500
"""

from __future__ import annotations

import argparse
import json
import random

from persona_steering.config import Trait, TRAIT_CONFIGS
from persona_steering.data import (
    CAAQuestion,
    CAADataset,
    CAA_PROMPTS_DIR,
    save_caa_dataset,
    load_caa_dataset,
    _strip_markdown_fences,
)
from persona_steering.utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAA A/B multiple-choice datasets via Claude API"
    )
    parser.add_argument(
        "--traits", nargs="*", default=None,
        help="Trait names to generate (default: all). Pass flag with no args for all traits.",
    )
    parser.add_argument(
        "--n-questions", type=int, default=500,
        help="Total number of A/B questions per trait (default: 500)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Questions to generate per API call (default: 100)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Claude model to use for generation",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be generated without making API calls",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing dataset files",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for a_is_positive randomisation (default: 42)",
    )
    return parser.parse_args()


def generate_caa_batch(
    client: object,
    model: str,
    trait: Trait,
    positive_label: str,
    negative_label: str,
    n_questions: int,
    batch_index: int,
) -> list[dict]:
    """Generate a batch of A/B scenarios via Claude API.

    Returns list of dicts with keys: scenario, option_matching_positive, option_matching_negative.
    """
    prompt = f"""Generate {n_questions} A/B multiple-choice scenarios for studying the trait "{trait.value}" in language models.

Each scenario should:
- Describe a realistic everyday situation (1-2 sentences)
- Provide two first-person response options (A and B)
- One option should demonstrate being {positive_label}, the other being {negative_label}
- Options should be natural-sounding (1-2 sentences each), not cartoonishly extreme
- Scenarios should be diverse: workplace, social, family, health, creative, ethical, etc.
- Options should start with "I" or similar first-person framing

For EACH scenario, always put the {positive_label} option as "option_positive" and the {negative_label} option as "option_negative". The caller will randomise A/B assignment.

This is batch {batch_index + 1}, so ensure scenarios are diverse and don't repeat themes from typical examples.

Return as a JSON array of objects with keys: "scenario", "option_positive", "option_negative".
Return ONLY the JSON array, no other text."""

    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(_strip_markdown_fences(resp.content[0].text))


def generate_caa_dataset(
    trait: Trait,
    client: object | None = None,
    model: str = "claude-sonnet-4-20250514",
    n_questions: int = 500,
    batch_size: int = 100,
    seed: int = 42,
) -> CAADataset:
    """Generate a full CAA dataset for one trait via Claude API."""
    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    tc = TRAIT_CONFIGS[trait]
    rng = random.Random(seed)

    all_raw = []
    n_batches = (n_questions + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_n = min(batch_size, n_questions - len(all_raw))
        log.info("  Generating batch %d/%d (%d questions)...", batch_idx + 1, n_batches, batch_n)

        raw = generate_caa_batch(
            client=client,
            model=model,
            trait=trait,
            positive_label=tc.positive_label,
            negative_label=tc.negative_label,
            n_questions=batch_n,
            batch_index=batch_idx,
        )
        all_raw.extend(raw)

    # Truncate to exact count
    all_raw = all_raw[:n_questions]

    # Build CAAQuestion objects with randomised A/B assignment
    questions = []
    for i, raw in enumerate(all_raw):
        a_is_positive = rng.random() < 0.5
        if a_is_positive:
            option_a = raw["option_positive"]
            option_b = raw["option_negative"]
        else:
            option_a = raw["option_negative"]
            option_b = raw["option_positive"]

        questions.append(CAAQuestion(
            id=i,
            scenario=raw["scenario"],
            option_a=option_a,
            option_b=option_b,
            a_is_positive=a_is_positive,
        ))

    dataset = CAADataset(
        trait=trait,
        positive_label=tc.positive_label,
        negative_label=tc.negative_label,
        questions=questions,
    )
    log.info("Generated CAA dataset for %s: %d questions (%.0f%% a_is_positive)",
             trait.value, dataset.n_questions,
             100 * sum(q.a_is_positive for q in questions) / len(questions))
    return dataset


def main() -> None:
    args = parse_args()

    if args.traits is None:
        print("No action specified. Use --traits to generate CAA datasets.")
        return

    if len(args.traits) == 0:
        traits = list(Trait)
    else:
        traits = [Trait(t) for t in args.traits]

    if args.dry_run:
        print("=== DRY RUN ===\n")
        print(f"Would generate {len(traits)} CAA dataset(s):")
        n_batches = (args.n_questions + args.batch_size - 1) // args.batch_size
        for trait in traits:
            tc = TRAIT_CONFIGS[trait]
            path = CAA_PROMPTS_DIR / f"{trait.value}.json"
            exists = path.exists()
            status = "EXISTS (use --force to overwrite)" if exists else "will create"
            print(f"  {trait.value}: {tc.positive_label} vs {tc.negative_label}")
            print(f"    {args.n_questions} A/B questions ({n_batches} API calls)")
            print(f"    -> {path} [{status}]")
        print(f"\nModel: {args.model}")
        print(f"Total API calls: {len(traits) * n_batches}")
        return

    import anthropic
    client = anthropic.Anthropic()

    for trait in traits:
        path = CAA_PROMPTS_DIR / f"{trait.value}.json"
        if path.exists() and not args.force:
            log.info("Skipping %s (file exists, use --force to overwrite)", trait.value)
            continue

        log.info("Generating CAA dataset for %s...", trait.value)
        dataset = generate_caa_dataset(
            trait=trait,
            client=client,
            model=args.model,
            n_questions=args.n_questions,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        save_caa_dataset(dataset)
        log.info("Saved %s: %d questions", trait.value, dataset.n_questions)

    log.info("Done.")


if __name__ == "__main__":
    main()
