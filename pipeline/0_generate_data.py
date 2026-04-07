#!/usr/bin/env python3
"""Generate trait datasets and persona YAML files via Claude API.

Usage:
    python scripts/generate_data.py --traits                     # generate all trait datasets
    python scripts/generate_data.py --traits assertiveness honesty  # specific traits
    python scripts/generate_data.py --dry-run --traits           # preview without API calls

Outputs:
    data/prompts/<trait>.json  — trait instruction variants + shared questions
"""

from __future__ import annotations

import argparse

from persona_steering.config import Trait, TRAIT_CONFIGS, PROMPTS_DIR
from persona_steering.data import generate_trait_dataset, save_trait_dataset, load_trait_dataset
from persona_steering.utils import log
from persona_steering.wandb_utils import init_run, finish_run, log_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate trait datasets via Claude API")
    parser.add_argument(
        "--traits", nargs="*", default=None,
        help="Trait names to generate (default: all). Pass flag with no args for all traits.",
    )
    parser.add_argument(
        "--n-variants", type=int, default=5,
        help="Number of instruction variant pairs per trait (default: 5)",
    )
    parser.add_argument(
        "--n-questions", type=int, default=100,
        help="Number of shared questions per trait (default: 100)",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine which traits to generate
    if args.traits is None:
        print("No action specified. Use --traits to generate trait datasets.")
        return

    if len(args.traits) == 0:
        traits = list(Trait)
    else:
        traits = [Trait(t) for t in args.traits]

    if args.dry_run:
        print("=== DRY RUN ===\n")
        print(f"Would generate {len(traits)} trait dataset(s):")
        for trait in traits:
            tc = TRAIT_CONFIGS[trait]
            path = PROMPTS_DIR / f"{trait.value}.json"
            exists = path.exists()
            status = "EXISTS (use --force to overwrite)" if exists else "will create"
            print(f"  {trait.value}: {tc.positive_label} vs {tc.negative_label}")
            print(f"    {args.n_variants} instruction variants, {args.n_questions} questions")
            print(f"    -> {path} [{status}]")
        print(f"\nModel: {args.model}")
        print(f"Estimated API calls: {len(traits) * 2} (variants + questions per trait)")
        return

    import anthropic
    client = anthropic.Anthropic()

    init_run("step0_data", "claude", config=vars(args), method="iv")

    for trait in traits:
        path = PROMPTS_DIR / f"{trait.value}.json"
        if path.exists() and not args.force:
            log.info("Skipping %s (file exists, use --force to overwrite)", trait.value)
            continue

        log.info("Generating dataset for %s...", trait.value)
        dataset = generate_trait_dataset(
            trait=trait,
            client=client,
            model=args.model,
            n_variants=args.n_variants,
            n_questions=args.n_questions,
        )
        save_trait_dataset(dataset)
        log.info("Saved %s: %d variants, %d questions",
                 trait.value, dataset.n_variants, dataset.n_questions)
        log_metrics({"data/traits_done": traits.index(trait) + 1, "data/traits_total": len(traits)})

    log.info("Done.")
    finish_run()


if __name__ == "__main__":
    main()
