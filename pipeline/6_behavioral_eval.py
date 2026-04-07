#!/usr/bin/env python3
"""Score existing responses with Claude-as-judge for behavioural evaluation.

For each persona x trait x direction, loads JSONL responses from step 1,
samples N responses, and scores each with LLMJudge.score_trait().
Saves results with resume support (intermediate saves after each combo).

Usage:
    python pipeline/6_behavioral_eval.py \
      --responses-dir outputs/gemma-2-27b-it/responses \
      --output-dir outputs/gemma-2-27b-it/eval \
      --n-samples 20

    # Subset run
    python pipeline/6_behavioral_eval.py \
      --responses-dir outputs/gemma-2-27b-it/responses \
      --personas farmer therapist \
      --traits assertiveness \
      --n-samples 5

    # Dry run
    python pipeline/6_behavioral_eval.py \
      --responses-dir outputs/gemma-2-27b-it/responses \
      --n-samples 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from persona_steering.config import Trait, PERSONA_SLUGS
from persona_steering.evaluation import LLMJudge
from persona_steering.utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score responses with Claude-as-judge"
    )
    parser.add_argument(
        "--responses-dir", type=str, required=True,
        help="Directory containing response JSONL files from step 1",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for eval results (default: sibling 'eval' dir)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Number of responses to sample per persona x trait x direction",
    )
    parser.add_argument(
        "--personas", nargs="+", default=None,
        help="Persona slugs to evaluate (default: all found)",
    )
    parser.add_argument(
        "--traits", nargs="+", default=None,
        help="Trait names to evaluate (default: all found)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Judge model name",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be scored without making API calls",
    )
    return parser.parse_args()


def load_responses(path: Path) -> list[dict]:
    """Load a JSONL response file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_assistant_text(entry: dict) -> str:
    """Extract the assistant response text from a response entry."""
    conv = entry.get("conversation", [])
    for msg in reversed(conv):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def discover_combos(
    responses_dir: Path,
    filter_personas: set[str] | None,
    filter_traits: set[str] | None,
) -> list[tuple[str, str, str]]:
    """Discover all (persona, trait, direction) combos from response files."""
    trait_values = {t.value for t in Trait}
    combos = []

    for path in sorted(responses_dir.glob("*.jsonl")):
        stem = path.stem  # e.g. farmer_assertiveness_pos
        if not stem.endswith(("_pos", "_neg")):
            continue
        direction = stem.rsplit("_", 1)[-1]
        rest = stem.rsplit("_", 1)[0]  # e.g. farmer_assertiveness

        # Parse persona and trait
        persona_slug = None
        trait_name = None
        for tv in trait_values:
            if rest.endswith(f"_{tv}"):
                persona_slug = rest[:-(len(tv) + 1)]
                trait_name = tv
                break

        if persona_slug is None:
            continue
        if filter_personas and persona_slug not in filter_personas:
            continue
        if filter_traits and trait_name not in filter_traits:
            continue

        combos.append((persona_slug, trait_name, direction))

    return combos


def main() -> None:
    args = parse_args()

    responses_dir = Path(args.responses_dir)
    if not responses_dir.exists():
        log.error("Responses directory not found: %s", responses_dir)
        return

    output_dir = Path(args.output_dir) if args.output_dir else responses_dir.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "behavioral_scores.json"

    # Load existing results for resume
    if scores_path.exists():
        with open(scores_path) as f:
            results = json.load(f)
        log.info("Loaded existing results from %s", scores_path)
    else:
        results = {}

    filter_personas = set(args.personas) if args.personas else None
    filter_traits = set(args.traits) if args.traits else None

    combos = discover_combos(responses_dir, filter_personas, filter_traits)
    log.info("Found %d (persona, trait, direction) combos", len(combos))

    if not combos:
        log.error("No response files found in %s", responses_dir)
        return

    rng = random.Random(args.seed)

    # Count work
    todo = []
    for persona, trait, direction in combos:
        key = f"{direction}_scores"
        if persona in results and trait in results[persona] and key in results[persona][trait]:
            continue
        todo.append((persona, trait, direction))

    log.info("Evaluation plan:")
    log.info("  Responses dir: %s", responses_dir)
    log.info("  N samples:     %d per combo", args.n_samples)
    log.info("  Total combos:  %d (%d already done, %d remaining)",
             len(combos), len(combos) - len(todo), len(todo))
    log.info("  Estimated API calls: %d", len(todo) * args.n_samples)

    if args.dry_run:
        print(f"\n=== DRY RUN === Would score {len(todo)} combos "
              f"({len(todo) * args.n_samples} API calls).")
        for persona, trait, direction in todo[:10]:
            print(f"  {persona} / {trait} / {direction}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return

    # Initialize judge
    judge = LLMJudge(model=args.model)
    log.info("Using judge model: %s", args.model)

    scored = 0
    for ci, (persona, trait, direction) in enumerate(todo):
        filename = f"{persona}_{trait}_{direction}.jsonl"
        path = responses_dir / filename

        if not path.exists():
            log.warning("Missing file: %s", path)
            continue

        entries = load_responses(path)
        if not entries:
            log.warning("Empty file: %s", path)
            continue

        # Sample
        if args.n_samples < len(entries):
            sampled = rng.sample(entries, args.n_samples)
        else:
            sampled = entries

        log.info("[%d/%d] Scoring %s/%s/%s (%d samples)...",
                 ci + 1, len(todo), persona, trait, direction, len(sampled))

        scores = []
        for entry in sampled:
            text = extract_assistant_text(entry)
            if not text:
                continue

            trait_enum = Trait(trait)
            result = judge.score_trait(text, trait_enum)
            scores.append(result.score)

        if not scores:
            log.warning("No valid scores for %s/%s/%s", persona, trait, direction)
            continue

        # Store results
        results.setdefault(persona, {}).setdefault(trait, {})
        key = f"{direction}_scores"
        mean_key = f"{direction}_mean"
        results[persona][trait][key] = scores
        results[persona][trait][mean_key] = sum(scores) / len(scores)

        # Compute effect size if both directions are now available
        if "pos_mean" in results[persona][trait] and "neg_mean" in results[persona][trait]:
            results[persona][trait]["effect_size"] = (
                results[persona][trait]["pos_mean"] - results[persona][trait]["neg_mean"]
            )

        scored += 1

        # Intermediate save every 5 combos
        if scored % 5 == 0:
            with open(scores_path, "w") as f:
                json.dump(results, f, indent=2)
            log.info("Intermediate save (%d combos scored)", scored)

    # Final save
    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Done. Scored %d combos. Results saved to %s", scored, scores_path)

    # Summary
    for persona in sorted(results.keys()):
        for trait in sorted(results[persona].keys()):
            entry = results[persona][trait]
            if "effect_size" in entry:
                log.info("  %s/%s: pos=%.2f neg=%.2f effect=%.2f",
                         persona, trait,
                         entry.get("pos_mean", 0), entry.get("neg_mean", 0),
                         entry["effect_size"])


if __name__ == "__main__":
    main()
