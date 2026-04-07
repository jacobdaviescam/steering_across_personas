#!/usr/bin/env python3
"""Generate responses via vLLM for all persona x trait x direction combos.

For each persona x trait x direction (pos/neg):
  - Combine persona system prompt variant + trait instruction as system message
  - Use shared questions as user messages
  - Generate responses via VLLMGenerator
  - Save as JSONL

Usage:
    python pipeline/1_generate.py --model google/gemma-2-9b-it
    python pipeline/1_generate.py --model google/gemma-2-9b-it --personas farmer politician --traits assertiveness
    python pipeline/1_generate.py --model google/gemma-2-9b-it --n-questions 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Import assistant_axis from reference checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from assistant_axis import VLLMGenerator, format_conversation

from persona_steering.config import PERSONA_SLUGS, Trait, OUTPUTS_DIR
from persona_steering.data import load_all_trait_datasets
from persona_steering.personas import load_all_personas
from persona_steering.utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses via vLLM for persona x trait x direction combos"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (e.g. google/gemma-2-9b-it)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/{model_short}/responses)",
    )
    parser.add_argument(
        "--n-questions", type=int, default=None,
        help="Number of questions to sample per variant (default: all)",
    )
    parser.add_argument(
        "--personas", nargs="+", default=None,
        help="Persona slugs (default: all)",
    )
    parser.add_argument(
        "--traits", nargs="+", default=None,
        help="Trait names (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for question sampling",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=2048,
        help="Max model context length for vLLM",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens to generate per response",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be generated without loading model",
    )
    return parser.parse_args()


from persona_steering.utils import model_short_name
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_artifact


def main() -> None:
    args = parse_args()

    # Resolve output directory
    short = model_short_name(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "responses"

    # Load personas
    all_personas = load_all_personas()
    if args.personas:
        slug_set = set(args.personas)
        personas = [p for p in all_personas if p.slug in slug_set]
        missing = slug_set - {p.slug for p in personas}
        if missing:
            log.warning("Unknown persona slugs (skipping): %s", missing)
    else:
        personas = all_personas

    # Load traits
    if args.traits:
        traits = [Trait(t) for t in args.traits]
    else:
        traits = list(Trait)

    # Load trait datasets
    datasets = load_all_trait_datasets(traits)

    # Select questions once per trait (same questions across all personas)
    rng = random.Random(args.seed)
    sampled_questions: dict[str, list[str]] = {}
    for trait in traits:
        ds = datasets.get(trait)
        if ds is None:
            continue
        qs = ds.questions
        if args.n_questions is not None and args.n_questions < len(qs):
            qs = rng.sample(qs, args.n_questions)
        sampled_questions[trait.value] = qs

    # Build all generation jobs
    jobs: list[dict] = []
    for persona in personas:
        for trait in traits:
            ds = datasets.get(trait)
            if ds is None:
                log.warning("No dataset for trait %s, skipping", trait.value)
                continue

            questions = sampled_questions[trait.value]

            for variant in ds.instruction_variants:
                for direction in ("pos", "neg"):
                    instruction = (
                        variant.positive_instruction
                        if direction == "pos"
                        else variant.negative_instruction
                    )

                    # Combine persona system prompt + trait instruction
                    vi = min(variant.variant_index, len(persona.system_prompt_variants) - 1)
                    system_content = persona.system_prompt_variants[vi] if persona.system_prompt_variants else ""
                    if system_content and instruction:
                        system_content = f"{system_content}\n\n{instruction}"
                    elif instruction:
                        system_content = instruction

                    for qi, question in enumerate(questions):
                        jobs.append({
                            "system_content": system_content,
                            "question": question,
                            "persona": persona.slug,
                            "trait": trait.value,
                            "direction": direction,
                            "variant_index": variant.variant_index,
                            "question_index": qi,
                        })

    log.info("Generation plan:")
    log.info("  Model:      %s", args.model)
    log.info("  Personas:   %d (%s)", len(personas), [p.slug for p in personas])
    log.info("  Traits:     %d (%s)", len(traits), [t.value for t in traits])
    n_qs = args.n_questions or "all"
    log.info("  Questions:  %s per variant", n_qs)
    log.info("  Total jobs: %d", len(jobs))
    log.info("  Output:     %s", output_dir)

    if args.dry_run:
        print(f"\n=== DRY RUN === Would generate {len(jobs)} responses.")
        return

    # W&B tracking
    init_run("step1_responses", short, config=vars(args))

    # Group jobs by (persona, trait, direction) for output files
    from collections import defaultdict
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for job in jobs:
        key = (job["persona"], job["trait"], job["direction"])
        groups[key].append(job)

    # Initialize vLLM generator
    generator = VLLMGenerator(
        model_name=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size or 1,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    generator.load()
    tokenizer = generator.llm.get_tokenizer()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each group (resume: skip files that already exist)
    skipped = 0
    files_done = 0
    total_files = len(groups)
    for (persona_slug, trait_name, direction), group_jobs in groups.items():
        output_file = output_dir / f"{persona_slug}_{trait_name}_{direction}.jsonl"

        if output_file.exists():
            log.info("Skipping %s (already exists)", output_file.name)
            skipped += len(group_jobs)
            continue

        # Build conversations for this group
        conversations = []
        for job in group_jobs:
            conv = format_conversation(job["system_content"], job["question"], tokenizer)
            conversations.append(conv)

        # Generate batch
        log.info("Generating %d responses for %s/%s/%s...",
                 len(conversations), persona_slug, trait_name, direction)
        responses = generator.generate_batch(conversations)

        # Save as JSONL
        with open(output_file, "w") as f:
            for job, conv, response in zip(group_jobs, conversations, responses):
                full_conv = conv + [{"role": "assistant", "content": response}]
                entry = {
                    "conversation": full_conv,
                    "persona": job["persona"],
                    "trait": job["trait"],
                    "direction": job["direction"],
                    "variant_index": job["variant_index"],
                    "question_index": job["question_index"],
                }
                f.write(json.dumps(entry) + "\n")

        log.info("Saved %d responses to %s", len(responses), output_file)
        files_done += 1
        log_metrics({"responses/files_done": files_done, "responses/files_total": total_files})

    generated = len(jobs) - skipped
    log.info("Done. Generated %d total responses (%d skipped from prior run).",
             generated, skipped)

    if os.environ.get("WANDB_UPLOAD_RESPONSES", "").lower() in ("true", "1", "yes"):
        log_artifact(f"{short}-responses", "responses", output_dir, glob_pattern="*.jsonl")
    finish_run()


if __name__ == "__main__":
    main()
