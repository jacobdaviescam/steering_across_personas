#!/usr/bin/env python3
"""Generate responses for all OLMo training-stage checkpoints.

Iterates over OLMO_TRAINING_STAGES, loading each checkpoint and running
the standard generation pipeline. Outputs are saved under:
    outputs/OLMo-2-1124-7B/{stage_label}/responses/

Usage:
    python pipeline/t1_trajectory_generate.py --dry-run
    python pipeline/t1_trajectory_generate.py
    python pipeline/t1_trajectory_generate.py --stages base sft dpo instruct
    python pipeline/t1_trajectory_generate.py --n-questions 5 --stages base
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from assistant_axis import VLLMGenerator, format_conversation

from persona_steering.config import (
    OLMO_TRAINING_STAGES,
    OUTPUTS_DIR,
    CheckpointSpec,
    Trait,
)
from persona_steering.data import load_all_trait_datasets
from persona_steering.personas import load_all_personas
from persona_steering.utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses across OLMo training stages"
    )
    parser.add_argument(
        "--stages", nargs="+", default=None,
        help="Stage labels to run (default: all). E.g. --stages base sft dpo",
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
        "--n-questions", type=int, default=None,
        help="Number of questions per variant (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=2048,
        help="Max context length for vLLM",
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
        help="Max tokens per response",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without loading models",
    )
    return parser.parse_args()


def model_short_name(hf_id: str) -> str:
    return hf_id.split("/")[-1]


def output_dir_for_stage(spec: CheckpointSpec) -> Path:
    """outputs/{base_model_short}/{stage_label}/responses"""
    base_short = model_short_name(spec.model.hf_id)
    return OUTPUTS_DIR / base_short / spec.stage_label / "responses"


def generate_for_stage(
    spec: CheckpointSpec,
    personas: list,
    traits: list[Trait],
    datasets: dict,
    sampled_questions: dict[str, list[str]],
    args: argparse.Namespace,
) -> None:
    """Run generation for a single training stage checkpoint."""
    output_dir = output_dir_for_stage(spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build jobs
    from collections import defaultdict

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for persona in personas:
        for trait in traits:
            ds = datasets.get(trait)
            if ds is None:
                continue
            questions = sampled_questions[trait.value]
            for variant in ds.instruction_variants:
                for direction in ("pos", "neg"):
                    instruction = (
                        variant.positive_instruction
                        if direction == "pos"
                        else variant.negative_instruction
                    )
                    vi = min(variant.variant_index, len(persona.system_prompt_variants) - 1)
                    system_content = persona.system_prompt_variants[vi] if persona.system_prompt_variants else ""
                    if system_content and instruction:
                        system_content = f"{system_content}\n\n{instruction}"
                    elif instruction:
                        system_content = instruction

                    for qi, question in enumerate(questions):
                        job = {
                            "system_content": system_content,
                            "question": question,
                            "persona": persona.slug,
                            "trait": trait.value,
                            "direction": direction,
                            "variant_index": variant.variant_index,
                            "question_index": qi,
                        }
                        key = (persona.slug, trait.value, direction)
                        groups[key].append(job)

    total_jobs = sum(len(g) for g in groups.values())

    # Check how many are already done
    existing = sum(
        1 for key in groups
        if (output_dir / f"{key[0]}_{key[1]}_{key[2]}.jsonl").exists()
    )
    if existing == len(groups):
        log.info("[%s] All %d files already exist, skipping", spec.stage_label, existing)
        return

    log.info("[%s] %d jobs across %d files (%d already done)",
             spec.stage_label, total_jobs, len(groups), existing)

    if args.dry_run:
        return

    # Load model for this stage
    hf_id = spec.resolved_hf_id
    log.info("[%s] Loading model %s%s...",
             spec.stage_label, hf_id,
             f" (revision={spec.revision})" if spec.revision else "")

    vllm_kwargs = dict(
        model_name=hf_id,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if spec.revision:
        vllm_kwargs["revision"] = spec.revision

    generator = VLLMGenerator(**vllm_kwargs)
    generator.load()
    tokenizer = generator.llm.get_tokenizer()

    for (persona_slug, trait_name, direction), group_jobs in groups.items():
        output_file = output_dir / f"{persona_slug}_{trait_name}_{direction}.jsonl"
        if output_file.exists():
            continue

        conversations = [
            format_conversation(job["system_content"], job["question"], tokenizer)
            for job in group_jobs
        ]

        log.info("[%s] Generating %d responses for %s/%s/%s...",
                 spec.stage_label, len(conversations), persona_slug, trait_name, direction)
        responses = generator.generate_batch(conversations)

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

    # Cleanup model to free GPU memory before loading next checkpoint
    del generator
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    log.info("[%s] Generation complete.", spec.stage_label)


def main() -> None:
    args = parse_args()

    # Select stages
    if args.stages:
        stage_set = set(args.stages)
        stages = [s for s in OLMO_TRAINING_STAGES if s.stage_label in stage_set]
        missing = stage_set - {s.stage_label for s in stages}
        if missing:
            log.error("Unknown stages: %s. Available: %s",
                      missing, [s.stage_label for s in OLMO_TRAINING_STAGES])
            return
    else:
        stages = OLMO_TRAINING_STAGES

    # Load personas
    all_personas = load_all_personas()
    if args.personas:
        slug_set = set(args.personas)
        personas = [p for p in all_personas if p.slug in slug_set]
    else:
        personas = all_personas

    # Load traits
    traits = [Trait(t) for t in args.traits] if args.traits else list(Trait)

    # Load trait datasets and sample questions (shared across all stages)
    datasets = load_all_trait_datasets(traits)
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

    log.info("=== Training Trajectory Generation ===")
    log.info("Stages:   %s", [s.stage_label for s in stages])
    log.info("Personas: %d", len(personas))
    log.info("Traits:   %d", len(traits))

    for spec in stages:
        log.info("--- Stage: %s (%s) ---", spec.stage_label, spec.description)
        generate_for_stage(spec, personas, traits, datasets, sampled_questions, args)

    log.info("=== All stages complete ===")


if __name__ == "__main__":
    main()
