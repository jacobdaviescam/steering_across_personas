#!/usr/bin/env python3
"""Generate steered responses: apply source persona's steering vector to target persona.

For each target persona x trait:
  1. Generate baseline responses (no steering) using 5 sampled questions
  2. For each source persona, apply source's steering vector via ActivationSteering
     during HF generation (same 5 questions)

Uses ProbingModel + ActivationSteering (forward hooks), NOT vLLM.

Usage:
    python pipeline/8_steered_generation.py \
        --model google/gemma-2-27b-it \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --layer 22 --alpha 4.0 --n-questions 5

    # Subset run:
    python pipeline/8_steered_generation.py \
        --model google/gemma-2-27b-it \
        --vectors-dir outputs/gemma-2-27b-it/vectors \
        --source-personas farmer therapist \
        --target-personas farmer therapist \
        --traits assertiveness --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Import assistant_axis from reference checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from persona_steering.config import Trait, OUTPUTS_DIR, TARGET_LAYER, STEERED_RESPONSES_SUBDIR
from persona_steering.data import load_all_trait_datasets
from persona_steering.personas import load_all_personas
from persona_steering.utils import log

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate steered responses (source vector applied to target persona)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (e.g. google/gemma-2-27b-it)",
    )
    parser.add_argument(
        "--vectors-dir", type=str, required=True,
        help="Directory containing steering vector .pt files from step 3",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/{model}/steered_responses)",
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Layer to steer at (default: {TARGET_LAYER})",
    )
    parser.add_argument(
        "--alpha", type=float, default=4.0,
        help="Steering coefficient (default: 4.0)",
    )
    parser.add_argument(
        "--n-questions", type=int, default=5,
        help="Number of questions per combo (default: 5)",
    )
    parser.add_argument(
        "--source-personas", nargs="+", default=None,
        help="Source persona slugs (default: all)",
    )
    parser.add_argument(
        "--target-personas", nargs="+", default=None,
        help="Target persona slugs (default: all)",
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
        "--max-new-tokens", type=int, default=512,
        help="Max tokens to generate per response",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be generated without loading model",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help=(
            "L2-normalise each steering vector before applying, then rescale "
            "by the mean ||v|| across all loaded vectors so --alpha keeps "
            "comparable semantics to the un-normalised default. With this "
            "flag, residue differences across (source, trait) reflect "
            "direction differences, not magnitude differences."
        ),
    )
    return parser.parse_args()


from persona_steering.utils import model_short_name
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_artifact, ensure_dir


def load_steering_vector(pt_path: Path, layer: int) -> torch.Tensor:
    """Load a steering vector .pt file and extract the target layer."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    full_vector = data["vector"]  # (n_layers, hidden_dim)
    if layer >= full_vector.shape[0]:
        raise ValueError(f"Layer {layer} out of range for {pt_path.name} (max {full_vector.shape[0] - 1})")
    return full_vector[layer].float()  # (hidden_dim,)


def find_vector_file(vectors_dir: Path, persona_slug: str, trait_value: str) -> Path | None:
    """Find the .pt vector file for a persona-trait combo."""
    path = vectors_dir / f"{persona_slug}_{trait_value}.pt"
    return path if path.exists() else None


def main() -> None:
    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    short = model_short_name(args.model)
    vectors_dir = ensure_dir(f"{short}-vectors", vectors_dir, "*.pt")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / STEERED_RESPONSES_SUBDIR

    if not args.dry_run:
        init_run("step8_steered_gen", short, config=vars(args))

    # Load personas
    all_personas = load_all_personas()
    persona_map = {p.slug: p for p in all_personas}

    if args.source_personas:
        source_slugs = args.source_personas
    else:
        source_slugs = [p.slug for p in all_personas]

    if args.target_personas:
        target_slugs = args.target_personas
    else:
        target_slugs = [p.slug for p in all_personas]

    # Validate
    for s in source_slugs + target_slugs:
        if s not in persona_map:
            log.error("Unknown persona slug: %s", s)
            return

    # Load traits
    if args.traits:
        traits = [Trait(t) for t in args.traits]
    else:
        traits = list(Trait)

    # Load trait datasets for question sampling
    datasets = load_all_trait_datasets(traits)

    # Sample questions (same set per trait, shared across all combos)
    rng = random.Random(args.seed)
    sampled_questions: dict[str, list[str]] = {}
    for trait in traits:
        ds = datasets.get(trait)
        if ds is None:
            continue
        qs = list(ds.questions)
        if args.n_questions < len(qs):
            qs = rng.sample(qs, args.n_questions)
        sampled_questions[trait.value] = qs

    # Preload steering vectors
    steering_vectors: dict[tuple[str, str], torch.Tensor] = {}
    missing_vectors = []
    for source in source_slugs:
        for trait in traits:
            pt_path = find_vector_file(vectors_dir, source, trait.value)
            if pt_path is not None:
                steering_vectors[(source, trait.value)] = load_steering_vector(pt_path, args.layer)
            else:
                missing_vectors.append(f"{source}_{trait.value}")

    if missing_vectors:
        log.warning("Missing vectors (will skip): %s", missing_vectors)

    if args.normalize and steering_vectors:
        norms = torch.stack([v.float().norm() for v in steering_vectors.values()])
        mean_norm = float(norms.mean())
        log.info(
            "Normalising %d vectors to L2=1 then rescaling by mean ||v||=%.2f "
            "(raw norm range: %.1f - %.1f)",
            len(steering_vectors), mean_norm, float(norms.min()), float(norms.max()),
        )
        for k, v in steering_vectors.items():
            steering_vectors[k] = v / (v.norm() + 1e-8) * mean_norm

    # Count jobs
    n_baseline = 0
    n_steered = 0
    n_skip_baseline = 0
    n_skip_steered = 0

    for target in target_slugs:
        for trait in traits:
            if trait.value not in sampled_questions:
                continue
            baseline_file = output_dir / f"baseline_{target}_{trait.value}.jsonl"
            if baseline_file.exists():
                n_skip_baseline += 1
            else:
                n_baseline += 1

            for source in source_slugs:
                if (source, trait.value) not in steering_vectors:
                    continue
                steered_file = output_dir / f"{source}_{target}_{trait.value}.jsonl"
                if steered_file.exists():
                    n_skip_steered += 1
                else:
                    n_steered += 1

    n_questions = args.n_questions
    total_gens = (n_baseline + n_steered) * n_questions
    total_skip = (n_skip_baseline + n_skip_steered) * n_questions

    log.info("Steered generation plan:")
    log.info("  Model:      %s", args.model)
    log.info("  Layer:      %d, alpha: %.1f", args.layer, args.alpha)
    log.info("  Sources:    %d (%s)", len(source_slugs), source_slugs)
    log.info("  Targets:    %d (%s)", len(target_slugs), target_slugs)
    log.info("  Traits:     %d (%s)", len(traits), [t.value for t in traits])
    log.info("  Questions:  %d per combo", n_questions)
    log.info("  Baseline files:  %d new, %d skip", n_baseline, n_skip_baseline)
    log.info("  Steered files:   %d new, %d skip", n_steered, n_skip_steered)
    log.info("  Total generations: %d (skipping %d)", total_gens, total_skip)

    if args.dry_run:
        print(f"\n=== DRY RUN === Would generate {total_gens} responses ({total_skip} skipped).")
        return

    if total_gens == 0:
        log.info("Nothing to generate (all outputs exist).")
        return

    # Load model
    from assistant_axis.internals.model import ProbingModel
    from assistant_axis.steering import ActivationSteering
    from assistant_axis.generation import generate_response, format_conversation

    log.info("Loading model %s...", args.model)
    probing = ProbingModel(args.model)
    model = probing.model
    tokenizer = probing.tokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate baselines
    log.info("--- Generating baselines ---")
    files_done = 0
    for target in target_slugs:
        persona = persona_map[target]
        system_prompt = persona.default_system_prompt

        for trait in traits:
            if trait.value not in sampled_questions:
                continue

            baseline_file = output_dir / f"baseline_{target}_{trait.value}.jsonl"
            if baseline_file.exists():
                log.info("Skipping baseline %s (exists)", baseline_file.name)
                continue

            questions = sampled_questions[trait.value]
            results = []

            for question in questions:
                conv = format_conversation(system_prompt, question, tokenizer)
                response = generate_response(
                    model, tokenizer, conv,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                results.append({
                    "source_persona": None,
                    "target_persona": target,
                    "trait": trait.value,
                    "question": question,
                    "response": response,
                    "alpha": 0.0,
                    "layer": args.layer,
                })

            with open(baseline_file, "w") as f:
                for entry in results:
                    f.write(json.dumps(entry) + "\n")
            log.info("Saved baseline: %s (%d responses)", baseline_file.name, len(results))
            files_done += 1
            log_metrics({"generation/files_done": files_done, "generation/phase": 0})

    # Generate steered responses
    files_done = 0
    log.info("--- Generating steered responses ---")
    for target in target_slugs:
        persona = persona_map[target]
        system_prompt = persona.default_system_prompt

        for trait in traits:
            if trait.value not in sampled_questions:
                continue
            questions = sampled_questions[trait.value]

            for source in source_slugs:
                if (source, trait.value) not in steering_vectors:
                    continue

                steered_file = output_dir / f"{source}_{target}_{trait.value}.jsonl"
                if steered_file.exists():
                    log.info("Skipping %s (exists)", steered_file.name)
                    continue

                vec = steering_vectors[(source, trait.value)]
                results = []

                for question in questions:
                    conv = format_conversation(system_prompt, question, tokenizer)

                    with ActivationSteering(
                        model,
                        steering_vectors=[vec],
                        coefficients=[args.alpha],
                        layer_indices=[args.layer],
                    ):
                        response = generate_response(
                            model, tokenizer, conv,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                        )

                    results.append({
                        "source_persona": source,
                        "target_persona": target,
                        "trait": trait.value,
                        "question": question,
                        "response": response,
                        "alpha": args.alpha,
                        "layer": args.layer,
                    })

                with open(steered_file, "w") as f:
                    for entry in results:
                        f.write(json.dumps(entry) + "\n")
                log.info("Saved steered: %s (%d responses)", steered_file.name, len(results))
                files_done += 1
                log_metrics({"generation/files_done": files_done, "generation/phase": 1})

    log.info("Done. Steered responses saved to %s", output_dir)

    log_artifact(f"{short}-steered-responses", "steered_responses", output_dir, glob_pattern="*.jsonl")
    finish_run()


if __name__ == "__main__":
    main()
