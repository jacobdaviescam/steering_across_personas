#!/usr/bin/env python3
"""N1: Generate naturalistic responses (no IV instruction).

For each (persona, trait), sample N trait-relevant questions from the trait
dataset, generate the model's response under each of the persona's system
prompt variants only -- no positive/negative trait instruction is appended.
This is the eval distribution that future-Fig-3 measures the null probe
against: free-form responses where the persona system prompt is the only
manipulation, and the model's natural inclination on the trait surfaces (or
doesn't) as a function of persona.

Output layout (mirrors the existing v2/responses/ pattern):
    {output-dir}/{persona}_{trait}.jsonl
        one JSON object per response, with fields:
          persona, trait, variant_index, question_index,
          question, system_prompt, response.

Usage:
    python pipeline/n1_naturalistic_generate.py \
        --model google/gemma-2-27b-it \
        --output-dir outputs/gemma-2-27b-it/v2/naturalistic/responses \
        --n-questions 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from assistant_axis import VLLMGenerator, format_conversation  # noqa: E402

from persona_steering.config import PERSONA_SLUGS, Trait, OUTPUTS_DIR  # noqa: E402
from persona_steering.data import load_all_trait_datasets  # noqa: E402
from persona_steering.personas import load_all_personas  # noqa: E402
from persona_steering.utils import log, model_short_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--n-questions", type=int, default=10,
                   help="Questions per persona-variant. With 5 variants per "
                        "persona, n=10 yields 50 generations per cell, 4000 "
                        "total across the 80-cell grid.")
    p.add_argument("--personas", nargs="+", default=list(PERSONA_SLUGS))
    p.add_argument("--traits", nargs="+", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir or
                   OUTPUTS_DIR / model_short_name(args.model) / "v2"
                   / "naturalistic" / "responses")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    personas = load_all_personas()
    persona_by_slug = {p.slug: p for p in personas}
    selected_personas = [persona_by_slug[s] for s in args.personas
                         if s in persona_by_slug]
    if not selected_personas:
        log.error("No matching personas in %s", args.personas)
        return

    datasets = load_all_trait_datasets()
    trait_filter = set(args.traits) if args.traits else None

    if args.dry_run:
        log.info("DRY RUN — would generate for personas=%s traits=%s",
                 [p.slug for p in selected_personas], args.traits or "all")
        return

    generator = VLLMGenerator(
        model_name=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size or 1,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    generator.load()
    tokenizer = generator.llm.get_tokenizer()

    for persona in selected_personas:
        for trait_value, dataset in datasets.items():
            if trait_filter and trait_value not in trait_filter:
                continue
            try:
                trait = Trait(trait_value)
            except ValueError:
                continue
            out_path = out_dir / f"{persona.slug}_{trait.value}.jsonl"
            if out_path.exists():
                log.info("Skipping existing %s", out_path)
                continue

            qs = list(dataset.questions)
            sampled = rng.sample(qs, min(args.n_questions, len(qs)))

            convos: list[list[dict]] = []
            meta: list[tuple[int, int, str, str]] = []
            for vi, sys_prompt in enumerate(persona.system_prompt_variants):
                for q in sampled:
                    qi_global = qs.index(q)
                    convos.append(format_conversation(sys_prompt, q, tokenizer))
                    meta.append((vi, qi_global, q, sys_prompt))

            log.info("Generating %d responses for %s/%s",
                     len(convos), persona.slug, trait.value)
            responses = generator.generate_batch(convos)

            with open(out_path, "w") as f:
                for (vi, qi, q, sys_prompt), conv, text in zip(meta, convos, responses):
                    full_conv = conv + [{"role": "assistant", "content": text}]
                    f.write(json.dumps({
                        "persona": persona.slug,
                        "trait": trait.value,
                        "variant_index": vi,
                        "question_index": qi,
                        "question": q,
                        "system_prompt": sys_prompt,
                        "response": text,
                        "conversation": full_conv,
                    }) + "\n")
            log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
