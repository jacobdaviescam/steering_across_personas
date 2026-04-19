#!/usr/bin/env python3
"""X3a: Generate neutral-prompt responses under each context.

Used as the substrate for context direction extraction in Task 3.
For each context (10 personas + null + nonsense), generate one response per
neutral prompt with the persona's first system prompt variant. NO trait
instruction is added — context fingerprint is the only signal.

Usage:
    python pipeline/x3a_neutral_generations.py \\
        --model google/gemma-2-27b-it \\
        --output-dir outputs/gemma-2-27b-it/v2/neutral_responses
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from assistant_axis import VLLMGenerator, format_conversation

from persona_steering.config import OUTPUTS_DIR, PERSONA_SLUGS, PROMPTS_DIR
from persona_steering.personas import load_all_personas
from persona_steering.utils import log, model_short_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--neutral-prompts", type=str,
                   default=str(PROMPTS_DIR / "neutral.json"))
    p.add_argument("--contexts", nargs="+", default=PERSONA_SLUGS)
    p.add_argument("--n-prompts", type=int, default=None,
                   help="Limit to first N prompts (default: all)")
    p.add_argument("--variant-index", type=int, default=0,
                   help="Persona system-prompt variant to use (default: 0)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    short = model_short_name(args.model)
    out = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "v2" / "neutral_responses"

    with open(args.neutral_prompts) as f:
        prompts = json.load(f)["questions"]
    if args.n_prompts is not None:
        prompts = prompts[: args.n_prompts]

    all_personas = {p.slug: p for p in load_all_personas()}
    selected = [all_personas[s] for s in args.contexts if s in all_personas]
    missing = set(args.contexts) - {p.slug for p in selected}
    if missing:
        log.warning("Skipping unknown contexts: %s", missing)

    log.info("Plan: %d contexts x %d prompts = %d generations",
             len(selected), len(prompts), len(selected) * len(prompts))
    log.info("Output: %s", out)

    if args.dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)

    gen = VLLMGenerator(
        model_name=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    gen.load()
    tokenizer = gen.llm.get_tokenizer()

    for persona in selected:
        out_file = out / f"{persona.slug}.jsonl"
        if out_file.exists():
            log.info("Skipping %s (exists)", out_file.name)
            continue

        vi = min(args.variant_index, max(0, len(persona.system_prompt_variants) - 1))
        system_content = persona.system_prompt_variants[vi] if persona.system_prompt_variants else ""

        convs = [format_conversation(system_content, q, tokenizer) for q in prompts]
        log.info("Generating %d for %s...", len(convs), persona.slug)
        responses = gen.generate_batch(convs)

        with open(out_file, "w") as f:
            for qi, (conv, resp, q) in enumerate(zip(convs, responses, prompts)):
                full = conv + [{"role": "assistant", "content": resp}]
                f.write(json.dumps({
                    "conversation": full,
                    "context": persona.slug,
                    "prompt": q,
                    "variant_index": 0,  # encoded as v0 for activation extractor compatibility
                    "question_index": qi,
                }) + "\n")
        log.info("Wrote %s", out_file)


if __name__ == "__main__":
    main()
