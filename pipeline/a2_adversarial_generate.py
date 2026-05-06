#!/usr/bin/env python3
"""A2: Generate model responses to adversarial questions per persona.

Reads adversarial question sets from a1_generate_adversarial_questions.py and
generates the model's response under each persona's system prompt variants.
The eliciting question itself was constructed so that the persona's natural
answer expresses the trait in the OPPOSITE direction from the trait's
positive label.

Output (mirrors n1_naturalistic_generate's schema so n3/n4 can reuse):
    {output-dir}/{persona}_{trait}.jsonl

Usage:
    python pipeline/a2_adversarial_generate.py \
        --model google/gemma-2-27b-it \
        --questions-dir data/prompts/adversarial \
        --output-dir   outputs/gemma-2-27b-it/v2/adversarial/responses
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from assistant_axis import VLLMGenerator, format_conversation  # noqa: E402

from persona_steering.config import OUTPUTS_DIR  # noqa: E402
from persona_steering.personas import load_all_personas  # noqa: E402
from persona_steering.utils import log, model_short_name  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--questions-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    qdir = Path(args.questions_dir)
    out_dir = Path(args.output_dir or
                   OUTPUTS_DIR / model_short_name(args.model) / "v2"
                   / "adversarial" / "responses")
    out_dir.mkdir(parents=True, exist_ok=True)

    personas = {p.slug: p for p in load_all_personas()}

    generator = VLLMGenerator(
        model_name=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size or 1,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    generator.load()
    tokenizer = generator.llm.get_tokenizer()

    for qfile in sorted(qdir.glob("*.json")):
        payload = json.loads(qfile.read_text())
        persona_slug = payload["persona"]
        trait_value = payload["trait"]
        if persona_slug not in personas:
            log.warning("Persona %s not in registry; skipping", persona_slug)
            continue
        persona = personas[persona_slug]
        questions = payload["questions"]

        out_path = out_dir / f"{persona_slug}_{trait_value}.jsonl"
        if out_path.exists():
            log.info("Skipping existing %s", out_path)
            continue

        convos = []
        meta = []
        for vi, sys_prompt in enumerate(persona.system_prompt_variants):
            for qi, q in enumerate(questions):
                convos.append(format_conversation(sys_prompt, q, tokenizer))
                meta.append((vi, qi, q, sys_prompt))

        log.info("Adversarial: %d generations for %s/%s",
                 len(convos), persona_slug, trait_value)
        responses = generator.generate_batch(convos)

        with open(out_path, "w") as f:
            for (vi, qi, q, sys_prompt), conv, text in zip(meta, convos, responses):
                full_conv = conv + [{"role": "assistant", "content": text}]
                f.write(json.dumps({
                    "persona": persona_slug,
                    "trait": trait_value,
                    "variant_index": vi,
                    "question_index": qi,
                    "question": q,
                    "system_prompt": sys_prompt,
                    "response": text,
                    "conversation": full_conv,
                    "adversarial": True,
                }) + "\n")
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
