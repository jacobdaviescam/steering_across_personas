#!/usr/bin/env python3
"""Extract activations at the answer token position for CAA-style A/B prompts.

For each persona x trait x direction (pos/neg):
  1. Construct conversation: [system: persona, user: A/B question, assistant: answer_letter]
  2. Forward pass (no generation) with hooks on all layers
  3. Extract activation at the answer token position only -> (n_layers, hidden_dim)

Usage:
    python pipeline/2c_caa_activations.py --model google/gemma-2-27b-it
    python pipeline/2c_caa_activations.py --model google/gemma-2-27b-it \
        --personas farmer politician --traits assertiveness --batch-size 16
    python pipeline/2c_caa_activations.py --model google/gemma-2-27b-it --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Import assistant_axis from reference checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from assistant_axis.internals import ProbingModel

from persona_steering.config import Trait, TRAIT_CONFIGS, OUTPUTS_DIR, PERSONA_SLUGS
from persona_steering.data import load_caa_dataset, CAADataset, CAAQuestion
from persona_steering.personas import load_persona, load_all_personas
from persona_steering.utils import get_device, log, model_short_name
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CAA answer-token activations for persona x trait combos"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (e.g. google/gemma-2-27b-it)",
    )
    parser.add_argument(
        "--personas", nargs="*", default=None,
        help="Persona slugs to process (default: all)",
    )
    parser.add_argument(
        "--traits", nargs="*", default=None,
        help="Trait names to process (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/{model}/caa_activations)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for forward passes (default: 16)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for model (default: auto)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be extracted without loading model",
    )
    return parser.parse_args()


def format_caa_user_message(q: CAAQuestion) -> str:
    """Format a CAA question as a user message."""
    return f"{q.scenario}\n\n(A) {q.option_a}\n(B) {q.option_b}"


def get_answer_letter(q: CAAQuestion, direction: str) -> str:
    """Get the answer letter for a question given the direction.

    For direction="pos": return the letter corresponding to the positive option.
    For direction="neg": return the letter corresponding to the negative option.
    """
    if direction == "pos":
        return "A" if q.a_is_positive else "B"
    else:
        return "B" if q.a_is_positive else "A"


def find_answer_token_position(
    tokenizer,
    conversation_without_assistant: list[dict[str, str]],
    conversation_with_assistant: list[dict[str, str]],
    answer_letter: str,
) -> int:
    """Find the token position of the answer letter using incremental-prefix method.

    1. Tokenize without assistant turn (add_generation_prompt=True) -> prefix_ids
    2. Tokenize with assistant turn (add_generation_prompt=False) -> full_ids
    3. Answer token = last content token in full_ids[len(prefix_ids):]
    4. Verify by decoding that token contains "A" or "B"

    Returns the absolute token position in the full sequence.
    """
    prefix_text = tokenizer.apply_chat_template(
        conversation_without_assistant,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        conversation_with_assistant,
        tokenize=False,
        add_generation_prompt=False,
    )

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # The assistant turn tokens are full_ids[len(prefix_ids):]
    assistant_ids = full_ids[len(prefix_ids):]

    if not assistant_ids:
        raise ValueError(f"No assistant tokens found. prefix={len(prefix_ids)}, full={len(full_ids)}")

    # Find the token containing the answer letter (scan from end, skip special tokens)
    special_ids = set(tokenizer.all_special_ids)
    answer_pos = None
    for offset in range(len(assistant_ids) - 1, -1, -1):
        tid = assistant_ids[offset]
        if tid in special_ids:
            continue
        decoded = tokenizer.decode([tid])
        if answer_letter in decoded:
            answer_pos = len(prefix_ids) + offset
            break

    if answer_pos is None:
        # Fallback: use the first non-special assistant token
        for offset, tid in enumerate(assistant_ids):
            if tid not in special_ids:
                answer_pos = len(prefix_ids) + offset
                decoded = tokenizer.decode([tid])
                log.warning(
                    "Could not find '%s' in assistant tokens, using first content token: '%s'",
                    answer_letter, decoded,
                )
                break

    if answer_pos is None:
        raise ValueError(f"No valid answer token found for '{answer_letter}'")

    return answer_pos


def extract_caa_activations(
    pm: ProbingModel,
    persona_system_prompt: str,
    dataset: CAADataset,
    direction: str,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Extract answer-token activations for all questions in a CAA dataset.

    Args:
        pm: ProbingModel with loaded model and tokenizer
        persona_system_prompt: System prompt for the persona
        dataset: CAADataset with A/B questions
        direction: "pos" or "neg"
        batch_size: Batch size for forward passes

    Returns:
        Dict mapping "q{id}" -> tensor of shape (n_layers, hidden_dim) in float16.
    """
    tokenizer = pm.tokenizer
    model = pm.model
    layers = pm.get_layers()
    n_layers = len(layers)
    supports_system = pm.supports_system_prompt()

    # Build all conversations and find answer positions
    samples = []
    for q in dataset.questions:
        answer_letter = get_answer_letter(q, direction)
        user_msg = format_caa_user_message(q)

        if supports_system:
            conv_no_assistant = [
                {"role": "system", "content": persona_system_prompt},
                {"role": "user", "content": user_msg},
            ]
            conv_with_assistant = [
                {"role": "system", "content": persona_system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": answer_letter},
            ]
        else:
            # For models without system prompt support (e.g. Gemma 2),
            # prepend persona context to the user message
            combined_user = f"{persona_system_prompt}\n\n{user_msg}"
            conv_no_assistant = [
                {"role": "user", "content": combined_user},
            ]
            conv_with_assistant = [
                {"role": "user", "content": combined_user},
                {"role": "assistant", "content": answer_letter},
            ]

        answer_pos = find_answer_token_position(
            tokenizer, conv_no_assistant, conv_with_assistant, answer_letter,
        )

        # Tokenize the full conversation for the forward pass
        full_text = tokenizer.apply_chat_template(
            conv_with_assistant, tokenize=False, add_generation_prompt=False,
        )
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        samples.append({
            "question_id": q.id,
            "input_ids": full_ids,
            "answer_pos": answer_pos,
        })

    # Process in batches
    results = {}

    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]

        # Pad to same length (left padding)
        max_len = max(len(s["input_ids"]) for s in batch)
        padded_ids = []
        answer_positions = []

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        for s in batch:
            ids = s["input_ids"]
            pad_len = max_len - len(ids)
            padded = [pad_id] * pad_len + ids
            padded_ids.append(padded)
            # Adjust answer position for left padding
            answer_positions.append(s["answer_pos"] + pad_len)

        input_tensor = torch.tensor(padded_ids, dtype=torch.long).to(pm.device)
        attention_mask = (input_tensor != pad_id).long()

        # Register hooks on all layers
        captured = {}
        hooks = []

        for layer_idx, layer_module in enumerate(layers):
            def make_hook(li):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[li] = h.detach()
                return hook_fn
            hooks.append(layer_module.register_forward_hook(make_hook(layer_idx)))

        # Forward pass
        with torch.inference_mode():
            model(input_tensor, attention_mask=attention_mask)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Extract answer-token activations
        for i, s in enumerate(batch):
            pos = answer_positions[i]
            # Stack across layers: (n_layers, hidden_dim)
            act = torch.stack([captured[li][i, pos, :] for li in range(n_layers)])
            key = f"q{s['question_id']}"
            results[key] = act.cpu().half()

        # Free memory (clear the closure dict in place to avoid use-after-del)
        captured.clear()
        del input_tensor, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main() -> None:
    args = parse_args()

    short = model_short_name(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "caa_activations"

    # Determine personas
    if args.personas is None:
        persona_slugs = PERSONA_SLUGS
    else:
        persona_slugs = args.personas

    # Determine traits
    if args.traits is None:
        traits = list(Trait)
    else:
        traits = [Trait(t) for t in args.traits]

    # Load CAA datasets
    datasets = {}
    for trait in traits:
        try:
            datasets[trait] = load_caa_dataset(trait)
        except FileNotFoundError:
            log.error("No CAA dataset for %s. Run pipeline/0c_generate_caa_data.py first.", trait.value)
            return

    # Build work list
    work = []
    for persona_slug in persona_slugs:
        for trait in traits:
            for direction in ("pos", "neg"):
                output_path = output_dir / f"{persona_slug}_{trait.value}_{direction}.pt"
                work.append((persona_slug, trait, direction, output_path))

    # Filter already-done
    remaining = [(p, t, d, o) for p, t, d, o in work if not o.exists()]

    if args.dry_run:
        print("=== DRY RUN ===\n")
        print(f"Model: {args.model}")
        print(f"Output: {output_dir}")
        print(f"Personas: {len(persona_slugs)}")
        print(f"Traits: {len(traits)}")
        print(f"Questions per trait: {[datasets[t].n_questions for t in traits]}")
        print(f"Total files: {len(work)} ({len(remaining)} remaining, {len(work) - len(remaining)} already done)")
        total_fwd = sum(datasets[t].n_questions for _, t, _, _ in remaining)
        print(f"Total forward passes: {total_fwd}")
        print(f"Batches (batch_size={args.batch_size}): {(total_fwd + args.batch_size - 1) // args.batch_size}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if not remaining:
        log.info("All files already exist. Nothing to do.")
        return

    init_run("step2c_caa_activations", short, config=vars(args), method="caa")

    # Load model
    log.info("Loading model %s...", args.model)
    device = args.device or str(get_device())
    pm = ProbingModel(args.model, device=device)
    n_layers = len(pm.get_layers())
    log.info("Model loaded. %d layers, hidden_dim=%d", n_layers, pm.hidden_size)

    # Load all needed personas
    persona_cache = {}

    done = 0
    for persona_slug, trait, direction, output_path in tqdm(remaining, desc="CAA extraction"):
        # Load persona if not cached
        if persona_slug not in persona_cache:
            persona_cache[persona_slug] = load_persona(persona_slug)

        persona = persona_cache[persona_slug]
        dataset = datasets[trait]

        log.info("Extracting %s/%s/%s (%d questions)...",
                 persona_slug, trait.value, direction, dataset.n_questions)

        activations = extract_caa_activations(
            pm=pm,
            persona_system_prompt=persona.default_system_prompt,
            dataset=dataset,
            direction=direction,
            batch_size=args.batch_size,
        )

        if activations:
            torch.save(activations, output_path)
            log.info("Saved %d activations to %s", len(activations), output_path.name)
        else:
            log.warning("No activations extracted for %s/%s/%s", persona_slug, trait.value, direction)

        done += 1
        log_metrics({"caa_activations/done": done, "caa_activations/total": len(remaining)})

    pm.close()
    log_artifact(f"{short}-caa-activations", "caa_activations", output_dir, glob_pattern="*.pt")
    finish_run()
    log.info("Done.")


if __name__ == "__main__":
    main()
