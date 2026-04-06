#!/usr/bin/env python3
"""Extract CAA activations across OLMo training-stage checkpoints.

For each checkpoint × persona × trait × direction:
  1. Construct prompt: persona context + A/B scenario + answer token
  2. Forward pass with hooks on all layers (no generation needed)
  3. Extract activation at the answer token position

This replaces the instruction-variant approach (generate + extract) with
CAA (forward-pass only), which works for base/pretrain models that can't
follow instructions.

Outputs:
    outputs/OLMo-2-1124-7B/{stage_label}/caa_activations/

Usage:
    python pipeline/t1_trajectory_activations.py --dry-run
    python pipeline/t1_trajectory_activations.py --stages base instruct
    python pipeline/t1_trajectory_activations.py --stages base --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from assistant_axis.internals import ProbingModel

from persona_steering.config import (
    OLMO_TRAINING_STAGES,
    OUTPUTS_DIR,
    PERSONA_SLUGS,
    CheckpointSpec,
    Trait,
)
from persona_steering.data import load_caa_dataset, CAADataset, CAAQuestion
from persona_steering.personas import load_persona, load_all_personas
from persona_steering.utils import log
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CAA activations across OLMo training stages"
    )
    parser.add_argument(
        "--stages", nargs="+", default=None,
        help="Stage labels to run (default: all)",
    )
    parser.add_argument(
        "--personas", nargs="*", default=None,
        help="Persona slugs (default: all)",
    )
    parser.add_argument(
        "--traits", nargs="*", default=None,
        help="Trait names (default: all)",
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
        help="Preview without loading models",
    )
    return parser.parse_args()


def model_short_name(hf_id: str) -> str:
    return hf_id.split("/")[-1]


def output_dir_for_stage(spec: CheckpointSpec) -> Path:
    base_short = model_short_name(spec.model.hf_id)
    return OUTPUTS_DIR / base_short / spec.stage_label / "caa_activations"


# ---------------------------------------------------------------------------
# Prompt formatting — handles both chat-template and raw-text models
# ---------------------------------------------------------------------------

def format_caa_user_message(q: CAAQuestion) -> str:
    return f"{q.scenario}\n\n(A) {q.option_a}\n(B) {q.option_b}"


def get_answer_letter(q: CAAQuestion, direction: str) -> str:
    if direction == "pos":
        return "A" if q.a_is_positive else "B"
    else:
        return "B" if q.a_is_positive else "A"


def _has_chat_template(tokenizer) -> bool:
    """Check if the tokenizer has a usable chat template."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


def build_prompt_raw(persona_system_prompt: str, user_msg: str, answer_letter: str) -> str:
    """Build a raw-text prompt for models without chat templates (base/pretrain).

    Format:
        Context: {persona}
        Question: {scenario + options}
        Answer: {letter}
    """
    parts = []
    if persona_system_prompt:
        parts.append(f"Context: {persona_system_prompt}\n")
    parts.append(f"Question: {user_msg}\n")
    parts.append(f"Answer: {answer_letter}")
    return "\n".join(parts)


def tokenize_and_find_answer(
    tokenizer,
    persona_system_prompt: str,
    q: CAAQuestion,
    direction: str,
    has_template: bool,
) -> tuple[list[int], int]:
    """Tokenize a CAA prompt and find the answer token position.

    Returns (input_ids, answer_token_position).
    """
    answer_letter = get_answer_letter(q, direction)
    user_msg = format_caa_user_message(q)

    if has_template:
        supports_system = "system" in tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False,
        ).lower() if True else False

        # Always try system prompt first; if template doesn't use it, fall back
        try:
            if supports_system:
                conv_no_asst = [
                    {"role": "system", "content": persona_system_prompt},
                    {"role": "user", "content": user_msg},
                ]
                conv_with_asst = [
                    {"role": "system", "content": persona_system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": answer_letter},
                ]
            else:
                combined = f"{persona_system_prompt}\n\n{user_msg}"
                conv_no_asst = [{"role": "user", "content": combined}]
                conv_with_asst = [
                    {"role": "user", "content": combined},
                    {"role": "assistant", "content": answer_letter},
                ]

            prefix_text = tokenizer.apply_chat_template(
                conv_no_asst, tokenize=False, add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                conv_with_asst, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            # Fall back to raw text
            has_template = False

    if not has_template:
        raw = build_prompt_raw(persona_system_prompt, user_msg, answer_letter)
        # Prefix is everything before the answer letter
        prefix_text = raw[: raw.rfind(answer_letter)]
        full_text = raw

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Find answer token in the assistant/answer portion
    assistant_ids = full_ids[len(prefix_ids):]

    if not assistant_ids:
        # Fallback: last token
        return full_ids, len(full_ids) - 1

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
        # Use first non-special assistant token
        for offset, tid in enumerate(assistant_ids):
            if tid not in special_ids:
                answer_pos = len(prefix_ids) + offset
                break

    if answer_pos is None:
        answer_pos = len(full_ids) - 1

    return full_ids, answer_pos


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_caa_activations(
    pm: ProbingModel,
    persona_system_prompt: str,
    dataset: CAADataset,
    direction: str,
    batch_size: int,
    has_template: bool,
) -> dict[str, torch.Tensor]:
    """Extract answer-token activations for all questions in a CAA dataset."""
    tokenizer = pm.tokenizer
    model = pm.model
    layers = pm.get_layers()
    n_layers = len(layers)

    # Build all inputs
    samples = []
    for q in dataset.questions:
        input_ids, answer_pos = tokenize_and_find_answer(
            tokenizer, persona_system_prompt, q, direction, has_template,
        )
        samples.append({
            "question_id": q.id,
            "input_ids": input_ids,
            "answer_pos": answer_pos,
        })

    results = {}

    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]

        max_len = max(len(s["input_ids"]) for s in batch)
        padded_ids = []
        answer_positions = []

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        for s in batch:
            ids = s["input_ids"]
            pad_len = max_len - len(ids)
            padded = [pad_id] * pad_len + ids  # left-pad
            padded_ids.append(padded)
            answer_positions.append(s["answer_pos"] + pad_len)

        input_tensor = torch.tensor(padded_ids, dtype=torch.long).to(pm.device)
        attention_mask = (input_tensor != pad_id).long()

        # Register hooks
        captured = {}
        hooks = []
        for layer_idx, layer_module in enumerate(layers):
            def make_hook(li):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[li] = h.detach()
                return hook_fn
            hooks.append(layer_module.register_forward_hook(make_hook(layer_idx)))

        with torch.inference_mode():
            model(input_tensor, attention_mask=attention_mask)

        for h in hooks:
            h.remove()

        for i, s in enumerate(batch):
            pos = answer_positions[i]
            act = torch.stack([captured[li][i, pos, :] for li in range(n_layers)])
            results[f"q{s['question_id']}"] = act.cpu().half()

        del captured, input_tensor, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Per-stage runner
# ---------------------------------------------------------------------------

def run_stage(
    spec: CheckpointSpec,
    persona_slugs: list[str],
    traits: list[Trait],
    datasets: dict[Trait, CAADataset],
    args: argparse.Namespace,
) -> None:
    """Extract CAA activations for a single training stage."""
    output_dir = output_dir_for_stage(spec)

    # Build work list
    work = []
    for slug in persona_slugs:
        for trait in traits:
            for direction in ("pos", "neg"):
                path = output_dir / f"{slug}_{trait.value}_{direction}.pt"
                work.append((slug, trait, direction, path))

    remaining = [(s, t, d, p) for s, t, d, p in work if not p.exists()]

    if not remaining:
        log.info("[%s] All %d files exist, skipping", spec.stage_label, len(work))
        return

    log.info("[%s] %d files to extract (%d already done)",
             spec.stage_label, len(remaining), len(work) - len(remaining))

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model — for revision checkpoints, download first then load from local path
    hf_id = spec.resolved_hf_id
    log.info("[%s] Loading model %s%s...",
             spec.stage_label, hf_id,
             f" (revision={spec.revision})" if spec.revision else "")

    model_path = hf_id
    if spec.revision:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(hf_id, revision=spec.revision)
        log.info("[%s] Downloaded checkpoint to %s", spec.stage_label, model_path)

    pm_kwargs = dict(device=args.device)
    pm = ProbingModel(model_path, **pm_kwargs)
    n_layers = len(pm.get_layers())
    log.info("[%s] Model loaded. %d layers, hidden_dim=%d",
             spec.stage_label, n_layers, pm.hidden_size)

    has_template = _has_chat_template(pm.tokenizer)
    log.info("[%s] Chat template: %s", spec.stage_label, "yes" if has_template else "no (using raw prompts)")

    # Cache personas
    persona_cache = {}

    for slug, trait, direction, output_path in tqdm(remaining, desc=f"[{spec.stage_label}]"):
        if slug not in persona_cache:
            persona_cache[slug] = load_persona(slug)

        persona = persona_cache[slug]
        dataset = datasets[trait]

        activations = extract_caa_activations(
            pm=pm,
            persona_system_prompt=persona.default_system_prompt,
            dataset=dataset,
            direction=direction,
            batch_size=args.batch_size,
            has_template=has_template,
        )

        if activations:
            torch.save(activations, output_path)
        else:
            log.warning("[%s] No activations for %s/%s/%s",
                        spec.stage_label, slug, trait.value, direction)

    pm.close()
    log.info("[%s] Done.", spec.stage_label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    persona_slugs = args.personas if args.personas else PERSONA_SLUGS
    traits = [Trait(t) for t in args.traits] if args.traits else list(Trait)

    # Load CAA datasets (shared across all stages — same prompts everywhere)
    datasets: dict[Trait, CAADataset] = {}
    for trait in traits:
        try:
            datasets[trait] = load_caa_dataset(trait)
        except FileNotFoundError:
            log.error("No CAA dataset for %s. Run pipeline/0c_generate_caa_data.py first.", trait.value)
            return

    base_short = model_short_name(stages[0].model.hf_id) if stages else "olmo"
    init_run("t1_trajectory_activations", base_short, config=vars(args), method="caa")

    log.info("=== Training Trajectory CAA Activations ===")
    log.info("Stages:   %s", [s.stage_label for s in stages])
    log.info("Personas: %d", len(persona_slugs))
    log.info("Traits:   %d (%d questions each)",
             len(traits), datasets[traits[0]].n_questions if traits else 0)

    for i, spec in enumerate(stages):
        log.info("--- Stage: %s (%s) ---", spec.stage_label, spec.description)
        run_stage(spec, persona_slugs, traits, datasets, args)
        log_metrics({"trajectory/stages_done": i + 1, "trajectory/stages_total": len(stages)})

    finish_run()
    log.info("=== All stages complete ===")


if __name__ == "__main__":
    main()
