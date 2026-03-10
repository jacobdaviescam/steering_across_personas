#!/usr/bin/env python3
"""Extract activations for all OLMo training-stage checkpoints.

Iterates over OLMO_TRAINING_STAGES, loading each checkpoint and running
activation extraction on generated responses. Outputs:
    outputs/OLMo-2-1124-7B/{stage_label}/activations/

Usage:
    python pipeline/t2_trajectory_activations.py
    python pipeline/t2_trajectory_activations.py --stages base sft
    python pipeline/t2_trajectory_activations.py --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from assistant_axis.internals import (
    ProbingModel,
    ConversationEncoder,
    ActivationExtractor,
    SpanMapper,
)

from persona_steering.config import (
    OLMO_TRAINING_STAGES,
    OUTPUTS_DIR,
    CheckpointSpec,
)
from persona_steering.utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract activations across OLMo training stages"
    )
    parser.add_argument(
        "--stages", nargs="+", default=None,
        help="Stage labels to run (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Max sequence length",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for model",
    )
    return parser.parse_args()


def model_short_name(hf_id: str) -> str:
    return hf_id.split("/")[-1]


def stage_dirs(spec: CheckpointSpec) -> tuple[Path, Path]:
    """Return (responses_dir, activations_dir) for a stage."""
    base_short = model_short_name(spec.model.hf_id)
    root = OUTPUTS_DIR / base_short / spec.stage_label
    return root / "responses", root / "activations"


def extract_file(
    jsonl_path: Path,
    extractor: ActivationExtractor,
    encoder: ConversationEncoder,
    span_mapper: SpanMapper,
    batch_size: int,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Extract mean assistant-turn activations from a response JSONL file."""
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        return {}

    results = {}

    for batch_start in range(0, len(entries), batch_size):
        batch_entries = entries[batch_start : batch_start + batch_size]
        conversations = [e["conversation"] for e in batch_entries]

        batch_activations, batch_metadata = extractor.batch_conversations(
            conversations, layer=None, max_length=max_length
        )

        _, batch_spans, span_metadata = encoder.build_batch_turn_spans(conversations)

        per_conv_activations = span_mapper.map_spans(
            batch_activations, batch_spans,
            {**batch_metadata, **span_metadata},
        )

        for i, entry in enumerate(batch_entries):
            conv_acts = per_conv_activations[i]
            if conv_acts.numel() == 0:
                log.warning("No activations for v%d_q%d in %s",
                            entry["variant_index"], entry["question_index"], jsonl_path.name)
                continue

            assistant_act = conv_acts[-1]  # (n_layers, hidden_dim)

            key = f"v{entry['variant_index']}_q{entry['question_index']}"
            results[key] = assistant_act.cpu().half()

        del batch_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def run_stage(spec: CheckpointSpec, args: argparse.Namespace) -> None:
    """Extract activations for a single training stage."""
    responses_dir, activations_dir = stage_dirs(spec)

    if not responses_dir.exists():
        log.warning("[%s] No responses dir at %s, skipping", spec.stage_label, responses_dir)
        return

    jsonl_files = sorted(responses_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.warning("[%s] No JSONL files in %s", spec.stage_label, responses_dir)
        return

    # Check how many are already done
    already_done = sum(
        1 for f in jsonl_files if (activations_dir / f"{f.stem}.pt").exists()
    )
    if already_done == len(jsonl_files):
        log.info("[%s] All %d activation files exist, skipping", spec.stage_label, already_done)
        return

    activations_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    hf_id = spec.resolved_hf_id
    log.info("[%s] Loading model %s%s...",
             spec.stage_label, hf_id,
             f" (revision={spec.revision})" if spec.revision else "")

    pm_kwargs = dict(device=args.device)
    if spec.revision:
        pm_kwargs["revision"] = spec.revision

    pm = ProbingModel(hf_id, **pm_kwargs)
    encoder = ConversationEncoder(pm.tokenizer, model_name=hf_id)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)
    log.info("[%s] Model loaded. %d layers, hidden_dim=%d",
             spec.stage_label, len(pm.get_layers()), pm.hidden_size)

    for jsonl_path in tqdm(jsonl_files, desc=f"[{spec.stage_label}] Extracting"):
        output_path = activations_dir / f"{jsonl_path.stem}.pt"
        if output_path.exists():
            continue

        activations = extract_file(
            jsonl_path, extractor, encoder, span_mapper,
            batch_size=args.batch_size, max_length=args.max_length,
        )

        if activations:
            torch.save(activations, output_path)
        else:
            log.warning("[%s] No activations from %s", spec.stage_label, jsonl_path.name)

    pm.close()
    log.info("[%s] Activation extraction complete.", spec.stage_label)


def main() -> None:
    args = parse_args()

    if args.stages:
        stage_set = set(args.stages)
        stages = [s for s in OLMO_TRAINING_STAGES if s.stage_label in stage_set]
    else:
        stages = OLMO_TRAINING_STAGES

    log.info("=== Training Trajectory Activations ===")
    log.info("Stages: %s", [s.stage_label for s in stages])

    for spec in stages:
        log.info("--- Stage: %s (%s) ---", spec.stage_label, spec.description)
        run_stage(spec, args)

    log.info("=== All stages complete ===")


if __name__ == "__main__":
    main()
