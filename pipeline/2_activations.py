#!/usr/bin/env python3
"""Extract activations from generated responses using assistant_axis infrastructure.

For each response JSONL file:
  - Load ProbingModel (HuggingFace model with forward hooks)
  - Extract mean assistant-turn activations via ActivationExtractor + SpanMapper
  - Save .pt files with per-sample activation tensors

Usage:
    python pipeline/2_activations.py --model google/gemma-2-9b-it
    python pipeline/2_activations.py --model google/gemma-2-9b-it --responses-dir outputs/gemma-2-9b-it/responses
    python pipeline/2_activations.py --model google/gemma-2-9b-it --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Import assistant_axis from reference checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis-ref"))

from assistant_axis.internals import (
    ProbingModel,
    ConversationEncoder,
    ActivationExtractor,
    SpanMapper,
)

from persona_steering.config import OUTPUTS_DIR
from persona_steering.utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract activations from generated response JSONL files"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (must match generation model)",
    )
    parser.add_argument(
        "--responses-dir", type=str, default=None,
        help="Directory containing response JSONL files (default: outputs/{model}/responses)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for .pt files (default: outputs/{model}/activations)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for activation extraction (default: 4)",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Max sequence length for batching (default: 2048)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for model (default: auto)",
    )
    return parser.parse_args()


from persona_steering.utils import model_short_name
from persona_steering.wandb_utils import init_run, finish_run, log_metrics, log_artifact, ensure_dir


def extract_file(
    jsonl_path: Path,
    extractor: ActivationExtractor,
    encoder: ConversationEncoder,
    span_mapper: SpanMapper,
    batch_size: int,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Extract mean assistant-turn activations from a response JSONL file.

    Returns:
        Dict mapping 'v{variant}_q{question}' -> tensor of shape (n_layers, hidden_dim).
    """
    # Load all conversations and metadata
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        return {}

    results = {}

    # Process in batches
    for batch_start in range(0, len(entries), batch_size):
        batch_entries = entries[batch_start : batch_start + batch_size]
        conversations = [e["conversation"] for e in batch_entries]

        # Extract activations for this batch (all layers)
        batch_activations, batch_metadata = extractor.batch_conversations(
            conversations, layer=None, max_length=max_length
        )

        # Build turn spans for this batch
        _, batch_spans, span_metadata = encoder.build_batch_turn_spans(conversations)

        # Map spans to per-turn mean activations
        per_conv_activations = span_mapper.map_spans(
            batch_activations, batch_spans,
            {**batch_metadata, **span_metadata},
        )

        # Extract the assistant-turn activations (last turn is the assistant response)
        for i, entry in enumerate(batch_entries):
            conv_acts = per_conv_activations[i]
            if conv_acts.numel() == 0:
                log.warning("No activations for v%d_q%d in %s",
                            entry["variant_index"], entry["question_index"], jsonl_path.name)
                continue

            # Take the last turn (assistant response) — shape (n_layers, hidden_dim)
            assistant_act = conv_acts[-1]  # (n_layers, hidden_dim)

            key = f"v{entry['variant_index']}_q{entry['question_index']}"
            results[key] = assistant_act.cpu().half()

        # Free GPU memory
        del batch_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main() -> None:
    args = parse_args()

    short = model_short_name(args.model)
    responses_dir = Path(args.responses_dir) if args.responses_dir else OUTPUTS_DIR / short / "responses"
    responses_dir = ensure_dir(f"{short}-responses", responses_dir, "*.jsonl")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSONL files
    jsonl_files = sorted(responses_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.error("No JSONL files found in %s", responses_dir)
        return

    log.info("Found %d response files in %s", len(jsonl_files), responses_dir)

    # W&B tracking
    init_run("step2_activations", short, config=vars(args))

    # Load model
    log.info("Loading model %s...", args.model)
    pm = ProbingModel(args.model, device=args.device)
    encoder = ConversationEncoder(pm.tokenizer, model_name=args.model)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)
    log.info("Model loaded. %d layers, hidden_dim=%d", len(pm.get_layers()), pm.hidden_size)

    # Process each file
    files_done = 0
    total_files = len(jsonl_files)
    for jsonl_path in tqdm(jsonl_files, desc="Extracting activations"):
        stem = jsonl_path.stem  # e.g. "farmer_assertiveness_pos"
        output_path = output_dir / f"{stem}.pt"

        if output_path.exists():
            log.info("Skipping %s (already exists)", output_path.name)
            files_done += 1
            continue

        log.info("Processing %s...", jsonl_path.name)
        activations = extract_file(
            jsonl_path, extractor, encoder, span_mapper,
            batch_size=args.batch_size, max_length=args.max_length,
        )

        if activations:
            torch.save(activations, output_path)
            log.info("Saved %d activations to %s", len(activations), output_path.name)
        else:
            log.warning("No activations extracted from %s", jsonl_path.name)

        files_done += 1
        log_metrics({"activations/files_done": files_done, "activations/files_total": total_files})

    pm.close()
    log.info("Done.")

    if os.environ.get("WANDB_UPLOAD_ACTIVATIONS", "").lower() in ("true", "1", "yes"):
        log_artifact(f"{short}-activations", "activations", output_dir, glob_pattern="*.pt")
    finish_run()


if __name__ == "__main__":
    main()
