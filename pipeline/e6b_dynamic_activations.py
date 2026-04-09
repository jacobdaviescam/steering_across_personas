#!/usr/bin/env python3
"""E6b: Extract activations at multiple token positions during long generation.

Tests the basin attractor hypothesis: as the model generates more tokens under a
persona, its trait representation should drift further from the default direction —
"settling into" the persona-specific basin.

Instead of mean-pooling over the full assistant turn, we extract the residual stream
at specific token positions to track representational drift during generation.

For each persona x trait x direction:
  1. Load generated responses (reuse from step 1, or generate long ones)
  2. Extract activations at specified token positions [50, 100, 200, 400, 800]
  3. Compute contrastive vectors at each position
  4. Save: dict mapping position -> (n_layers, hidden_dim) tensor

Usage:
    python pipeline/e6b_dynamic_activations.py --model google/gemma-2-27b-it
    python pipeline/e6b_dynamic_activations.py --model google/gemma-2-27b-it --positions 50 100 200 400 800
"""

from __future__ import annotations

import argparse
import json
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

from persona_steering.config import (
    BASIN_GRADIENTS,
    BASIN_PERSONA_SLUGS,
    OUTPUTS_DIR,
    TARGET_LAYER,
    Trait,
)
from persona_steering.utils import (
    cosine_similarity,
    get_device,
    log,
    model_short_name,
    save_json,
    parse_persona_trait_from_stem,
)
from persona_steering.wandb_utils import (
    init_run,
    finish_run,
    log_metrics,
    ensure_dir,
)

DEFAULT_POSITIONS = [50, 100, 200, 400, 800]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E6b: Extract activations at multiple token positions"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--responses-dir", type=str, default=None,
        help="Directory containing response JSONL files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for position-wise activation .pt files",
    )
    parser.add_argument(
        "--positions", type=int, nargs="+", default=DEFAULT_POSITIONS,
        help=f"Token positions to extract activations at (default: {DEFAULT_POSITIONS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Batch size (smaller than step 2 since we keep more activations)",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Max sequence length",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto)",
    )
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Layer to extract from (default: {TARGET_LAYER})",
    )
    return parser.parse_args()


def extract_positional_activations(
    jsonl_path: Path,
    extractor: ActivationExtractor,
    encoder: ConversationEncoder,
    span_mapper: SpanMapper,
    positions: list[int],
    layer: int,
    batch_size: int,
    max_length: int,
) -> dict[int, list[torch.Tensor]]:
    """Extract activations at specific token positions within the assistant turn.

    Returns:
        Dict mapping token_position -> list of (hidden_dim,) tensors (one per sample).
        Only includes positions where the response is long enough.
    """
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        return {}

    # Collect per-position activations
    position_activations: dict[int, list[torch.Tensor]] = {p: [] for p in positions}

    for batch_start in range(0, len(entries), batch_size):
        batch_entries = entries[batch_start : batch_start + batch_size]
        conversations = [e["conversation"] for e in batch_entries]

        # Get full-sequence activations at the target layer
        batch_activations, batch_metadata = extractor.batch_conversations(
            conversations, layer=layer, max_length=max_length,
        )

        # Build turn spans to find where the assistant turn starts
        _, batch_spans, span_metadata = encoder.build_batch_turn_spans(conversations)

        for i, entry in enumerate(batch_entries):
            # Find assistant turn start position in the token sequence
            conv_spans = batch_spans[i]
            if not conv_spans:
                continue

            # Last span is the assistant turn
            assistant_span = conv_spans[-1]
            assistant_start = assistant_span[0]
            assistant_end = assistant_span[1]
            assistant_length = assistant_end - assistant_start

            # Extract the full sequence activations for this sample
            # batch_activations shape: (batch, seq_len, hidden_dim) for single layer
            if i >= batch_activations.shape[0]:
                continue
            seq_acts = batch_activations[i]  # (seq_len, hidden_dim)

            for pos in positions:
                # Token position relative to assistant turn start
                abs_pos = assistant_start + pos
                if pos >= assistant_length or abs_pos >= seq_acts.shape[0]:
                    continue  # response not long enough for this position

                # Take a window around the position and mean-pool for stability
                window_start = max(assistant_start, abs_pos - 5)
                window_end = min(abs_pos + 5, assistant_end, seq_acts.shape[0])
                window_acts = seq_acts[window_start:window_end]  # (window, hidden_dim)
                position_activations[pos].append(window_acts.mean(dim=0).cpu().half())

        del batch_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return position_activations


def compute_positional_vectors(
    pos_activations: dict[int, list[torch.Tensor]],
    neg_activations: dict[int, list[torch.Tensor]],
    positions: list[int],
) -> dict[int, torch.Tensor]:
    """Compute contrastive vectors at each token position.

    Returns:
        Dict mapping position -> (hidden_dim,) contrastive vector.
    """
    vectors = {}
    for pos in positions:
        pos_acts = pos_activations.get(pos, [])
        neg_acts = neg_activations.get(pos, [])
        if not pos_acts or not neg_acts:
            continue

        pos_mean = torch.stack(pos_acts).float().mean(dim=0)
        neg_mean = torch.stack(neg_acts).float().mean(dim=0)
        vectors[pos] = pos_mean - neg_mean

    return vectors


def main() -> None:
    args = parse_args()

    short = model_short_name(args.model)
    responses_dir = (
        Path(args.responses_dir) if args.responses_dir
        else OUTPUTS_DIR / short / "responses"
    )
    responses_dir = ensure_dir(f"{short}-responses", responses_dir, "*.jsonl")
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else OUTPUTS_DIR / short / "analysis" / "basin" / "dynamic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = sorted(args.positions)
    layer = args.layer

    init_run("e6b_dynamic_activations", short, config=vars(args))

    # Load model
    log.info("Loading model %s...", args.model)
    device = args.device or str(get_device())
    pm = ProbingModel(args.model, device=device)
    encoder = ConversationEncoder(pm.tokenizer, model_name=args.model)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)
    log.info("Model loaded. Layer %d, positions %s", layer, positions)

    # Determine which persona x trait combos to process
    all_combos = set()
    for trait, gradient in BASIN_GRADIENTS.items():
        for slug, _ring in gradient:
            all_combos.add((slug, trait))

    # Process each combo
    results = {}
    done = 0

    for slug, trait in sorted(all_combos):
        pos_path = responses_dir / f"{slug}_{trait}_pos.jsonl"
        neg_path = responses_dir / f"{slug}_{trait}_neg.jsonl"

        if not pos_path.exists() or not neg_path.exists():
            log.warning("Missing response files for %s/%s, skipping", slug, trait)
            continue

        log.info("Extracting positional activations for %s/%s...", slug, trait)

        pos_acts = extract_positional_activations(
            pos_path, extractor, encoder, span_mapper,
            positions, layer, args.batch_size, args.max_length,
        )
        neg_acts = extract_positional_activations(
            neg_path, extractor, encoder, span_mapper,
            positions, layer, args.batch_size, args.max_length,
        )

        vectors = compute_positional_vectors(pos_acts, neg_acts, positions)

        if vectors:
            out_path = output_dir / f"{slug}_{trait}_positional.pt"
            torch.save({
                "vectors": {pos: v for pos, v in vectors.items()},
                "persona": slug,
                "trait": trait,
                "layer": layer,
                "positions": positions,
                "n_pos_samples": {p: len(pos_acts.get(p, [])) for p in positions},
                "n_neg_samples": {p: len(neg_acts.get(p, [])) for p in positions},
            }, out_path)

            norms = {p: v.norm().item() for p, v in vectors.items()}
            log.info("  Saved %s: positions %s, norms %s",
                     out_path.name, list(vectors.keys()),
                     {p: f"{n:.4f}" for p, n in norms.items()})

            results[f"{slug}_{trait}"] = {
                "positions": list(vectors.keys()),
                "norms": norms,
            }

        done += 1
        log_metrics({"dynamic/done": done, "dynamic/total": len(all_combos)})

    pm.close()

    # Save summary
    save_json(results, output_dir / "dynamic_summary.json")
    log.info("Done. Saved %d positional vector sets to %s", len(results), output_dir)

    finish_run()


if __name__ == "__main__":
    main()
