#!/usr/bin/env python3
"""Extract steering vectors for all persona × trait combinations.

Usage:
    python scripts/extract.py                          # defaults: Gemma 2 9B, all layers
    python scripts/extract.py --model gemma-9b         # explicit model
    python scripts/extract.py --layers 20 21 22        # specific layers only
    python scripts/extract.py --personas base_model neutral full_assistant  # subset
    python scripts/extract.py --traits honesty formality                    # subset

Outputs saved to outputs/vectors/.
"""

from __future__ import annotations

import argparse
import time

import torch
from nnsight import LanguageModel

from persona_steering.config import (
    GEMMA_2_9B,
    GEMMA_2_27B,
    LLAMA_3_70B,
    Trait,
    VECTORS_DIR,
)
from persona_steering.data import load_all_prompt_pairs
from persona_steering.extraction import SteeringVectorExtractor
from persona_steering.personas import load_all_personas
from persona_steering.utils import ensure_output_dirs, save_pickle, get_device, log

MODEL_PRESETS = {
    "gemma-9b": GEMMA_2_9B,
    "gemma-27b": GEMMA_2_27B,
    "llama-70b": LLAMA_3_70B,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract steering vectors")
    parser.add_argument(
        "--model", choices=list(MODEL_PRESETS.keys()), default="gemma-9b",
        help="Model preset (default: gemma-9b)",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Specific layers to extract (default: model's middle third)",
    )
    parser.add_argument(
        "--personas", nargs="+", default=None,
        help="Persona slugs to extract (default: all). E.g. base_model neutral full_assistant",
    )
    parser.add_argument(
        "--traits", nargs="+", default=None,
        help="Trait names to extract (default: all). E.g. honesty sycophancy",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for all_vectors.pkl (default: outputs/vectors/all_vectors.pkl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    # Model config
    config = MODEL_PRESETS[args.model]
    layers = tuple(args.layers) if args.layers else config.extraction_layers

    # Load model
    device = get_device()
    log.info("Loading %s on %s...", config.name, device)
    t0 = time.time()
    model = LanguageModel(config.hf_id, device_map="auto", torch_dtype=torch.float16)
    log.info("Model loaded in %.1fs", time.time() - t0)

    # Load personas
    all_personas = load_all_personas()
    if args.personas:
        slug_set = set(args.personas)
        personas = [p for p in all_personas if p.slug in slug_set]
        missing = slug_set - {p.slug for p in personas}
        if missing:
            log.warning("Unknown persona slugs (skipping): %s", missing)
    else:
        personas = all_personas

    # Load traits
    if args.traits:
        traits = [Trait(t) for t in args.traits]
    else:
        traits = list(Trait)

    # Load prompts
    prompt_pairs = load_all_prompt_pairs(traits)

    # Print summary
    n_passes = 2 * len(personas) * sum(len(prompt_pairs.get(t, [])) for t in traits)
    log.info("Extraction plan:")
    log.info("  Model:    %s (%s)", config.name, config.hf_id)
    log.info("  Personas: %s", [p.slug for p in personas])
    log.info("  Traits:   %s", [t.value for t in traits])
    log.info("  Layers:   %d (%d-%d)", len(layers), min(layers), max(layers))
    log.info("  Pairs:    %s", {t.value: len(prompt_pairs.get(t, [])) for t in traits})
    log.info("  Forward passes: %d", n_passes)

    # Extract
    t0 = time.time()
    extractor = SteeringVectorExtractor(model, model.tokenizer, config)
    all_vectors = extractor.extract_all(personas, traits, prompt_pairs, layers)
    elapsed = time.time() - t0

    # Save
    output_path = args.output or str(VECTORS_DIR / "all_vectors.pkl")
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    save_pickle(all_vectors, output_path)

    # Also save individual vectors
    count = 0
    for persona_slug, trait_dict in all_vectors.items():
        for trait, layer_dict in trait_dict.items():
            for layer, vec in layer_dict.items():
                vec.save()
                count += 1

    log.info("Done in %.1fs. Saved %d vectors to %s", elapsed, count, VECTORS_DIR)

    # Print summary stats
    for persona_slug, trait_dict in all_vectors.items():
        for trait, layer_dict in trait_dict.items():
            mags = [v.magnitude for v in layer_dict.values()]
            log.info("  %s/%s: mag range [%.4f, %.4f]",
                     persona_slug, trait.value, min(mags), max(mags))


if __name__ == "__main__":
    main()
