"""Contrastive prompt pair generation and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from persona_steering.config import PROMPTS_DIR, Trait, TraitConfig, TRAIT_CONFIGS
from persona_steering.utils import log


@dataclass
class ContrastivePromptPair:
    """A pair of prompts designed to elicit opposing trait expressions."""
    positive_prompt: str  # elicits the trait (e.g. honest response)
    negative_prompt: str  # elicits the opposite (e.g. deceptive response)
    trait: Trait
    metadata: dict | None = None


def load_prompt_pairs(trait: Trait, prompts_dir: Path = PROMPTS_DIR) -> list[ContrastivePromptPair]:
    """Load contrastive prompt pairs for a trait from JSON."""
    path = prompts_dir / f"{trait.value}.json"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file for {trait.value} at {path}")

    with open(path) as f:
        data = json.load(f)

    pairs = []
    for entry in data["pairs"]:
        pairs.append(ContrastivePromptPair(
            positive_prompt=entry["positive"],
            negative_prompt=entry["negative"],
            trait=trait,
            metadata=entry.get("metadata"),
        ))
    log.info("Loaded %d prompt pairs for %s", len(pairs), trait.value)
    return pairs


def load_all_prompt_pairs(
    traits: list[Trait] | None = None,
    prompts_dir: Path = PROMPTS_DIR,
) -> dict[Trait, list[ContrastivePromptPair]]:
    """Load prompt pairs for multiple traits."""
    traits = traits or list(Trait)
    return {t: load_prompt_pairs(t, prompts_dir) for t in traits}


def save_prompt_pairs(
    pairs: list[ContrastivePromptPair],
    trait: Trait,
    prompts_dir: Path = PROMPTS_DIR,
) -> Path:
    """Save contrastive prompt pairs to JSON."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    path = prompts_dir / f"{trait.value}.json"

    tc = TRAIT_CONFIGS[trait]
    data = {
        "trait": trait.value,
        "positive_label": tc.positive_label,
        "negative_label": tc.negative_label,
        "pairs": [
            {
                "positive": p.positive_prompt,
                "negative": p.negative_prompt,
                **({"metadata": p.metadata} if p.metadata else {}),
            }
            for p in pairs
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    log.info("Saved %d prompt pairs to %s", len(pairs), path)
    return path


def generate_prompt_pairs(
    trait: Trait,
    n: int = 20,
    client: object | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> list[ContrastivePromptPair]:
    """Generate synthetic contrastive prompt pairs via LLM.

    Args:
        trait: The trait to generate pairs for.
        n: Number of pairs to generate.
        client: An anthropic.Anthropic client instance. If None, creates one.
        model: Model to use for generation.

    Returns:
        List of ContrastivePromptPair objects.
    """
    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    tc = TRAIT_CONFIGS[trait]

    prompt = f"""Generate {n} contrastive prompt pairs for studying the trait "{trait.value}" in language models.

Each pair should consist of:
- A "positive" prompt that would naturally elicit a {tc.positive_label} response
- A "negative" prompt that would naturally elicit a {tc.negative_label} response

The prompts should:
- Be diverse in topic and framing
- Differ ONLY in which pole of the trait they elicit
- Be natural-sounding user messages (not artificial)
- Be 1-3 sentences each

Return as JSON array with objects having "positive" and "negative" keys.
Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    data = json.loads(response.content[0].text)

    pairs = [
        ContrastivePromptPair(
            positive_prompt=entry["positive"],
            negative_prompt=entry["negative"],
            trait=trait,
            metadata={"generated": True},
        )
        for entry in data
    ]

    log.info("Generated %d prompt pairs for %s", len(pairs), trait.value)
    return pairs
