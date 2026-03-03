"""Trait dataset: instruction variants + shared questions for contrastive extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from persona_steering.config import PROMPTS_DIR, Trait, TraitConfig, TRAIT_CONFIGS
from persona_steering.utils import log


@dataclass
class TraitInstructionPair:
    """A positive/negative instruction pair for a single variant."""
    positive_instruction: str
    negative_instruction: str
    variant_index: int


@dataclass
class TraitDataset:
    """Full dataset for one trait: instruction variants + shared questions.

    The extraction method uses:
        [persona system prompt variant]
        [trait instruction (pos or neg)]
        User: [shared question]
        Assistant:

    Same question under pos vs neg instruction -> diff isolates trait signal.
    """
    trait: Trait
    positive_label: str
    negative_label: str
    description: str
    instruction_variants: list[TraitInstructionPair]
    questions: list[str]
    eval_prompt: str = ""

    @property
    def n_variants(self) -> int:
        return len(self.instruction_variants)

    @property
    def n_questions(self) -> int:
        return len(self.questions)


def load_trait_dataset(trait: Trait, prompts_dir: Path = PROMPTS_DIR) -> TraitDataset:
    """Load a trait dataset from JSON."""
    path = prompts_dir / f"{trait.value}.json"
    if not path.exists():
        raise FileNotFoundError(f"No dataset file for {trait.value} at {path}")

    with open(path) as f:
        data = json.load(f)

    variants = []
    for i, v in enumerate(data["instruction_variants"]):
        variants.append(TraitInstructionPair(
            positive_instruction=v["positive"],
            negative_instruction=v["negative"],
            variant_index=i,
        ))

    dataset = TraitDataset(
        trait=trait,
        positive_label=data["positive_label"],
        negative_label=data["negative_label"],
        description=data.get("description", ""),
        instruction_variants=variants,
        questions=data["questions"],
        eval_prompt=data.get("eval_prompt", ""),
    )
    log.info("Loaded trait dataset for %s: %d variants, %d questions",
             trait.value, dataset.n_variants, dataset.n_questions)
    return dataset


def load_all_trait_datasets(
    traits: list[Trait] | None = None,
    prompts_dir: Path = PROMPTS_DIR,
) -> dict[Trait, TraitDataset]:
    """Load trait datasets for multiple traits."""
    traits = traits or list(Trait)
    return {t: load_trait_dataset(t, prompts_dir) for t in traits}


def save_trait_dataset(dataset: TraitDataset, prompts_dir: Path = PROMPTS_DIR) -> Path:
    """Save a trait dataset to JSON."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    path = prompts_dir / f"{dataset.trait.value}.json"

    data = {
        "trait": dataset.trait.value,
        "positive_label": dataset.positive_label,
        "negative_label": dataset.negative_label,
        "description": dataset.description,
        "instruction_variants": [
            {
                "positive": v.positive_instruction,
                "negative": v.negative_instruction,
            }
            for v in dataset.instruction_variants
        ],
        "questions": dataset.questions,
        "eval_prompt": dataset.eval_prompt,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    log.info("Saved trait dataset to %s", path)
    return path


def generate_trait_dataset(
    trait: Trait,
    client: object | None = None,
    model: str = "claude-sonnet-4-20250514",
    n_variants: int = 5,
    n_questions: int = 100,
) -> TraitDataset:
    """Generate a trait dataset (instruction variants + questions) via Claude API.

    Args:
        trait: The trait to generate a dataset for.
        client: An anthropic.Anthropic client instance. If None, creates one.
        model: Model to use for generation.
        n_variants: Number of instruction variant pairs.
        n_questions: Number of shared questions.

    Returns:
        TraitDataset with generated content.
    """
    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    tc = TRAIT_CONFIGS[trait]

    # --- Generate instruction variants ---
    variant_prompt = f"""Generate {n_variants} pairs of instruction variants for studying the trait "{trait.value}" in language models.

Each pair should consist of:
- A "positive" instruction that steers a model towards being {tc.positive_label}
- A "negative" instruction that steers a model towards being {tc.negative_label}

The instructions should:
- Be system-prompt-style directives (1-2 sentences each)
- Each pair should use different framing/wording to capture the trait from different angles
- Be applicable to any conversation topic (not topic-specific)

Return as JSON array with objects having "positive" and "negative" keys.
Return ONLY the JSON array, no other text."""

    variant_resp = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": variant_prompt}],
    )
    variant_data = json.loads(variant_resp.content[0].text)

    # --- Generate shared questions ---
    question_prompt = f"""Generate {n_questions} diverse questions that a user might ask in a conversation.

These questions will be used to test how a language model's responses change
when given different trait instructions (specifically: {tc.positive_label} vs {tc.negative_label}).

The questions should:
- Be diverse in topic (health, technology, relationships, career, ethics, science, daily life, etc.)
- Be open-ended enough that responses can vary in {trait.value}
- Be natural-sounding user messages (1-2 sentences)
- NOT be about the trait itself

Return as a JSON array of strings.
Return ONLY the JSON array, no other text."""

    question_resp = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": question_prompt}],
    )
    questions = json.loads(question_resp.content[0].text)

    # --- Generate eval prompt ---
    eval_prompt_text = f"""Rate the following response on a scale from 0 to 1 for the trait "{trait.value}".

0 = maximally {tc.negative_label}
1 = maximally {tc.positive_label}

Consider the overall tone, content choices, and communication style.
Return ONLY a JSON object with keys "score" (float 0-1) and "explanation" (brief string)."""

    # --- Assemble dataset ---
    variants = []
    for i, v in enumerate(variant_data):
        variants.append(TraitInstructionPair(
            positive_instruction=v["positive"],
            negative_instruction=v["negative"],
            variant_index=i,
        ))

    dataset = TraitDataset(
        trait=trait,
        positive_label=tc.positive_label,
        negative_label=tc.negative_label,
        description=f"Contrastive dataset for {trait.value}: {tc.positive_label} vs {tc.negative_label}",
        instruction_variants=variants,
        questions=questions,
        eval_prompt=eval_prompt_text,
    )

    log.info("Generated trait dataset for %s: %d variants, %d questions",
             trait.value, dataset.n_variants, dataset.n_questions)
    return dataset
