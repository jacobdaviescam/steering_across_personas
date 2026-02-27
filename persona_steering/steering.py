"""Applying steering vectors: same-persona and cross-persona."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from persona_steering.config import PersonaConfig, Trait
from persona_steering.extraction import SteeringVector
from persona_steering.personas import PersonaInducer
from persona_steering.utils import get_device, log


@dataclass
class SteeringResult:
    """Result of applying a steering vector."""
    steered_output: str
    baseline_output: str
    persona: str
    trait: Trait
    layer: int
    alpha: float
    vector_source_persona: str  # which persona the vector came from
    metadata: dict = field(default_factory=dict)

    @property
    def is_cross_persona(self) -> bool:
        return self.persona != self.vector_source_persona


def apply_steering(
    model,
    tokenizer,
    prompt: str,
    vector: SteeringVector,
    alpha: float = 1.0,
    max_new_tokens: int = 256,
) -> tuple[str, str]:
    """Apply a steering vector during generation via nnsight.

    Args:
        model: An nnsight LanguageModel instance.
        tokenizer: The model's tokenizer.
        prompt: The input prompt.
        vector: Steering vector to apply.
        alpha: Scaling factor for the vector.
        max_new_tokens: Max tokens to generate.

    Returns:
        Tuple of (steered_output, baseline_output).
    """
    device = get_device()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    vec = vector.vector.to(device)

    # Baseline generation
    with model.trace(inputs, max_new_tokens=max_new_tokens) as tracer:
        baseline_out = model.output.save()
    baseline_text = tokenizer.decode(baseline_out.value[0], skip_special_tokens=True)

    # Steered generation
    with model.trace(inputs, max_new_tokens=max_new_tokens) as tracer:
        hidden = model.model.layers[vector.layer].output[0]
        hidden[:] = hidden + alpha * vec
        steered_out = model.output.save()
    steered_text = tokenizer.decode(steered_out.value[0], skip_special_tokens=True)

    return steered_text, baseline_text


def steer_with_persona(
    model,
    tokenizer,
    persona: PersonaConfig,
    user_message: str,
    vector: SteeringVector,
    alpha: float = 1.0,
    max_new_tokens: int = 256,
) -> SteeringResult:
    """Apply steering under a specific persona.

    Combines persona induction (system prompt) with steering vector injection.
    """
    prompt = PersonaInducer.from_system_prompt(persona, user_message)

    steered, baseline = apply_steering(
        model, tokenizer, prompt, vector, alpha, max_new_tokens,
    )

    return SteeringResult(
        steered_output=steered,
        baseline_output=baseline,
        persona=persona.slug,
        trait=vector.trait,
        layer=vector.layer,
        alpha=alpha,
        vector_source_persona=vector.persona,
    )


def cross_persona_steering_experiment(
    model,
    tokenizer,
    personas: list[PersonaConfig],
    traits: list[Trait],
    vectors: dict[str, dict[Trait, dict[int, SteeringVector]]],
    test_prompts: list[str],
    layer: int,
    alphas: list[float] | None = None,
    max_new_tokens: int = 256,
) -> list[SteeringResult]:
    """Run the full cross-persona steering protocol (Step 3).

    For every (source_persona, target_persona, trait) combination:
    apply the source's vector while the model runs under the target persona.

    Args:
        model: nnsight LanguageModel.
        tokenizer: Tokenizer.
        personas: List of persona configs.
        traits: List of traits to test.
        vectors: Nested dict from extract_all().
        test_prompts: Prompts to steer on.
        layer: Which layer to steer at.
        alphas: Steering strengths to try.
        max_new_tokens: Max generation length.

    Returns:
        List of SteeringResult for all combinations.
    """
    alphas = alphas or [1.0]
    results = []

    for source_persona in personas:
        for target_persona in personas:
            for trait in traits:
                src_vectors = vectors.get(source_persona.slug, {}).get(trait, {})
                vec = src_vectors.get(layer)
                if vec is None:
                    log.warning("No vector for %s/%s at layer %d",
                                source_persona.name, trait.value, layer)
                    continue

                for alpha in alphas:
                    for prompt in test_prompts:
                        result = steer_with_persona(
                            model, tokenizer, target_persona, prompt,
                            vec, alpha, max_new_tokens,
                        )
                        results.append(result)

    log.info("Cross-persona experiment: %d results", len(results))
    return results
