"""Persona induction: system prompts, activation injection, few-shot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from persona_steering.config import PERSONAS_DIR, PersonaConfig
from persona_steering.utils import cosine_similarity, get_device, log


@dataclass
class PersonaActivations:
    """Cached activations from a persona-induced model."""
    persona: PersonaConfig
    mean_activations: dict[int, torch.Tensor]  # layer -> mean activation
    prompts_used: int = 0


class PersonaInducer:
    """Induce and manage personas in a language model.

    Works with nnsight LanguageModel instances. Supports three induction
    methods: system prompt, activation injection, and few-shot.
    """

    def __init__(self, model, tokenizer=None):
        """
        Args:
            model: An nnsight LanguageModel instance.
            tokenizer: Optional tokenizer override (uses model.tokenizer by default).
        """
        self.model = model
        self.tokenizer = tokenizer or model.tokenizer
        self._cached_activations: dict[str, PersonaActivations] = {}

    @staticmethod
    def from_system_prompt(
        persona: PersonaConfig,
        user_message: str,
        variant_index: int = 0,
    ) -> str:
        """Format a user message with persona system prompt prepended.

        Args:
            persona: Persona configuration.
            user_message: The user's message.
            variant_index: Which system prompt variant to use (default: 0).

        Returns:
            The full prompt string ready for tokenization.
        """
        parts = []
        if persona.system_prompt_variants:
            idx = min(variant_index, len(persona.system_prompt_variants) - 1)
            parts.append(persona.system_prompt_variants[idx])
        if persona.few_shot_examples:
            for ex in persona.few_shot_examples:
                parts.append(f"User: {ex['user']}\nAssistant: {ex['assistant']}")
        parts.append(f"User: {user_message}\nAssistant:")
        return "\n\n".join(parts)

    @staticmethod
    def format_with_instruction(
        persona: PersonaConfig,
        instruction: str,
        question: str,
        variant_index: int = 0,
    ) -> str:
        """Format a prompt with persona system prompt + trait instruction + question.

        Layered formatting for instruction-variant extraction:
            [persona system prompt variant]
            [trait instruction]
            User: [question]
            Assistant:

        Args:
            persona: Persona configuration.
            instruction: Trait instruction (positive or negative).
            question: Shared question from the trait dataset.
            variant_index: Which system prompt variant to use.

        Returns:
            The full prompt string ready for tokenization.
        """
        parts = []
        if persona.system_prompt_variants:
            idx = min(variant_index, len(persona.system_prompt_variants) - 1)
            parts.append(persona.system_prompt_variants[idx])
        parts.append(instruction)
        parts.append(f"User: {question}\nAssistant:")
        return "\n\n".join(parts)

    @staticmethod
    def from_few_shot(persona: PersonaConfig, user_message: str) -> str:
        """Format a prompt with few-shot examples only (no system prompt)."""
        parts = []
        for ex in persona.few_shot_examples:
            parts.append(f"User: {ex['user']}\nAssistant: {ex['assistant']}")
        parts.append(f"User: {user_message}\nAssistant:")
        return "\n\n".join(parts)

    def from_activation_injection(
        self,
        persona: PersonaConfig,
        user_message: str,
        layers: tuple[int, ...],
        alpha: float = 1.0,
    ) -> tuple[str, dict[int, torch.Tensor]]:
        """Prepare prompt and activation offsets for injection.

        Returns:
            Tuple of (formatted prompt, dict of layer -> offset tensor).
        """
        if persona.activation_injection is None:
            raise ValueError(f"Persona {persona.name} has no activation injection config")

        prompt = f"User: {user_message}\nAssistant:"

        offsets = {}
        injection = persona.activation_injection
        vector_path = Path(injection["vector_path"])
        if vector_path.exists():
            stored = torch.load(vector_path, map_location=get_device(), weights_only=True)
            for layer in layers:
                if layer in stored:
                    offsets[layer] = stored[layer] * alpha

        return prompt, offsets

    def collect_activations(
        self,
        persona: PersonaConfig,
        prompts: list[str],
        layers: tuple[int, ...],
    ) -> PersonaActivations:
        """Collect mean activations for a persona across prompts.

        Uses nnsight tracing to capture all layers in a single forward pass
        per prompt. Total forward passes = len(prompts).
        """
        device = get_device()
        all_activations: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

        for prompt_text in prompts:
            full_prompt = self.from_system_prompt(persona, prompt_text)
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(device)

            saved = {}
            with self.model.trace(inputs):
                for layer in layers:
                    hidden = self.model.model.layers[layer].output[0]
                    saved[layer] = hidden.mean(dim=1).squeeze(0).save()

            for layer in layers:
                all_activations[layer].append(saved[layer].value.detach().cpu())

        # Average across prompts
        mean_acts = {}
        for layer in layers:
            stacked = torch.stack(all_activations[layer])
            mean_acts[layer] = stacked.mean(dim=0)

        result = PersonaActivations(
            persona=persona,
            mean_activations=mean_acts,
            prompts_used=len(prompts),
        )
        self._cached_activations[persona.slug] = result
        log.info("Collected activations for persona '%s' (%d prompts, %d layers)",
                 persona.name, len(prompts), len(layers))
        return result

    def validate_persona(
        self,
        persona_a: PersonaConfig,
        persona_b: PersonaConfig,
        layers: tuple[int, ...],
    ) -> dict[int, float]:
        """Validate that two personas produce distinct activation signatures.

        Returns per-layer cosine similarity between mean activations.
        Lower similarity = more distinct personas.
        """
        a = self._cached_activations.get(persona_a.slug)
        b = self._cached_activations.get(persona_b.slug)

        if a is None or b is None:
            raise ValueError("Must collect activations before validating. "
                             "Call collect_activations() for both personas first.")

        sims = {}
        for layer in layers:
            if layer in a.mean_activations and layer in b.mean_activations:
                sims[layer] = cosine_similarity(
                    a.mean_activations[layer],
                    b.mean_activations[layer],
                )

        log.info("Persona similarity '%s' vs '%s': mean=%.4f",
                 persona_a.name, persona_b.name,
                 sum(sims.values()) / len(sims) if sims else 0.0)
        return sims


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_persona(name: str, personas_dir: Path = PERSONAS_DIR) -> PersonaConfig:
    """Load a persona config from YAML."""
    path = personas_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No persona config at {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return PersonaConfig(
        name=data["name"],
        slug=data.get("slug", name),
        system_prompt_variants=data.get("system_prompt_variants", []),
        few_shot_examples=data.get("few_shot_examples", []),
        activation_injection=data.get("activation_injection"),
        description=data.get("description", ""),
        tags=data.get("tags", []),
    )


def load_all_personas(personas_dir: Path = PERSONAS_DIR) -> list[PersonaConfig]:
    """Load all persona configs from a directory."""
    configs = []
    for path in sorted(personas_dir.glob("*.yaml")):
        configs.append(load_persona(path.stem, personas_dir))
    return configs
