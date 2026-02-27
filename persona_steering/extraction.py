"""Steering vector extraction via nnsight."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from persona_steering.config import ModelConfig, PersonaConfig, Trait, VECTORS_DIR
from persona_steering.data import ContrastivePromptPair
from persona_steering.personas import PersonaInducer
from persona_steering.utils import get_device, log, save_pickle, load_pickle


@dataclass
class SteeringVector:
    """A steering vector with full provenance metadata."""
    vector: torch.Tensor          # shape: (hidden_dim,)
    layer: int
    trait: Trait
    persona: str                  # persona slug
    model_name: str
    n_pairs: int                  # number of prompt pairs used
    metadata: dict = field(default_factory=dict)

    @property
    def magnitude(self) -> float:
        return self.vector.norm().item()

    @property
    def direction(self) -> torch.Tensor:
        """Unit vector in the steering direction."""
        return self.vector / self.vector.norm()

    def save(self, path: Path | None = None) -> Path:
        if path is None:
            VECTORS_DIR.mkdir(parents=True, exist_ok=True)
            path = VECTORS_DIR / f"{self.persona}_{self.trait.value}_L{self.layer}.pkl"
        save_pickle(self, path)
        return path

    @staticmethod
    def load(path: Path) -> SteeringVector:
        return load_pickle(path)


class SteeringVectorExtractor:
    """Extract contrastive steering vectors using nnsight.

    Core method: run paired prompts through the model, capture residual
    stream activations, compute the mean difference as the steering vector.
    """

    def __init__(self, model, tokenizer=None, model_config: ModelConfig | None = None):
        """
        Args:
            model: An nnsight LanguageModel instance.
            tokenizer: Optional tokenizer (defaults to model.tokenizer).
            model_config: Model configuration for metadata.
        """
        self.model = model
        self.tokenizer = tokenizer or model.tokenizer
        self.model_config = model_config
        self.inducer = PersonaInducer(model, self.tokenizer)

    def _get_activation(
        self,
        prompt: str,
        layer: int,
    ) -> torch.Tensor:
        """Get mean residual stream activation at a layer for a prompt.

        Returns:
            Tensor of shape (hidden_dim,).
        """
        device = get_device()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with self.model.trace(inputs) as tracer:
            hidden = self.model.model.layers[layer].output[0]
            # Mean over sequence positions, squeeze batch
            mean_act = hidden.mean(dim=1).squeeze(0).save()

        return mean_act.value.detach().cpu()

    def extract_single(
        self,
        persona: PersonaConfig,
        trait: Trait,
        pair: ContrastivePromptPair,
        layer: int,
    ) -> torch.Tensor:
        """Extract a steering vector from a single contrastive pair.

        Returns:
            Difference vector (positive - negative), shape (hidden_dim,).
        """
        pos_prompt = PersonaInducer.from_system_prompt(persona, pair.positive_prompt)
        neg_prompt = PersonaInducer.from_system_prompt(persona, pair.negative_prompt)

        pos_act = self._get_activation(pos_prompt, layer)
        neg_act = self._get_activation(neg_prompt, layer)

        return pos_act - neg_act

    def extract_contrastive_vectors(
        self,
        persona: PersonaConfig,
        trait: Trait,
        prompt_pairs: list[ContrastivePromptPair],
        layers: tuple[int, ...],
    ) -> dict[int, SteeringVector]:
        """Extract steering vectors across layers for a persona-trait combo.

        Computes the mean contrastive difference across all prompt pairs
        for each specified layer.

        Args:
            persona: Persona config for induction.
            trait: The trait being extracted.
            prompt_pairs: Contrastive prompt pairs.
            layers: Layers to extract from.

        Returns:
            Dict mapping layer index to SteeringVector.
        """
        model_name = self.model_config.name if self.model_config else "unknown"
        vectors = {}

        for layer in layers:
            log.info("Extracting layer %d for %s / %s", layer, persona.name, trait.value)
            diffs = []

            for pair in prompt_pairs:
                diff = self.extract_single(persona, trait, pair, layer)
                diffs.append(diff)

            mean_diff = torch.stack(diffs).mean(dim=0)

            vectors[layer] = SteeringVector(
                vector=mean_diff,
                layer=layer,
                trait=trait,
                persona=persona.slug,
                model_name=model_name,
                n_pairs=len(prompt_pairs),
            )

        log.info("Extracted %d layer vectors for %s / %s", len(vectors), persona.name, trait.value)
        return vectors

    def extract_all(
        self,
        personas: list[PersonaConfig],
        traits: list[Trait],
        prompt_pairs: dict[Trait, list[ContrastivePromptPair]],
        layers: tuple[int, ...],
    ) -> dict[str, dict[Trait, dict[int, SteeringVector]]]:
        """Extract vectors for all persona-trait-layer combinations.

        Returns:
            Nested dict: persona_slug -> trait -> layer -> SteeringVector.
        """
        results = {}
        for persona in personas:
            results[persona.slug] = {}
            for trait in traits:
                pairs = prompt_pairs.get(trait, [])
                if not pairs:
                    log.warning("No prompt pairs for trait %s, skipping", trait.value)
                    continue
                results[persona.slug][trait] = self.extract_contrastive_vectors(
                    persona, trait, pairs, layers,
                )
        return results
