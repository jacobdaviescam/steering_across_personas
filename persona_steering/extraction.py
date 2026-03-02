"""Steering vector extraction via nnsight."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

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
        norm = self.vector.norm()
        if norm < 1e-10:
            return self.vector
        return self.vector / norm

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

    Optimised to:
    1. Capture all layers in a single forward pass (not one pass per layer).
    2. Process prompts individually but extract all layers at once.

    This reduces forward passes from (n_prompts × n_layers) to just n_prompts.
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

    def _get_activations_all_layers(
        self,
        prompt: str,
        layers: tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        """Get mean residual stream activations at all layers in one forward pass.

        Returns:
            Dict mapping layer index to tensor of shape (hidden_dim,).
        """
        device = get_device()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        saved = {}
        with self.model.trace(inputs):
            for layer in layers:
                hidden = self.model.model.layers[layer].output[0]
                saved[layer] = hidden.mean(dim=1).squeeze(0).save()

        return {layer: saved[layer].value.detach().cpu() for layer in layers}

    def extract_contrastive_vectors(
        self,
        persona: PersonaConfig,
        trait: Trait,
        prompt_pairs: list[ContrastivePromptPair],
        layers: tuple[int, ...],
    ) -> dict[int, SteeringVector]:
        """Extract steering vectors across layers for a persona-trait combo.

        One forward pass per prompt captures all layers simultaneously.
        Total forward passes = 2 × len(prompt_pairs) (one pos, one neg each).

        Args:
            persona: Persona config for induction.
            trait: The trait being extracted.
            prompt_pairs: Contrastive prompt pairs.
            layers: Layers to extract from.

        Returns:
            Dict mapping layer index to SteeringVector.
        """
        model_name = self.model_config.name if self.model_config else "unknown"

        # Accumulate per-layer diffs across all pairs
        layer_diffs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

        log.info("Extracting %s / %s (%d pairs, %d layers, %d forward passes)",
                 persona.name, trait.value, len(prompt_pairs), len(layers),
                 2 * len(prompt_pairs))

        for pair in tqdm(prompt_pairs, desc=f"{persona.slug}/{trait.value}", leave=False):
            pos_prompt = PersonaInducer.from_system_prompt(persona, pair.positive_prompt)
            neg_prompt = PersonaInducer.from_system_prompt(persona, pair.negative_prompt)

            # One forward pass each → all layers captured
            pos_acts = self._get_activations_all_layers(pos_prompt, layers)
            neg_acts = self._get_activations_all_layers(neg_prompt, layers)

            for layer in layers:
                layer_diffs[layer].append(pos_acts[layer] - neg_acts[layer])

        # Average across pairs to get final steering vector per layer
        vectors = {}
        for layer in layers:
            mean_diff = torch.stack(layer_diffs[layer]).mean(dim=0)
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

        Total forward passes = 2 × n_personas × n_traits × n_pairs.
        With 6 personas, 4 traits, 20 pairs: 960 forward passes.
        All layers captured per pass, so this is the minimum possible.

        Returns:
            Nested dict: persona_slug -> trait -> layer -> SteeringVector.
        """
        n_total = sum(
            len(prompt_pairs.get(trait, []))
            for _ in personas
            for trait in traits
        )
        log.info("Starting extraction: %d personas × %d traits = %d forward pass pairs",
                 len(personas), len(traits), n_total)

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
