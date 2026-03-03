"""Steering vector extraction via nnsight."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from persona_steering.config import ModelConfig, PersonaConfig, Trait, VECTORS_DIR
from persona_steering.data import TraitDataset
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
    n_pairs: int                  # total extraction pairs used
    n_variants: int = 0           # number of instruction variants
    n_questions: int = 0          # number of questions per variant
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

    Uses the instruction-variant approach: for each (variant, question) pair,
    runs the question under positive and negative instructions, then diffs.
    All layers are captured in a single forward pass per prompt.
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
        dataset: TraitDataset,
        layers: tuple[int, ...],
        n_questions: int | None = None,
        seed: int = 42,
    ) -> dict[int, SteeringVector]:
        """Extract steering vectors across layers for a persona-trait combo.

        Loops over instruction variants x sampled questions. For each pair,
        formats with persona system prompt variant + instruction + question,
        then diffs positive vs negative activations.

        Args:
            persona: Persona config for induction.
            trait: The trait being extracted.
            dataset: Trait dataset with instruction variants and questions.
            layers: Layers to extract from.
            n_questions: Number of questions to sample per variant (default: all).
            seed: Random seed for question sampling.

        Returns:
            Dict mapping layer index to SteeringVector.
        """
        import random

        model_name = self.model_config.name if self.model_config else "unknown"

        questions = dataset.questions
        if n_questions is not None and n_questions < len(questions):
            rng = random.Random(seed)
            questions = rng.sample(questions, n_questions)

        n_pairs = len(dataset.instruction_variants) * len(questions)
        log.info("Extracting %s / %s (%d variants x %d questions = %d pairs, %d layers)",
                 persona.name, trait.value, dataset.n_variants, len(questions),
                 n_pairs, len(layers))

        layer_diffs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

        for variant in dataset.instruction_variants:
            for question in tqdm(questions,
                                 desc=f"{persona.slug}/{trait.value}/v{variant.variant_index}",
                                 leave=False):
                pos_prompt = PersonaInducer.format_with_instruction(
                    persona, variant.positive_instruction, question,
                    variant_index=variant.variant_index,
                )
                neg_prompt = PersonaInducer.format_with_instruction(
                    persona, variant.negative_instruction, question,
                    variant_index=variant.variant_index,
                )

                pos_acts = self._get_activations_all_layers(pos_prompt, layers)
                neg_acts = self._get_activations_all_layers(neg_prompt, layers)

                for layer in layers:
                    layer_diffs[layer].append(pos_acts[layer] - neg_acts[layer])

        vectors = {}
        for layer in layers:
            mean_diff = torch.stack(layer_diffs[layer]).mean(dim=0)
            vectors[layer] = SteeringVector(
                vector=mean_diff,
                layer=layer,
                trait=trait,
                persona=persona.slug,
                model_name=model_name,
                n_pairs=n_pairs,
                n_variants=dataset.n_variants,
                n_questions=len(questions),
            )

        log.info("Extracted %d layer vectors for %s / %s", len(vectors), persona.name, trait.value)
        return vectors

    def extract_all(
        self,
        personas: list[PersonaConfig],
        traits: list[Trait],
        datasets: dict[Trait, TraitDataset],
        layers: tuple[int, ...],
        n_questions: int | None = None,
        seed: int = 42,
    ) -> dict[str, dict[Trait, dict[int, SteeringVector]]]:
        """Extract vectors for all persona-trait-layer combinations.

        Args:
            personas: List of persona configs.
            traits: List of traits to extract.
            datasets: Dict mapping trait to TraitDataset.
            layers: Layers to extract from.
            n_questions: Questions to sample per variant (default: all).
            seed: Random seed for sampling.

        Returns:
            Nested dict: persona_slug -> trait -> layer -> SteeringVector.
        """
        log.info("Starting extraction: %d personas x %d traits",
                 len(personas), len(traits))

        results = {}
        for persona in personas:
            results[persona.slug] = {}
            for trait in traits:
                dataset = datasets.get(trait)
                if dataset is None:
                    log.warning("No dataset for trait %s, skipping", trait.value)
                    continue
                results[persona.slug][trait] = self.extract_contrastive_vectors(
                    persona, trait, dataset, layers,
                    n_questions=n_questions, seed=seed,
                )
        return results
