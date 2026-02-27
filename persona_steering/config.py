"""Model configs, trait/persona definitions, and experiment parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
PERSONAS_DIR = DATA_DIR / "personas"
OUTPUTS_DIR = ROOT_DIR / "outputs"
VECTORS_DIR = OUTPUTS_DIR / "vectors"
ACTIVATIONS_DIR = OUTPUTS_DIR / "activations"
EVALUATIONS_DIR = OUTPUTS_DIR / "evaluations"
FIGURES_DIR = OUTPUTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------

class Trait(str, Enum):
    """Behavioural traits under investigation."""
    HONESTY = "honesty"
    SYCOPHANCY = "sycophancy"
    VERBOSITY = "verbosity"
    COMPLIANCE = "compliance"
    CONFIDENCE = "confidence"
    FORMALITY = "formality"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a target model."""
    name: str
    hf_id: str
    num_layers: int
    hidden_dim: int
    default_extraction_layers: tuple[int, ...] = ()
    dtype: str = "float16"

    @property
    def mid_layers(self) -> tuple[int, ...]:
        """Return the middle third of layers (common extraction target)."""
        start = self.num_layers // 3
        end = 2 * self.num_layers // 3
        return tuple(range(start, end))

    @property
    def extraction_layers(self) -> tuple[int, ...]:
        return self.default_extraction_layers or self.mid_layers


@dataclass(frozen=True)
class PersonaConfig:
    """Configuration for a persona induction."""
    name: str
    system_prompt: str = ""
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    activation_injection: dict | None = None
    description: str = ""

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "_")


@dataclass(frozen=True)
class TraitConfig:
    """Configuration for a single trait's extraction."""
    trait: Trait
    positive_label: str  # e.g. "honest"
    negative_label: str  # e.g. "deceptive"
    prompt_file: str = ""  # path to contrastive prompts

    def __post_init__(self):
        if not self.prompt_file:
            object.__setattr__(self, "prompt_file", f"{self.trait.value}.json")


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    model: ModelConfig
    personas: list[PersonaConfig]
    traits: list[TraitConfig]
    extraction_layers: tuple[int, ...] | None = None
    n_prompt_pairs: int = 20
    steering_alphas: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    seed: int = 42
    eval_model: str = "claude-sonnet-4-20250514"

    @property
    def layers(self) -> tuple[int, ...]:
        return self.extraction_layers or self.model.extraction_layers


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

GEMMA_2_9B = ModelConfig(
    name="Gemma 2 9B",
    hf_id="google/gemma-2-9b",
    num_layers=42,
    hidden_dim=3584,
    default_extraction_layers=tuple(range(14, 28)),
)

GEMMA_2_27B = ModelConfig(
    name="Gemma 2 27B",
    hf_id="google/gemma-2-27b",
    num_layers=46,
    hidden_dim=4608,
    default_extraction_layers=tuple(range(15, 31)),
)

LLAMA_3_70B = ModelConfig(
    name="Llama 3 70B",
    hf_id="meta-llama/Meta-Llama-3-70B",
    num_layers=80,
    hidden_dim=8192,
    default_extraction_layers=tuple(range(27, 54)),
)


# ---------------------------------------------------------------------------
# Trait presets
# ---------------------------------------------------------------------------

TRAIT_CONFIGS: dict[Trait, TraitConfig] = {
    Trait.HONESTY: TraitConfig(Trait.HONESTY, "honest", "deceptive"),
    Trait.SYCOPHANCY: TraitConfig(Trait.SYCOPHANCY, "sycophantic", "straightforward"),
    Trait.VERBOSITY: TraitConfig(Trait.VERBOSITY, "verbose", "concise"),
    Trait.COMPLIANCE: TraitConfig(Trait.COMPLIANCE, "compliant", "refusing"),
    Trait.CONFIDENCE: TraitConfig(Trait.CONFIDENCE, "confident", "uncertain"),
    Trait.FORMALITY: TraitConfig(Trait.FORMALITY, "formal", "casual"),
}
