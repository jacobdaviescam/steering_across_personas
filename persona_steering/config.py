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
    ASSERTIVENESS = "assertiveness"
    EMPATHY = "empathy"
    RISK_TAKING = "risk_taking"
    HONESTY = "honesty"
    CONFIDENCE = "confidence"
    DEFERENCE = "deference"
    WARMTH = "warmth"
    IMPULSIVITY = "impulsivity"


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
    """Configuration for a persona induction.

    Each persona is a concrete character archetype with multiple system
    prompt variants for robust extraction.
    """
    name: str
    slug: str = ""
    system_prompt_variants: list[str] = field(default_factory=list)
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    activation_injection: dict | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.slug:
            object.__setattr__(self, "slug", self.name.lower().replace(" ", "_"))

    @property
    def default_system_prompt(self) -> str:
        """Return the first system prompt variant, or empty string."""
        return self.system_prompt_variants[0] if self.system_prompt_variants else ""


@dataclass(frozen=True)
class TraitConfig:
    """Configuration for a single trait's extraction."""
    trait: Trait
    positive_label: str  # e.g. "assertive"
    negative_label: str  # e.g. "deferential"
    prompt_file: str = ""  # path to dataset JSON

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
    n_questions_per_variant: int = 20
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
    hf_id="google/gemma-2-27b-it",
    num_layers=46,
    hidden_dim=4608,
    default_extraction_layers=(18, 20, 22, 24, 26),
)

# Target layer for single-layer analyses (centre of extraction window,
# matches the layer used by Lu et al. for the assistant axis).
TARGET_LAYER = 22

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
    Trait.ASSERTIVENESS: TraitConfig(Trait.ASSERTIVENESS, "assertive", "deferential"),
    Trait.EMPATHY: TraitConfig(Trait.EMPATHY, "empathetic", "detached"),
    Trait.RISK_TAKING: TraitConfig(Trait.RISK_TAKING, "risk-seeking", "risk-averse"),
    Trait.HONESTY: TraitConfig(Trait.HONESTY, "honest", "deceptive"),
    Trait.CONFIDENCE: TraitConfig(Trait.CONFIDENCE, "confident", "uncertain"),
    Trait.DEFERENCE: TraitConfig(Trait.DEFERENCE, "deferential", "authoritative"),
    Trait.WARMTH: TraitConfig(Trait.WARMTH, "warm", "cold"),
    Trait.IMPULSIVITY: TraitConfig(Trait.IMPULSIVITY, "impulsive", "deliberate"),
}


# ---------------------------------------------------------------------------
# Canonical persona slugs
# ---------------------------------------------------------------------------

PERSONA_SLUGS: list[str] = [
    "farmer",
    "politician",
    "therapist",
    "drill_sergeant",
    "street_hustler",
    "professor",
    "tech_ceo",
    "kindergarten_teacher",
    "surgeon",
    "con_artist",
]
