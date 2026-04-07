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

# Output subdirectory names
RESPONSES_SUBDIR = "responses"
ACTIVATIONS_SUBDIR = "activations"
VECTORS_SUBDIR = "vectors"
ANALYSIS_SUBDIR = "analysis"
EVAL_SUBDIR = "eval"
FIGURES_SUBDIR = "figures"
STEERED_RESPONSES_SUBDIR = "steered_responses"
CAA_ACTIVATIONS_SUBDIR = "caa_activations"
CAA_VECTORS_SUBDIR = "caa_vectors"
CAA_ANALYSIS_SUBDIR = "caa_analysis"


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
    # Safety-relevant traits (R5)
    REFUSAL = "refusal"
    DECEPTIVENESS = "deceptiveness"
    POWER_SEEKING = "power_seeking"
    SYCOPHANCY = "sycophancy"


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

OLMO_2_7B = ModelConfig(
    name="OLMo 2 7B",
    hf_id="allenai/OLMo-2-1124-7B",
    num_layers=32,
    hidden_dim=4096,
    default_extraction_layers=tuple(range(10, 22)),
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
    Trait.ASSERTIVENESS: TraitConfig(Trait.ASSERTIVENESS, "assertive", "deferential"),
    Trait.EMPATHY: TraitConfig(Trait.EMPATHY, "empathetic", "detached"),
    Trait.RISK_TAKING: TraitConfig(Trait.RISK_TAKING, "risk-seeking", "risk-averse"),
    Trait.HONESTY: TraitConfig(Trait.HONESTY, "honest", "deceptive"),
    Trait.CONFIDENCE: TraitConfig(Trait.CONFIDENCE, "confident", "uncertain"),
    Trait.DEFERENCE: TraitConfig(Trait.DEFERENCE, "deferential", "authoritative"),
    Trait.WARMTH: TraitConfig(Trait.WARMTH, "warm", "cold"),
    Trait.IMPULSIVITY: TraitConfig(Trait.IMPULSIVITY, "impulsive", "deliberate"),
    # Safety-relevant traits (R5)
    Trait.REFUSAL: TraitConfig(Trait.REFUSAL, "refuses inappropriate requests", "always compliant"),
    Trait.DECEPTIVENESS: TraitConfig(Trait.DECEPTIVENESS, "actively misleading", "straightforward"),
    Trait.POWER_SEEKING: TraitConfig(Trait.POWER_SEEKING, "seeks control and influence", "defers to others"),
    Trait.SYCOPHANCY: TraitConfig(Trait.SYCOPHANCY, "agrees with user regardless", "independent judgment"),
}


# ---------------------------------------------------------------------------
# Canonical persona slugs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Training trajectory (OLMo checkpoints)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointSpec:
    """A specific model checkpoint for the training trajectory experiment."""
    model: ModelConfig
    stage_label: str  # short label for output dirs (e.g. "base", "sft", "dpo")
    hf_id: str = ""  # override model.hf_id (for post-training variants)
    revision: str = ""  # HuggingFace branch/revision (for pretraining checkpoints)
    description: str = ""

    @property
    def resolved_hf_id(self) -> str:
        return self.hf_id or self.model.hf_id

    @property
    def output_subdir(self) -> str:
        """Subdirectory name under the model's output folder."""
        return self.stage_label


# OLMo 2 7B training stages — from early pretraining through RLVR.
# Pretraining checkpoints use `revision=` on the base model repo.
# Post-training stages are separate HF repos.
OLMO_TRAINING_STAGES: list[CheckpointSpec] = [
    CheckpointSpec(
        model=OLMO_2_7B,
        stage_label="pretrain_1pct",
        revision="stage1-step10000-tokens42B",
        description="~1% of pretraining (42B / 3.9T tokens)",
    ),
    CheckpointSpec(
        model=OLMO_2_7B,
        stage_label="pretrain_10pct",
        revision="stage1-step93000-tokens391B",
        description="~10% of pretraining (391B tokens)",
    ),
    CheckpointSpec(
        model=OLMO_2_7B,
        stage_label="pretrain_50pct",
        revision="stage1-step465000-tokens1951B",
        description="~50% of pretraining (1951B tokens)",
    ),
    CheckpointSpec(
        model=OLMO_2_7B,
        stage_label="base",
        description="Full base model (4T tokens, no post-training)",
    ),
    CheckpointSpec(
        model=OLMO_2_7B,
        stage_label="sft",
        hf_id="allenai/OLMo-2-1124-7B-SFT",
        description="Supervised fine-tuning",
    ),
    CheckpointSpec(
        model=OLMO_2_7B,
        stage_label="dpo",
        hf_id="allenai/OLMo-2-1124-7B-DPO",
        description="Direct preference optimization",
    ),
    CheckpointSpec(
        model=OLMO_2_7B,
        stage_label="instruct",
        hf_id="allenai/OLMo-2-1124-7B-Instruct",
        description="Instruct (+ RLVR)",
    ),
]


PERSONA_SLUGS: list[str] = [
    # Original 10
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
    # R5: 20 additional contexts
    "diplomat",
    "journalist",
    "nurse",
    "salesperson",
    "lawyer",
    "judge",
    "soldier",
    "activist",
    "priest",
    "hacker",
    "detective",
    "nonprofit_ceo",
    "used_car_dealer",
    "hostage_negotiator",
    "cult_leader",
    "whistleblower",
    "lobbyist",
    "undercover_agent",
    "emergency_dispatcher",
    "parole_officer",
]
