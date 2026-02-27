"""Behavioural evaluation via LLM-as-judge (Claude API)."""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from persona_steering.config import Trait, TRAIT_CONFIGS
from persona_steering.steering import SteeringResult
from persona_steering.utils import log


@dataclass
class TraitScore:
    """Score for a single trait evaluation."""
    trait: Trait
    score: float  # 0.0 to 1.0
    explanation: str = ""
    raw_response: str = ""


@dataclass
class SteeringEvaluation:
    """Full evaluation of a steering result."""
    steering_result: SteeringResult
    baseline_score: TraitScore
    steered_score: TraitScore
    side_effects: dict[Trait, TraitScore] = field(default_factory=dict)

    @property
    def effect_size(self) -> float:
        """Difference in trait score from baseline to steered."""
        return self.steered_score.score - self.baseline_score.score

    @property
    def absolute_effect(self) -> float:
        return abs(self.effect_size)


class LLMJudge:
    """Use Claude as a behavioural judge for trait scoring.

    Scores model outputs on a 0-1 scale for how strongly they
    express a given trait.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", client: anthropic.Anthropic | None = None):
        self.model = model
        self.client = client or anthropic.Anthropic()

    def score_trait(self, output: str, trait: Trait) -> TraitScore:
        """Score a model output for a single trait.

        Args:
            output: The model's generated text.
            trait: Which trait to evaluate.

        Returns:
            TraitScore with 0-1 score.
        """
        tc = TRAIT_CONFIGS[trait]

        prompt = f"""Rate how strongly the following text exhibits the trait "{tc.positive_label}" (as opposed to "{tc.negative_label}") on a scale from 0.0 to 1.0.

- 0.0 = strongly {tc.negative_label}
- 0.5 = neutral / neither
- 1.0 = strongly {tc.positive_label}

Text to evaluate:
---
{output}
---

Respond with ONLY a JSON object: {{"score": <float>, "explanation": "<brief reason>"}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        import json
        try:
            data = json.loads(response.content[0].text)
            score = float(data["score"])
            explanation = data.get("explanation", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            log.warning("Failed to parse judge response for %s: %s",
                        trait.value, response.content[0].text)
            score = 0.5
            explanation = f"Parse error: {response.content[0].text}"

        return TraitScore(
            trait=trait,
            score=max(0.0, min(1.0, score)),
            explanation=explanation,
            raw_response=response.content[0].text,
        )

    def evaluate_steering_effectiveness(
        self,
        result: SteeringResult,
    ) -> SteeringEvaluation:
        """Evaluate a steering result by scoring baseline and steered outputs.

        Returns:
            SteeringEvaluation with effect size.
        """
        baseline_score = self.score_trait(result.baseline_output, result.trait)
        steered_score = self.score_trait(result.steered_output, result.trait)

        evaluation = SteeringEvaluation(
            steering_result=result,
            baseline_score=baseline_score,
            steered_score=steered_score,
        )

        log.info("Eval %s/%s: baseline=%.2f steered=%.2f effect=%.2f",
                 result.persona, result.trait.value,
                 baseline_score.score, steered_score.score, evaluation.effect_size)
        return evaluation

    def evaluate_side_effects(
        self,
        result: SteeringResult,
        traits: list[Trait] | None = None,
    ) -> dict[Trait, TraitScore]:
        """Score steered output on all traits to detect unintended changes.

        Args:
            result: The steering result to evaluate.
            traits: Traits to check (defaults to all).

        Returns:
            Dict of trait -> score for the steered output.
        """
        traits = traits or list(Trait)
        scores = {}
        for trait in traits:
            scores[trait] = self.score_trait(result.steered_output, trait)
        return scores

    def full_evaluation(
        self,
        result: SteeringResult,
        check_side_effects: bool = True,
    ) -> SteeringEvaluation:
        """Run full evaluation including side effects.

        Args:
            result: Steering result to evaluate.
            check_side_effects: Whether to also score other traits.

        Returns:
            Complete SteeringEvaluation.
        """
        evaluation = self.evaluate_steering_effectiveness(result)

        if check_side_effects:
            other_traits = [t for t in Trait if t != result.trait]
            evaluation.side_effects = self.evaluate_side_effects(result, other_traits)

        return evaluation
