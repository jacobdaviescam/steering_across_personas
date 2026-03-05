"""Behavioural evaluation via LLM-as-judge (Claude API)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

from persona_steering.config import Trait, TRAIT_CONFIGS
from persona_steering.utils import log


@dataclass
class TraitScore:
    """Score for a single trait evaluation."""
    trait: Trait
    score: float  # 0.0 to 1.0
    explanation: str = ""
    raw_response: str = ""


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

