"""OpenRouter-backed Claude judge.

Drop-in alternative to evaluation.LLMJudge that routes through OpenRouter
instead of the Anthropic API. Lets us use Claude as a judge without an
Anthropic API key.

Configuration
-------------
- ``OPENROUTER_API_KEY`` environment variable is required.
- Default model is ``anthropic/claude-sonnet-4.5``. Pass a different model id
  (e.g. ``anthropic/claude-opus-4.7``) to the constructor to swap it out.
- We hit OpenRouter's OpenAI-compatible chat completions endpoint via httpx
  to avoid an extra SDK dependency.

Two scorers are exposed:

* ``score_trait(text, trait)`` -> 0..1, matching ``evaluation.LLMJudge``.
* ``score_persona_match(text, persona_slug, persona_description)`` -> 0..1,
  measures how much a free-text response reads as that persona's natural
  output. Used by the naturalistic-response and adversarial-cells pipelines.

Both methods use a small JSON output schema and parse defensively.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from persona_steering.config import Trait, TRAIT_CONFIGS
from persona_steering.utils import log


DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 60.0


@dataclass
class JudgeScore:
    score: float
    explanation: str = ""
    raw: str = ""


class OpenRouterJudge:
    """LLM-as-judge backed by OpenRouter (default: Claude Sonnet 4.5)."""

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key: Optional[str] = None,
                 max_retries: int = 4,
                 retry_base_delay: float = 1.5,
                 timeout: float = DEFAULT_TIMEOUT,
                 referer: str = "https://github.com/anonymous/persona-steering",
                 title: str = "persona-steering"):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Get a key at openrouter.ai and "
                "export OPENROUTER_API_KEY=sk-or-... before running this script."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        # OpenRouter requires HTTP-Referer + X-Title headers for attribution.
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": referer,
                "X-Title": title,
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------
    # internal: a single chat call with retry
    # ------------------------------------------------------------------
    def _chat(self, prompt: str, max_tokens: int = 256) -> str:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post("/chat/completions", json=body)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                last_err = e
                delay = self.retry_base_delay * (2 ** attempt)
                log.warning("OpenRouter call failed (attempt %d/%d): %s — sleep %.1fs",
                            attempt + 1, self.max_retries, e, delay)
                time.sleep(delay)
        raise RuntimeError(
            f"OpenRouter call failed after {self.max_retries} attempts: {last_err}"
        )

    @staticmethod
    def _parse_json(text: str, default_score: float = 0.5) -> JudgeScore:
        """Extract a JSON object with 'score' from the model's text. Robust to
        leading commentary and to ``json`` code fences."""
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            candidate = text[start:end + 1]
            try:
                data = json.loads(candidate)
                score = float(data.get("score", default_score))
                return JudgeScore(
                    score=max(0.0, min(1.0, score)),
                    explanation=str(data.get("explanation", "")),
                    raw=text,
                )
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                log.warning("Judge JSON parse failed (%s): %r", e, text[:200])
        return JudgeScore(score=default_score, explanation="parse_error", raw=text)

    # ------------------------------------------------------------------
    # public: trait scorer (0..1 along the trait axis)
    # ------------------------------------------------------------------
    def score_trait(self, output: str, trait: Trait) -> JudgeScore:
        tc = TRAIT_CONFIGS[trait]
        prompt = (
            f"Rate how strongly the following text exhibits the trait "
            f"\"{tc.positive_label}\" (as opposed to \"{tc.negative_label}\") "
            f"on a 0.0--1.0 scale.\n"
            f"- 0.0 = strongly {tc.negative_label}\n"
            f"- 0.5 = neutral / neither\n"
            f"- 1.0 = strongly {tc.positive_label}\n\n"
            f"Text to evaluate:\n---\n{output}\n---\n\n"
            f"Respond with ONLY a JSON object:\n"
            f'  {{"score": <float between 0 and 1>, "explanation": "<one sentence>"}}'
        )
        return self._parse_json(self._chat(prompt))

    # ------------------------------------------------------------------
    # public: persona match scorer (used by adversarial / naturalistic)
    # ------------------------------------------------------------------
    def score_persona_match(self, output: str, persona_slug: str,
                            persona_description: str) -> JudgeScore:
        """How much does this free-text response read as the persona's natural
        output, regardless of trait? 0 = unrelated, 1 = unmistakably this
        persona."""
        prompt = (
            f"Below is a description of a fictional speaker, then a piece of "
            f"text. Rate how much the text reads as something this speaker "
            f"would naturally produce, on a 0.0--1.0 scale.\n\n"
            f"- 0.0 = clearly not this speaker / generic assistant tone\n"
            f"- 0.5 = could be anyone, no strong fit either way\n"
            f"- 1.0 = unmistakably this speaker's voice and reasoning\n\n"
            f"Speaker ({persona_slug}):\n{persona_description}\n\n"
            f"Text:\n---\n{output}\n---\n\n"
            f"Respond with ONLY a JSON object:\n"
            f'  {{"score": <float>, "explanation": "<one sentence>"}}'
        )
        return self._parse_json(self._chat(prompt))
