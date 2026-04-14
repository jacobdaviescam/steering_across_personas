"""Shared helpers for the council experiments (c1, c2, c9, c11).

These experiments build on the existing ``outputs/{model}/`` tree
(activations, vectors, responses) and write their own artifacts under
``outputs/{model}/council/{c1,c2,c9,c11}/``.

The helpers here are intentionally thin wrappers around files on disk so the
council scripts don't accumulate their own copy of the IO logic.

All tensors are returned on CPU. Use ``.to(device)`` downstream if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from persona_steering.config import (
    ACTIVATIONS_SUBDIR,
    OUTPUTS_DIR,
    PERSONA_SLUGS,
    TARGET_LAYER,
    Trait,
    VECTORS_SUBDIR,
)
from persona_steering.utils import log, model_short_name


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

COUNCIL_SUBDIR = "council"
ASSISTANT_DEFAULT_SLUG = "assistant_default"


def model_dir(model: str) -> Path:
    """Return ``outputs/{short-model-name}/``."""
    return OUTPUTS_DIR / model_short_name(model)


def council_dir(model: str, experiment: str) -> Path:
    """Return ``outputs/{short-model}/council/{experiment}/`` (created)."""
    d = model_dir(model) / COUNCIL_SUBDIR / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Activations / vectors
# ---------------------------------------------------------------------------

def _activations_path(model: str, persona: str, trait: str, direction: str) -> Path:
    return model_dir(model) / ACTIVATIONS_SUBDIR / f"{persona}_{trait}_{direction}.pt"


def _vector_path(model: str, persona: str, trait: str) -> Path:
    return model_dir(model) / VECTORS_SUBDIR / f"{persona}_{trait}.pt"


def load_persona_activations(
    model: str,
    persona: str,
    trait: str,
    direction: str,
    layer: int = TARGET_LAYER,
) -> torch.Tensor:
    """Load all layer-``layer`` activations for one (persona, trait, direction).

    The activation file is a dict of sample_key -> tensor of shape
    ``(n_layers, hidden_dim)``. Returns a stacked ``(n_samples, hidden_dim)``
    float32 tensor with NaN/Inf rows removed.
    """
    path = _activations_path(model, persona, trait, direction)
    if not path.exists():
        raise FileNotFoundError(f"Missing activation file: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)

    rows = []
    for key, t in blob.items():
        if not isinstance(t, torch.Tensor):
            continue
        if t.ndim != 2 or layer >= t.shape[0]:
            continue
        row = t[layer].float()
        if torch.isnan(row).any() or torch.isinf(row).any():
            continue
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No usable activations at layer {layer} in {path}")
    return torch.stack(rows, dim=0)


def load_trait_vector(
    model: str,
    persona: str,
    trait: str,
    layer: int = TARGET_LAYER,
) -> torch.Tensor:
    """Load a persona×trait contrastive vector at a single layer (hidden_dim,)."""
    path = _vector_path(model, persona, trait)
    if not path.exists():
        raise FileNotFoundError(f"Missing vector file: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    v = blob["vector"] if isinstance(blob, dict) else blob
    if v.ndim == 2:
        v = v[layer]
    return v.float()


def load_assistant_vector(
    model: str,
    trait: str,
    layer: int = TARGET_LAYER,
    slug: str = ASSISTANT_DEFAULT_SLUG,
) -> torch.Tensor:
    """Load the Assistant-baseline trait vector (no persona prompt).

    Raises :class:`FileNotFoundError` with an explanatory message if the
    baseline hasn't been produced yet — the caller is expected to handle this
    by running the pipeline with an empty system prompt under ``slug``.
    """
    path = _vector_path(model, slug, trait)
    if not path.exists():
        raise FileNotFoundError(
            f"Assistant-baseline vector for trait={trait!r} not found at {path}. "
            f"Run pipeline with an empty system prompt under persona slug={slug!r}."
        )
    return load_trait_vector(model, slug, trait, layer=layer)


# ---------------------------------------------------------------------------
# Persona baselines (mean activation across all prompts for a persona)
# ---------------------------------------------------------------------------

def persona_baseline(
    model: str,
    persona: str,
    traits: list[str] | None = None,
    layer: int = TARGET_LAYER,
) -> torch.Tensor:
    """Pooled-mean activation across all (trait, direction) files for a persona.

    Serves as the "persona identity" direction used by E9 residualization and
    E11 extrapolation.
    """
    traits = traits or [t.value for t in Trait]
    pools = []
    for trait in traits:
        for direction in ("pos", "neg"):
            try:
                pools.append(load_persona_activations(model, persona, trait, direction, layer))
            except FileNotFoundError as err:
                log.warning("persona_baseline skipping: %s", err)
    if not pools:
        raise RuntimeError(f"No activations found for persona={persona}")
    cat = torch.cat(pools, dim=0)
    return cat.mean(dim=0)


# ---------------------------------------------------------------------------
# Cosine helpers (vectorized numpy ops — cheap on the 4608-d vectors we use)
# ---------------------------------------------------------------------------

def cosine(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    a_np = a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    b_np = b.numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    a_np = a_np.reshape(-1).astype(np.float64)
    b_np = b_np.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a_np)
    nb = np.linalg.norm(b_np)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a_np.dot(b_np) / (na * nb))


def mean_pairwise_cosine(vectors: list[torch.Tensor] | np.ndarray) -> float:
    """Mean pairwise cosine similarity across a list of vectors."""
    if isinstance(vectors, list):
        mat = np.stack([v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v) for v in vectors])
    else:
        mat = np.asarray(vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    sims = unit @ unit.T
    n = sims.shape[0]
    if n < 2:
        return 1.0
    mask = ~np.eye(n, dtype=bool)
    return float(sims[mask].mean())


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@dataclass
class ScoredGeneration:
    text: str
    trait_score: float
    persona_coherence: float | None = None
    extra: dict | None = None


def score_generations(
    texts: list[str],
    trait: Trait,
    persona_description: str | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> list[ScoredGeneration]:
    """Score a batch of generations with the Claude LLM-as-judge.

    Always computes the 0–1 trait-expression score via
    :class:`persona_steering.evaluation.LLMJudge`. If ``persona_description``
    is supplied, also asks the judge to rate persona coherence 0–1 (used by
    E2 and E11).
    """
    from persona_steering.evaluation import LLMJudge

    judge = LLMJudge(model=model)
    out: list[ScoredGeneration] = []
    for text in texts:
        trait_score = judge.score_trait(text, trait).score
        persona_score: float | None = None
        if persona_description:
            persona_score = _score_persona_coherence(judge, text, persona_description)
        out.append(ScoredGeneration(text=text, trait_score=trait_score, persona_coherence=persona_score))
    return out


def _score_persona_coherence(judge, text: str, persona_description: str) -> float:
    """Reuse the judge's client to ask a persona-consistency question."""
    import json

    prompt = (
        "Rate how consistent the following text is with the persona described, "
        "on a scale from 0.0 to 1.0.\n\n"
        f"Persona description:\n{persona_description}\n\n"
        "Text:\n---\n"
        f"{text}\n---\n\n"
        'Respond with ONLY a JSON object: {"score": <float>, "explanation": "<brief reason>"}'
    )
    response = judge.client.messages.create(
        model=judge.model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        data = json.loads(response.content[0].text)
        return max(0.0, min(1.0, float(data["score"])))
    except (json.JSONDecodeError, KeyError, ValueError):
        log.warning("Persona-coherence judge parse failed: %s", response.content[0].text)
        return 0.5


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def all_traits() -> list[str]:
    return [t.value for t in Trait]


def all_personas() -> list[str]:
    return list(PERSONA_SLUGS)
