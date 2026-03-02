# Persona-Conditional Steering Vectors

Do steering vectors for the same trait change depending on which persona the model is operating under?

## Research Question

Steering vectors are extracted via contrastive activation differences. If we induce different personas along the **assistant axis** (Lu et al., 2026) — from deep roleplay through to fully aligned assistant — does the geometry of a trait's steering direction shift? If so, how does cross-persona transfer decay with axis distance, and what does this tell us about the relationship between persona and trait representations?

## Personas

Six personas positioned along (and outside) the assistant axis:

| Position | Persona | Description |
|----------|---------|-------------|
| off-axis | **Base Model** | Raw text completion, no chat template |
| -1.0 | **Deep Roleplay** | Fully embodied character, enigmatic, subversive |
| -0.5 | **Mild Roleplay** | Character with personality, still functional |
|  0.0 | **Neutral** | Minimal system prompt |
| +0.5 | **Mild Assistant** | Helpful AI, not performative |
| +1.0 | **Full Assistant** | Maximally helpful RLHF-aligned assistant |

## Traits

- **Honesty** — honest vs deceptive
- **Sycophancy** — sycophantic vs straightforward
- **Verbosity** — verbose vs concise
- **Formality** — formal vs casual

## Experimental Programme

1. **Persona Induction** — Induce 6 personas via system prompts and few-shot examples. Validate distinguishable activation signatures.

2. **Per-Persona Steering Extraction** — For each persona × trait combination, extract contrastive steering vectors from middle-layer residual stream activations using nnsight.

3. **Vector Comparison** — Compare vectors across personas. Key analysis: does cosine similarity decay with assistant axis distance?

4. **Cross-Persona Steering** — Apply vectors extracted under one persona while the model operates under a different persona. Measure behavioural effect via LLM-as-judge.

5. **Direction Analysis** — Compare steering directions to the assistant axis itself. Decompose into shared vs persona-specific components.

## Setup

```bash
pip install -e .
```

Requires GPU access and model weights for Gemma 2 or Llama 3 models.

Based on the assistant axis from [Lu et al. (2026)](https://arxiv.org/abs/2601.10387). Reference implementation: [safety-research/assistant-axis](https://github.com/safety-research/assistant-axis).

## Usage

Run notebooks sequentially:

```
notebooks/01_persona_induction.ipynb
notebooks/02_steering_extraction.ipynb
notebooks/03_vector_comparison.ipynb
notebooks/04_cross_persona_steering.ipynb
notebooks/05_direction_analysis.ipynb
notebooks/06_figures.ipynb
```

Or use the library directly:

```python
from persona_steering.config import GEMMA_2_9B, PersonaConfig, TraitConfig
from persona_steering.personas import PersonaInducer, load_axis_personas
from persona_steering.extraction import SteeringVectorExtractor
from persona_steering.steering import apply_steering
from persona_steering.analysis import compare_vectors, build_transfer_matrix
```

## Project Structure

```
persona_steering/   Core library
  config.py         Model configs, trait/persona definitions, axis ordering
  personas.py       Persona induction and axis loading
  extraction.py     Steering vector extraction via nnsight
  steering.py       Applying steering vectors
  evaluation.py     LLM-as-judge behavioural eval (Claude API)
  analysis.py       Comparison metrics and decomposition
  data.py           Contrastive prompt pair handling
  utils.py          Shared utilities
notebooks/          Step-by-step experimental notebooks (01-06)
data/prompts/       Contrastive prompt pairs per trait (JSON)
data/personas/      Persona induction configs (YAML)
outputs/            (gitignored) vectors, activations, figures
```
