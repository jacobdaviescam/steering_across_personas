# Persona-Conditional Steering Vectors

Do steering vectors for the same trait change depending on which persona the model is operating under?

## Research Question

Steering vectors are extracted via contrastive activation differences. But if we induce different *personas* in a model (base-model vs assistant vs intermediate positions), does the geometry of a trait's steering direction shift? If so, how does cross-persona transfer decay, and what does this tell us about the relationship between persona and trait representations?

## Experimental Programme

1. **Persona Induction** — Induce distinct personas (base model, assistant, intermediate) via system prompts, activation injection, or few-shot examples. Validate that personas produce distinguishable activation signatures.

2. **Per-Persona Steering Extraction** — For each persona × trait combination, extract contrastive steering vectors from middle-layer residual stream activations using nnsight.

3. **Cross-Persona Steering** — Apply vectors extracted under one persona while the model operates under a different persona. Measure behavioural effect and compare to same-persona steering.

4. **Direction Analysis** — Compare steering vector directions across personas. Decompose into shared vs persona-specific components. Relate steering directions to the inter-persona axis in activation space.

## Setup

```bash
pip install -e .
```

Requires GPU access and model weights for Gemma 2 or Llama 3 models.

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
from persona_steering.personas import PersonaInducer
from persona_steering.extraction import SteeringVectorExtractor
from persona_steering.steering import apply_steering
from persona_steering.analysis import compare_vectors, build_transfer_matrix
```

## Traits

- Honesty
- Sycophancy
- Verbosity
- Compliance
- Confidence
- Formality

## Project Structure

```
persona_steering/   Core library
  config.py         Model configs, trait/persona definitions
  personas.py       Persona induction methods
  extraction.py     Steering vector extraction via nnsight
  steering.py       Applying steering vectors
  evaluation.py     LLM-as-judge behavioural eval
  analysis.py       Comparison metrics and decomposition
  data.py           Contrastive prompt pair handling
  utils.py          Shared utilities
notebooks/          Step-by-step experimental notebooks
data/               Prompt pairs and persona configs
outputs/            (gitignored) vectors, activations, figures
```
