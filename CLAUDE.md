# Persona-Conditional Steering Vectors

## Project Overview
Research repo investigating whether steering vectors for the same trait change depending on the active persona. Personas are arranged along the assistant axis (Lu et al., 2026). Uses nnsight for activation access and Claude API for LLM-as-judge evaluation.

## Personas (6)
- **Base Model** (off-axis) — raw text completion control
- **Deep Roleplay** (-1.0) — far anti-assistant extreme
- **Mild Roleplay** (-0.5) — moderate roleplay
- **Neutral** (0.0) — midpoint
- **Mild Assistant** (+0.5) — moderate assistant
- **Full Assistant** (+1.0) — far assistant extreme

## Traits (4)
Honesty, sycophancy, verbosity, formality — each with 20 contrastive prompt pairs.

## Tech Stack
- Python 3.10+, PyTorch, Transformers, nnsight
- Anthropic Claude API for evaluation (anthropic SDK)
- Jupyter notebooks for experimental workflow
- Reference: assistant-axis-ref/ (cloned from safety-research/assistant-axis)

## Key Conventions
- All extraction uses nnsight's `model.trace()` for activation capture
- Steering vectors stored as `SteeringVector` dataclasses with full metadata
- Evaluation uses Claude as LLM judge — scores are 0-1 floats per trait
- Notebooks numbered 01-06, run sequentially
- Model configs in `config.py` as frozen dataclass presets
- Outputs go to `outputs/` (gitignored)
- `AXIS_PERSONA_ORDER` in config.py defines the canonical axis ordering
- `load_axis_personas()` returns only on-axis personas sorted by position

## Running
```bash
pip install -e .
python -c "import persona_steering"
```
Notebooks require GPU + model weights.
