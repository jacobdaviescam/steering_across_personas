# Persona-Conditional Steering Vectors

## Project Overview
Research repo investigating whether steering vectors for the same trait change depending on the active persona. Uses nnsight for activation access and Claude API for LLM-as-judge evaluation.

## Tech Stack
- Python 3.10+, PyTorch, Transformers, nnsight
- Anthropic Claude API for evaluation (anthropic SDK)
- Jupyter notebooks for experimental workflow

## Key Conventions
- All extraction uses nnsight's `model.trace()` context manager for activation capture
- Steering vectors are stored as `SteeringVector` dataclasses with full metadata
- Evaluation uses Claude as an LLM judge — scores are 0-1 floats per trait
- Notebooks are numbered and meant to be run sequentially (01 through 06)
- Model-specific configs live in `config.py` as frozen dataclass presets
- Outputs go to `outputs/` (gitignored)

## Running
```bash
pip install -e .
python -c "import persona_steering"  # verify install
```
Notebooks require GPU + model weights.

## File Layout
- `persona_steering/` — library code
- `notebooks/` — experimental notebooks (01-06)
- `data/prompts/` — contrastive prompt pairs per trait (JSON)
- `data/personas/` — persona induction configs (YAML)
- `outputs/` — gitignored results
