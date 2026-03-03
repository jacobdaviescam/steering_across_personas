# Persona-Conditional Steering Vectors

## Project Overview
Research repo investigating whether steering vectors for the same trait change depending on the active persona. Uses concrete character personas to test whether trait interactions differ across identities (e.g. assertiveness in a farmer vs a politician). Uses nnsight for activation access and Claude API for LLM-as-judge evaluation.

## Personas (10 concrete archetypes)
- **Farmer** — Midwestern grain farmer (quiet competence, plain-spoken honesty)
- **Politician** — Populist political figure (dominance, strategic honesty)
- **Therapist** — Licensed clinical psychologist (core empathy, gentle boundaries)
- **Drill Sergeant** — Military drill instructor (assertiveness as identity, suppressed empathy)
- **Street Hustler** — Urban street entrepreneur (situational honesty, constant risk)
- **Professor** — Tenured philosophy professor (intellectual authority)
- **Tech CEO** — Silicon Valley startup founder (defining risk, outsized confidence)
- **Kindergarten Teacher** — Early childhood educator (nurturing empathy, defining warmth)
- **Surgeon** — Trauma surgeon (decisive assertiveness, calculated risk)
- **Con Artist** — Charming confidence trickster (inverted honesty, weaponised empathy)

## Traits (8)
Assertiveness, empathy, risk-taking, honesty, confidence, deference, warmth, impulsivity.

## Extraction Method
Instruction-variant approach: 5 pos/neg instruction pairs × 20 sampled questions (from 100) = 100 pairs per persona×trait. Same question under pos vs neg instruction isolates trait signal from content.

## Tech Stack
- Python 3.10+, PyTorch, Transformers, nnsight
- Anthropic Claude API for evaluation (anthropic SDK) and data generation
- Jupyter notebooks for experimental workflow
- Reference: assistant-axis-ref/ (cloned from safety-research/assistant-axis)

## Key Conventions
- All extraction uses nnsight's `model.trace()` for activation capture
- Steering vectors stored as `SteeringVector` dataclasses with full metadata
- Evaluation uses Claude as LLM judge — scores are 0-1 floats per trait
- Notebooks numbered 01-06, run sequentially
- Model configs in `config.py` as frozen dataclass presets
- Outputs go to `outputs/` (gitignored)
- `PERSONA_SLUGS` in config.py defines the canonical persona list
- Persona configs use `system_prompt_variants` (list of 5) for robust extraction
- Trait datasets use `instruction_variants` (5 pos/neg pairs) + `questions` (100 shared)
- `load_all_personas()` returns all personas sorted alphabetically by file
- `load_trait_dataset(trait)` / `load_all_trait_datasets()` for trait data

## Running
```bash
pip install -e .
python -c "import persona_steering"
```

### Generating data
```bash
python scripts/generate_data.py --traits --dry-run   # preview
python scripts/generate_data.py --traits              # generate all trait datasets
```

Notebooks require GPU + model weights.
