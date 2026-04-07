# Persona-Conditional Steering Vectors

## Project Overview
Research repo investigating whether steering vectors for the same trait change depending on the active persona. Uses concrete character personas to test whether trait interactions differ across identities (e.g. assertiveness in a farmer vs a politician). Uses `assistant_axis` (from assistant-axis-ref/) for model loading, activation extraction, and steering. Uses Claude API for LLM-as-judge evaluation and data generation.

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
Instruction-variant approach: 5 pos/neg instruction pairs × 20 sampled questions (from 100) = 100 pairs per persona×trait. Same question under pos vs neg instruction isolates trait signal from content. Contrastive vector = mean(pos activations) - mean(neg activations).

## Tech Stack
- Python 3.10+, PyTorch, Transformers, vLLM
- `assistant_axis` (from `assistant-axis-ref/`) — ProbingModel, ActivationExtractor, ConversationEncoder, SpanMapper, VLLMGenerator, ActivationSteering
- Anthropic Claude API for evaluation (anthropic SDK) and data generation
- Reference: assistant-axis-ref/ (cloned from safety-research/assistant-axis)

## Pipeline (numbered scripts in `pipeline/`)

| Step | Script | What it does |
|------|--------|-------------|
| 0 | `pipeline/0_generate_data.py` | Generate trait datasets (instruction variants + questions) via Claude API |
| 1 | `pipeline/1_generate.py` | Generate responses via vLLM for all persona×trait×direction combos |
| 2 | `pipeline/2_activations.py` | Extract mean assistant-turn activations using ProbingModel + forward hooks |
| 3 | `pipeline/3_vectors.py` | Compute contrastive vectors: mean(pos) - mean(neg) |
| 4 | `pipeline/4_analysis.py` | Transfer matrices, clustering, decomposition, assistant axis alignment |

### Data flow
```
trait datasets (JSON)  →  1_generate  →  responses (JSONL per persona×trait×direction)
                                              ↓
persona configs (YAML)          2_activations  →  activations (.pt per file)
                                              ↓
                                    3_vectors  →  vectors (.pt per persona×trait)
                                              ↓
                                   4_analysis  →  transfer matrices, clusters, decomposition
```

### Output structure
```
outputs/{model}/
  responses/{persona}_{trait}_{pos|neg}.jsonl
  activations/{persona}_{trait}_{pos|neg}.pt
  vectors/{persona}_{trait}.pt
  analysis/transfer_matrix.npy, clusters.json, decomposition.json
```

## Key Conventions
- Pipeline scripts import `assistant_axis` via `sys.path.insert(0, "assistant-axis-ref")`
- Activation extraction uses PyTorch forward hooks (via ProbingModel/ActivationExtractor), NOT nnsight
- Steering vectors stored as .pt files: `{"vector": tensor(n_layers, hidden_dim), "persona": str, "trait": str, ...}`
- Evaluation uses Claude as LLM judge — scores are 0-1 floats per trait
- Model configs in `config.py` as frozen dataclass presets
- Outputs go to `outputs/` (gitignored)
- `PERSONA_SLUGS` in config.py defines the canonical persona list
- Persona configs use `system_prompt_variants` (list of 5) for robust extraction
- Trait datasets use `instruction_variants` (5 pos/neg pairs) + `questions` (100 shared)
- `load_all_personas()` returns all personas sorted alphabetically by file
- `load_trait_dataset(trait)` / `load_all_trait_datasets()` for trait data

## Package modules (`persona_steering/`)
- `config.py` — Trait enum, PersonaConfig, ModelConfig, paths, presets
- `personas.py` — YAML persona loading (`load_persona`, `load_all_personas`)
- `data.py` — Trait dataset loading/saving/generation
- `analysis.py` — Transfer matrices, clustering, shared/specific decomposition
- `evaluation.py` — Claude LLM-as-judge scoring
- `reference.py` — Reference vector loading
- `utils.py` — Logging, device, caching, cosine similarity

## Running
```bash
pip install -e .
python -c "import persona_steering"
```

### Generating data
```bash
python pipeline/0_generate_data.py --traits --dry-run   # preview
python pipeline/0_generate_data.py --traits              # generate all trait datasets
```

### Full pipeline (requires GPU + model weights)
```bash
# Run everything (both IV and CAA):
./run.sh google/gemma-2-27b-it
# Or individual steps:
python pipeline/1_generate.py --model google/gemma-2-27b-it
python pipeline/2_activations.py --model google/gemma-2-27b-it
python pipeline/3_vectors.py --activations-dir outputs/gemma-2-27b-it/activations
python pipeline/4_analysis.py --vectors-dir outputs/gemma-2-27b-it/vectors --layer 22
```
