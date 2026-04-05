# Persona-Conditional Steering Vectors

Do steering vectors for the same trait change depending on which persona the model is operating under?

## Research Question

Steering vectors are typically extracted as if traits are universal. But personality traits interact with identity — assertiveness means something different for a farmer than for a politician. We extract per-persona steering vectors for 8 traits across 10 concrete character personas and test whether the resulting vector geometry is persona-specific or trait-universal.

We compare two extraction methods:
- **Instruction-variant**: same question under positive vs negative trait instructions (controls for content)
- **CAA (Contrastive Activation Addition)**: A/B multiple-choice format with forced answer token

## Personas (10 concrete archetypes)

| Persona | Description |
|---------|-------------|
| **Farmer** | Midwestern grain farmer — quiet competence, plain-spoken honesty |
| **Politician** | Populist political figure — dominance, strategic honesty |
| **Therapist** | Licensed clinical psychologist — core empathy, gentle boundaries |
| **Drill Sergeant** | Military drill instructor — assertiveness as identity, suppressed empathy |
| **Street Hustler** | Urban street entrepreneur — situational honesty, constant risk |
| **Professor** | Tenured philosophy professor — intellectual authority |
| **Tech CEO** | Silicon Valley startup founder — defining risk, outsized confidence |
| **Kindergarten Teacher** | Early childhood educator — nurturing empathy, defining warmth |
| **Surgeon** | Trauma surgeon — decisive assertiveness, calculated risk |
| **Con Artist** | Charming confidence trickster — inverted honesty, weaponised empathy |

## Traits (8)

Assertiveness, empathy, risk-taking, honesty, confidence, deference, warmth, impulsivity.

## Key Findings

**Steering vectors are predominantly trait-universal, not persona-specific** — but the degree of universality depends on the extraction method.

### Instruction-Variant Method
- Mean cross-persona transfer: **0.82** cosine similarity
- Shared variance: 73–93% (honesty highest, risk-taking lowest)
- All personas cluster together — no distinct sub-populations
- Honesty strongly aligned with assistant axis (+0.57)

### CAA Method
- Mean cross-persona transfer: **0.64** cosine similarity (lower than IV)
- Shared variance: 50–78% (assertiveness highest, deference lowest)
- More persona-specific structure — persona identity "leaks" into CAA vectors
- Weaker axis alignment across all traits; sign flips for 6/8 traits vs IV

### Notable Patterns
- **Drill sergeant ↔ therapist** is consistently the most dissimilar pair under both methods
- **Impulsivity** and **risk-taking** are the most persona-conditioned traits
- **Honesty** is the most universal trait and most aligned with the assistant axis

## Pipeline

Numbered scripts in `pipeline/`:

| Step | Script | What it does |
|------|--------|-------------|
| 0 | `0_generate_data.py` | Generate trait datasets (instruction variants + questions) via Claude API |
| 0c | `0c_generate_caa_data.py` | Generate CAA-style A/B multiple-choice datasets via Claude API |
| 1 | `1_generate.py` | Generate responses via vLLM for all persona×trait×direction combos |
| 2 | `2_activations.py` | Extract mean assistant-turn activations using ProbingModel + forward hooks |
| 2c | `2c_caa_activations.py` | Extract answer-token activations for CAA A/B prompts |
| 3 | `3_vectors.py` | Compute contrastive vectors: mean(pos) - mean(neg) |
| 4 | `4_analysis.py` | Transfer matrices, clustering, decomposition, assistant axis alignment |
| 5 | `5_visualize.py` | Generate publication-ready figures |
| 6 | `6_behavioral_eval.py` | Claude LLM-as-judge behavioural scoring |
| 7 | `7_eval_analysis.py` | Analyse and visualise evaluation results |
| 8 | `8_steered_generation.py` | Apply source persona's steering vector to target persona during generation |
| 9 | `9_steering_eval.py` | Evaluate steered responses via Claude LLM-as-judge |

### Data flow

```
trait datasets (JSON)  →  1_generate  →  responses (JSONL per persona×trait×direction)
                                              ↓
persona configs (YAML)          2_activations  →  activations (.pt per file)
                                              ↓
                                    3_vectors  →  vectors (.pt per persona×trait)
                                              ↓
                                   4_analysis  →  transfer matrices, clusters, decomposition
                                              ↓
                                 5_visualize   →  figures (PNG)
                                              ↓
                         8_steered_generation  →  steered responses (JSONL)
                                              ↓
                              9_steering_eval  →  transfer scores (JSON)
```

## Output Structure

```
outputs/{model}/
  responses/                      Step 1 responses
  activations/                    Step 2 activation tensors
  caa_activations/                Step 2c CAA activation tensors
  vectors/                        Step 3 instruction-variant steering vectors
  caa_vectors/                    Step 3 CAA steering vectors
  analysis_instruction_variant/   Step 4 analysis (IV method)
  caa_analysis/                   Step 4 analysis (CAA method)
  figures/                        Step 5 figures (IV method)
  caa_figures/                    Step 5 figures (CAA method)
  steered_responses_alpha2/       Step 8 steered responses (α=2)
  steered_responses_alpha4/       Step 8 steered responses (α=4)
  eval/                           Step 9 evaluation scores
  axis.pt                         Assistant axis reference vector
```

## Project Structure

```
persona_steering/       Core library
  config.py             Trait enum, PersonaConfig, ModelConfig, paths, presets
  personas.py           YAML persona loading (load_persona, load_all_personas)
  data.py               Trait dataset loading/saving/generation (IV + CAA)
  analysis.py           Transfer matrices, clustering, shared/specific decomposition
  evaluation.py         Claude LLM-as-judge scoring
  reference.py          Reference vector loading
  utils.py              Logging, device, caching, cosine similarity
pipeline/               Numbered pipeline scripts (0–9)
data/personas/          Persona configs (10 YAML files)
data/prompts/           Trait datasets (instruction-variant JSON)
data/prompts/caa/       CAA A/B datasets (JSON)
assistant-axis-ref/     Reference checkout of safety-research/assistant-axis
outputs/                Generated outputs (partially tracked in git)
```

## Setup

```bash
pip install -e .
git clone https://github.com/safety-research/assistant-axis.git assistant-axis-ref
```

Requires GPU access and model weights for generation and activation extraction. Uses `google/gemma-2-27b-it` as the primary model.

Based on the assistant axis from [Lu et al. (2026)](https://arxiv.org/abs/2601.10387).

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
# .env
ANTHROPIC_API_KEY=sk-ant-...      # Required for steps 0, 6, 9 (Claude judge / data gen)
HF_TOKEN=hf_...                   # Required for gated models (Gemma 2, etc.)
WANDB_API_KEY=wandb_v1_...        # Optional, enables experiment tracking
WANDB_PROJECT=persona-steering    # W&B project name (default: persona-steering)
WANDB_EXPERIMENT=default          # Tag for filtering runs (e.g. gemma4-baseline)
```

The `.env` file is loaded automatically by all pipeline scripts via `python-dotenv`.

### Experiment tracking with W&B

W&B integration is optional. Install with:

```bash
pip install -e ".[tracking]"
```

When `WANDB_API_KEY` is set in `.env`, each pipeline step logs a W&B run with:
- **Metrics**: cosine similarities, effect sizes, correlations
- **Artifacts**: vectors, analysis results, evaluation scores, figures
- **Images**: all generated figures viewable in the W&B dashboard

Runs are tagged with `model:<name>`, `experiment:<name>`, and `step:<name>` for filtering. All runs for the same model are grouped together.

To disable W&B (even if the key is set): add `WANDB_DISABLED=true` to `.env`.

Large artifacts are opt-in to avoid excessive upload costs:
- Activations (~18GB): set `WANDB_UPLOAD_ACTIVATIONS=true`
- Responses (~200MB): set `WANDB_UPLOAD_RESPONSES=true`
