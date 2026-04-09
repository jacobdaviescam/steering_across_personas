# Persona-Conditional Steering Vectors

Do steering vectors for the same trait change depending on which persona the model is operating under?

## Research Question

Steering vectors are typically extracted as if traits are universal. But personality traits interact with identity — assertiveness means something different for a farmer than for a politician. We extract per-persona steering vectors for 8 traits across 10 concrete character personas and test whether the resulting vector geometry is persona-specific or trait-universal.

We compare two extraction methods:

### Instruction-Variant (IV)

The same question is answered twice under opposing trait instructions. Activations are extracted from the full assistant response. Because the question is identical in both conditions, content is controlled and the contrastive vector isolates the trait signal.

**Example** (assertiveness):
- **Positive instruction**: *"Express your views with confidence and conviction, stating your position clearly without unnecessary hedging."*
- **Negative instruction**: *"Approach all topics with humility and deference, acknowledging that others likely know better."*
- **Question**: *"What's the best way to negotiate a salary increase with my boss?"*

### Contrastive Activation Addition (CAA)

A/B multiple-choice scenarios where the model is forced to pick option A or B. Activations are extracted at the answer token position only. Polarity is randomized (50/50 which option is the positive trait).

**Example** (assertiveness):
- **Scenario**: *"Your doctor recommends a treatment you're uncomfortable with after doing your own research."*
- **Option A** (negative): *"I guess you know best. I'll just go with whatever you think is right."*
- **Option B** (positive): *"I've done some research and have concerns. Can we discuss other options?"*

### Pipeline branching

Both methods share the same persona configs and trait definitions. The pipeline branches at data generation and activation extraction, then reconverges:

```
                    ┌── 0_generate_data ── 1_generate ── 2_activations ──┐
                    │        (IV)             (IV)           (IV)         │
Data + Personas ────┤                                                    ├── 3_vectors ── 4_analysis ── 5/5b_visualize
                    │                                                    │        │
                    └── 0c_generate_caa_data ──────── 2c_caa_activations─┘        ├── 6_eval ── 7_eval_analysis
                              (CAA)                        (CAA)                  ├── 8_steered_gen ── 9_steering_eval
                                                                                  └── 10_oracle ── 11_oracle_analysis
```

Steps 3 onward are shared scripts — the method is determined by which input directory you point them at. To run both methods for the same model, use separate output directories (e.g. `--output-dir outputs/model/caa_vectors`). W&B runs are tagged `method:iv` or `method:caa` automatically via `infer_method()` which checks for "caa" in the directory path.

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
| 0 | `0_generate_data.py` | Generate IV trait datasets (instruction variants + questions) via Claude API |
| 0c | `0c_generate_caa_data.py` | Generate CAA-style A/B multiple-choice datasets via Claude API |
| 1 | `1_generate.py` | Generate responses via vLLM for all persona×trait×direction combos (IV only) |
| 2 | `2_activations.py` | Extract mean assistant-turn activations (IV only) |
| 2c | `2c_caa_activations.py` | Extract answer-token activations for CAA A/B prompts |
| 3 | `3_vectors.py` | Compute contrastive vectors: mean(pos) - mean(neg) |
| 4 | `4_analysis.py` | Transfer matrices, clustering, decomposition, assistant axis alignment |
| 5 | `5_visualize.py` | Generate publication-ready figures |
| 5b | `5b_persona_landscape.py` | Persona/trait landscape PCA and clustering |
| 5* | `5_activation_landscape.py` | Activation-space geometry and triangle-inequality bounds |
| 6 | `6_behavioral_eval.py` | Claude LLM-as-judge behavioural scoring |
| 7 | `7_eval_analysis.py` | Analyse and visualise evaluation results |
| 8 | `8_steered_generation.py` | Apply source persona's steering vector to target persona during generation |
| 9 | `9_steering_eval.py` | Evaluate steered responses via Claude LLM-as-judge |
| 10 | `10_oracle.py` | Interpret vectors/activations via Activation Oracle (LoRA decoder) |
| 11 | `11_oracle_analysis.py` | Analyse oracle results: trait/persona classification accuracy |
| t1 | `t1_trajectory_activations.py` | Extract CAA activations across OLMo training-stage checkpoints |
| t2 | `t2_trajectory_vectors.py` | Compute vectors for each training stage |
| t3 | `t3_trajectory_analysis.py` | Cross-stage transfer matrices, alignment, subspace overlap, cluster stability |
| t4 | `t4_trajectory_figures.py` | Publication figures for the training trajectory experiment |

### Data flow

```
IV branch:
  trait datasets (JSON)  →  1_generate  →  responses (JSONL)  →  2_activations  →  activations (.pt)
                                                                                        ↓
CAA branch:                                                                       3_vectors  →  vectors (.pt)
  CAA datasets (JSON)  →  2c_caa_activations  →  caa_activations (.pt) ────────────────↗        ↓
                                                                                          4_analysis  →  transfer matrices, clusters
                                                                                                ↓
                                                                                          5_visualize  →  figures (PNG)
                                                                                                ↓
                                                                                   8_steered_gen  →  steered responses (JSONL)
                                                                                                ↓
                                                                                    9_steering_eval  →  transfer scores (JSON)

Trajectory branch (OLMo checkpoints):
  CAA datasets  →  t1 (per-stage activations)  →  t2 (per-stage vectors)  →  t3 (cross-stage analysis)  →  t4 (figures)
```

## Output Structure

Default output directories (override any with `--output-dir`):

```
outputs/{model}/
  responses/              Step 1 IV responses
  activations/            Step 2 IV activation tensors
  caa_activations/        Step 2c CAA activation tensors
  vectors/                Step 3 contrastive steering vectors
  analysis/               Step 4 transfer matrices, clusters, decomposition
  figures/                Step 5 publication figures
  analysis_landscape/     Step 5* activation-space geometry
  eval/                   Step 6 behavioral evaluation scores
  steered_responses/      Step 8 steered responses
  oracle/                 Step 10 oracle interpretations
  oracle_analysis/        Step 11 oracle classification metrics
  axis.pt                 Assistant axis reference vector

outputs/OLMo-2-1124-7B/
  {stage_label}/caa_activations/  t1 per-stage activations
  {stage_label}/vectors/          t2 per-stage vectors
  trajectory/                     t3 cross-stage analysis
  figures/trajectory/             t4 trajectory figures
```

To run both IV and CAA for the same model, use `--output-dir` to separate them (e.g. `--output-dir outputs/model/caa_vectors`).

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
outputs/                Generated outputs (gitignored)
```

## Setup

```bash
pip install -e .
git clone https://github.com/safety-research/assistant-axis.git assistant-axis-ref
```

Requires GPU access and model weights for generation and activation extraction. Uses `google/gemma-2-27b-it` as the primary model.

### Running the pipeline

Run everything for a model with one command:

```bash
./run.sh google/gemma-2-27b-it          # both IV and CAA methods
./run.sh google/gemma-2-27b-it --iv     # instruction-variant only
./run.sh google/gemma-2-27b-it --caa    # CAA only
./run.sh google/gemma-2-27b-it --from 3 # resume from step 3
./run.sh --trajectory                   # OLMo training trajectory pipeline (t1–t4)
```

Or run individual steps (see pipeline table above for full list):

```bash
python pipeline/0_generate_data.py --traits
python pipeline/1_generate.py --model google/gemma-2-27b-it
python pipeline/2_activations.py --model google/gemma-2-27b-it
python pipeline/3_vectors.py --activations-dir outputs/gemma-2-27b-it/activations
python pipeline/4_analysis.py --vectors-dir outputs/gemma-2-27b-it/vectors --layer 22
```

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
WANDB_API_KEY=wandb_v1_...        # Enables experiment tracking (no-op if unset)
WANDB_PROJECT=persona-steering    # W&B project name (default: persona-steering)
```

The `.env` file is loaded automatically by all pipeline scripts via `python-dotenv`.

### Experiment tracking with W&B

W&B is included as a core dependency. When `WANDB_API_KEY` is set in `.env`, each pipeline step logs a W&B run with:
- **Metrics**: cosine similarities, effect sizes, correlations
- **Artifacts**: vectors, analysis results, evaluation scores, figures
- **Images**: all generated figures viewable in the W&B dashboard

Runs are tagged with `model:<name>`, `step:<name>`, and `method:iv` or `method:caa` for filtering. All runs for the same model are grouped together.

To disable W&B (even if the key is set): add `WANDB_DISABLED=true` to `.env`.

Artifact uploads (vectors, activations, responses) are **disabled by default** to avoid W&B storage costs. Metrics, images, and summaries still log normally. To enable artifact uploads, set `WANDB_UPLOAD_ARTIFACTS=true` in `.env`.
