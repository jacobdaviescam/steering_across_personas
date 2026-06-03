# Persona-Conditional Steering Vectors

**Do steering vectors for the same trait change depending on which persona a model is operating under?**

We extract per-persona steering vectors for 8 traits across concrete character personas
(farmer, politician, surgeon, …) and test whether trait geometry is persona-specific or
trait-universal — with implications for whether safety interventions generalise across
deployment contexts. Headline result: vectors are predominantly trait-universal but
meaningfully persona-conditioned, and **post-training (not pre-training) creates that
conditioning**. Full write-up in [docs/overview.md](docs/overview.md).

## Documentation

| Doc | What's in it |
|-----|--------------|
| [docs/overview.md](docs/overview.md) | Research question, method, headline findings, safety implications — **read this first** |
| [docs/experiments.md](docs/experiments.md) | Runbook: exact commands for each extended experiment (E2–E7) |
| [docs/causal_pipeline.md](docs/causal_pipeline.md) | The causal-figures X-series pipeline |
| [docs/results/](docs/results/) | Full results: [summary](docs/results/summary.md) (multi-model), [robustness_iv](docs/results/robustness_iv.md), [robustness_caa](docs/results/robustness_caa.md), [gemma4_e4b](docs/results/gemma4_e4b.md) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, the branch→PR workflow, and the data policy |

## Data — no GPU needed to start

All activations and steering vectors are published on the Hugging Face Hub:
🤗 **[girishgupta/persona-steering-activations](https://huggingface.co/datasets/girishgupta/persona-steering-activations)**
(v2: IV + CAA activations, vectors, responses, persona YAMLs, and prompts for **17 personas × 8 traits** on `google/gemma-2-27b-it`).

```bash
# Just the steering vectors (~110 MB) — enough to start analysing
huggingface-cli download girishgupta/persona-steering-activations vectors/ \
    --repo-type dataset --local-dir ./outputs/gemma-2-27b-it
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -e ".[dev]"
```

Then work through the **[notebooks/](notebooks/)** — a CPU-only onboarding series that
loads the vectors above and reproduces the headline findings. New contributors should
read [CONTRIBUTING.md](CONTRIBUTING.md) and pick up an
[issue](https://github.com/jacobdaviescam/steering_across_personas/issues).

## Personas & traits

**10 core archetypes:** farmer, politician, therapist, drill sergeant, street hustler,
professor, tech CEO, kindergarten teacher, surgeon, con artist. The published dataset
extends to **17** (adds control personas `null`/`nonsense` and extensions like
`sociopath`, `pathological_liar`, `six_year_old`). Configs live in `data/personas/`.

**8 traits:** assertiveness, empathy, risk-taking, honesty, confidence, deference,
warmth, impulsivity.

Two extraction methods are compared: **Instruction-Variant (IV)** — same question under
opposing trait instructions — and **Contrastive Activation Addition (CAA)** — forced A/B
choices. The contrastive vector is `mean(pos) − mean(neg)`. Details in
[docs/overview.md](docs/overview.md).

## Pipeline

Numbered scripts in `pipeline/`:
`0_generate_data` → `1_generate` → `2_activations` → `3_vectors` → `4_analysis` →
evaluation/steering (`6`–`9`). Steps 3+ are shared between IV and CAA. Run everything
for a model with:

```bash
./run.sh google/gemma-2-27b-it          # both IV and CAA
./run.sh google/gemma-2-27b-it --from 3 # resume from analysis (e.g. with HF vectors)
```

GPU + model weights are only needed for generation/extraction (steps 1–2); analysis runs
on CPU from the HF vectors. Full per-step commands and the extended experiments are in
[docs/experiments.md](docs/experiments.md).

## Setup notes

- Steering/extraction uses the [`assistant_axis`](https://github.com/safety-research/assistant-axis)
  reference: `git clone https://github.com/safety-research/assistant-axis.git assistant-axis-ref`
- Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (LLM judge / data generation)
  and `HF_TOKEN` (gated models). W&B logging activates automatically if `WANDB_API_KEY` is set.

Based on the assistant axis from [Lu et al. (2026)](https://arxiv.org/abs/2601.10387) and
Persona Vectors [Chen et al. (2025)](https://arxiv.org/abs/2507.21509).
