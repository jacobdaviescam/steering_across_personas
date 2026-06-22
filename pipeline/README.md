# Pipeline map

This folder has grown to ~60 scripts across several experiment families. Each file is
prefixed by the family it belongs to. **Start with the core numbered pipeline (0–11);
everything else is an experiment thread that builds on its outputs.**

New here? Read [`../docs/overview.md`](../docs/overview.md) first, then run the
[`../notebooks/`](../notebooks/) (CPU-only, no pipeline needed), then come back here when
you want to regenerate data or run an experiment.

Convention: scripts import the `persona_steering` package and take `--help`. Most read
from / write to `outputs/{model}/...` (gitignored; pull the data from the
[HF dataset](https://huggingface.co/datasets/girishgupta/persona-steering-activations)).

## Core pipeline (0–11) — the canonical flow

The backbone. Steps 3+ are shared between the IV and CAA extraction methods; point them
at the right input dir. Run the whole thing with `../run.sh <model>`.

| Step | Script | What it does |
|---|---|---|
| 0 | `0_generate_data.py` / `0c_generate_caa_data.py` | Generate trait datasets (IV instruction-variants / CAA A-B) via Claude API |
| 1 | `1_generate.py` | vLLM responses for every persona×trait×direction (IV) |
| 2 | `2_activations.py` / `2c_caa_activations.py` | Extract mean activations (IV full-response / CAA answer-token) |
| 3 | `3_vectors.py` | Contrastive vectors: `mean(pos) − mean(neg)` |
| 4 | `4_analysis.py` | Transfer matrices, clustering, shared/specific decomposition, axis alignment |
| 5 | `5_visualize.py`, `5b_persona_landscape.py`, `5_activation_landscape.py` | Publication figures / landscape geometry |
| 6–7 | `6_behavioral_eval.py`, `7_eval_analysis.py` | LLM-as-judge behavioural scoring + analysis |
| 8–9 | `8_steered_generation.py`, `9_steering_eval.py` | Apply steering vectors across personas + evaluate |
| 10–11 | `10_oracle.py`, `11_oracle_analysis.py` | Activation-oracle interpretation of vectors |

## Experiment families (prefixed)

Each family is a self-contained thread. The **Docs** column is the authoritative writeup;
read it before running the scripts.

| Prefix | Theme | Key scripts | Docs |
|---|---|---|---|
| `e1`–`e7` | **Extended / paper-hardening**: layer sweep, bootstrap ρ, IV–CAA decomposition, probe transfer, SAE features, basin geometry, sparse codes | `e2_bootstrap_rho`, `e4_probe_transfer`, `e6_basin_geometry` | [`../docs/experiments.md`](../docs/experiments.md) |
| `r1`–`r5` | **Robustness battery**: bootstrap stability, convergence, syntactic/phrasing invariance, general-vs-contextual, context similarity | `r1_bootstrap_vectors`, `r3_syntactic_invariance` | [`../docs/results/robustness_iv.md`](../docs/results/robustness_iv.md), [`robustness_caa.md`](../docs/results/robustness_caa.md) |
| `x1`–`x9` | **Causal-figures (X-series)**: context classifier, probe regimes, context directions, causal steering sweep, probe cross-transfer, steer-then-probe | `x3c_causal_sweep`, `x5_probe_cross_transfer`, `x8_steer_then_probe` | [`../docs/causal_pipeline.md`](../docs/causal_pipeline.md), [`../docs/findings.md`](../docs/findings.md) |
| `t1`–`t4` | **Training trajectory**: extract/analyse trait vectors across OLMo checkpoints | `t1_trajectory_activations`, `t3_trajectory_analysis` | [`../docs/overview.md`](../docs/overview.md) (trajectory section) |
| `n1`,`n3`,`n4` + `a1`–`a3` | **Naturalistic & adversarial**: free-form generation, LLM judge, adversarial questions | `n1_naturalistic_generate`, `a2_adversarial_generate` | [`RUNBOOK_naturalistic_adversarial.md`](RUNBOOK_naturalistic_adversarial.md) |
| `c1`,`c2`,`c9`,`c11` | **"Council" probes**: assistant centroid, shared/specific behaviour, residualization, persona extrapolation | `c2_shared_specific`, `c9_residualization` | [`../docs/findings.md`](../docs/findings.md) |
| `p1` | **Persona-residue classifier** on steered outputs (the paper's residue result) | `p1_classifier_on_steered` | [`../docs/x8_steer_then_probe.md`](../docs/x8_steer_then_probe.md) |
| `sae` | Standalone SAE feature comparison (Gemma Scope 1/2) | `sae_experiment` | [`../docs/results/summary.md`](../docs/results/summary.md) |

## Conventions

- `--help` on any script lists its flags. Inputs/outputs are `outputs/{model}/...`.
- IV vs CAA is chosen by which input dir you pass (`activations/` vs `caa_activations/`).
- Plotting helpers share a script's prefix with a trailing letter (e.g. `x4b`, `x8b`).
- **Do not commit anything under `outputs/`** — see [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
