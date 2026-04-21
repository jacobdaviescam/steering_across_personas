# Project Overview — branches, experiments, and next steps

Snapshot assembled 2026-04-14. Pulls together (a) the `experiments/paper-hardening`
branch you've been driving locally, (b) the colleague's `origin/robustness_experiments`
branch on the remote, and (c) the four-experiment first-pass council plan that we
just scaffolded under `pipeline/c{9,1,2,11}_*.py`.

## Branch map

| Branch | Where | Purpose | Last commit |
|---|---|---|---|
| `main` | local + remote | clean baseline (pipeline 0–9, basic analysis) | 2026-04-08 |
| `experiments/paper-hardening` | local (current) + remote | ICML-2026 paper-hardening: E1–E7 + basin geometry + SDL + SAE + W&B + tmux | 2026-04-13 `af3baab` |
| `origin/robustness_experiments` | colleague, remote only | R1–R5 robustness battery + full writeups (IV and CAA) + null/nonsense baselines | 2026-04 (`478578a`) |
| `feature/manifold-probing` | local | earlier bootstrap/manifold probe scaffolding (subsumed by paper-hardening) | 2026-03-10 |
| `feature/activation-oracle` | local | oracle-signal analysis (parked) | 2026-03-18 |
| `experiment/training-trajectory` | local + remote | OLMo training-stage trajectory (T1–T4 scripts) | 2026-03-17 |
| `redesign/concrete-personas` | local + remote | persona redesign (merged into paper-hardening lineage) | 2026-03-10 |

## Experiments already on disk

### `experiments/paper-hardening` (your branch)

| Script | Question | Outputs |
|---|---|---|
| `e1_layer_sweep.py` | How does context dependence of trait vectors vary across layers? | `experiments/layer_sweep.{json,png,pdf}` |
| `e2_bootstrap_rho.py` | Bootstrap stability + correlation between persona-trait structure and residuals | |
| `e3_iv_caa_decomposition.py` | Decompose IV vs CAA vectors into shared + specific | |
| `e4_probe_transfer.py` | Train probes, measure cross-persona transfer (uses SAGA solver for speed) | |
| `e5_sae_features.py` | Project steering vectors onto Gemma Scope JumpReLU SAE features | needs SAE loader |
| `e6_basin_geometry.py`, `e6b_dynamic_activations.py`, `e6c_basin_figures.py` | Basin geometry around the default-Assistant baseline using ringed persona gradients | |
| `e7_sparse_codes.py` | Sparse-code analysis of persona-trait vectors | |
| `t1–t4_trajectory_*.py` | OLMo training trajectory (from `experiment/training-trajectory`) | |

W&B integration lives in `persona_steering/wandb_utils.py`; `run_all_experiments.sh`
orchestrates the lot in tmux.

### `origin/robustness_experiments` (colleague's branch)

Five focused robustness experiments on the canonical 10×8 grid at layer 22,
with writeups under `RobustnessResultsIV.md` and `RobustnessResultsCAA.md`.
Adds `null` and `nonsense` personas as baselines (see `data/personas/{null,nonsense}.yaml`).

| Script | What it measures | Headline result (IV) |
|---|---|---|
| `r1_bootstrap_vectors.py` | 50× bootstrap-resampled vectors per combo; pairwise cosine + full-data alignment | pairwise 0.990 ± 0.008; full-data 0.995 ± 0.004 |
| `r2_convergence.py` | Vector ↔ reference cosine as a function of N pairs ∈ {1,2,5,10,20,50,100,525} | (see W&B run linked in `RobustnessResultsIV.md`) |
| `r3_syntactic_invariance.py` | Within-persona-across-variant vs. across-persona-within-variant cosine + Mann-Whitney | within 0.655 ± 0.140; across 0.719 ± 0.088; p = 0.0072 |
| `r4_general_vs_contextual.py` | Cosine of each persona vector to the general (all-persona-mean) vector, plus null/nonsense baselines | heat map + per-trait ordering by context dependence |
| `r5_context_similarity.py` | Same-context-pair vs. random-pair cosine significance | labeled 0.838; random 0.805; p = 0.091 (trend, not significant) |

CAA results (`RobustnessResultsCAA.md`) mirror the IV pattern: R1 stability
0.989 ± 0.006 despite ~10× fewer contrastive pairs per combo.

Each section links its W&B run so metrics and figures can be opened
directly from the writeup.

## Local state notes

- `outputs/gemma-2-27b-it/activations/` on this machine is empty. Activations,
  steered responses, and anything requiring the 27B weights live on the RunPod
  network volume (not mounted here), so the council scripts in this repo are
  written to run *on the pod* against that tree.
- `outputs/gemma-2-27b-it/vectors/` has 80 persona×trait `.pt` files already,
  so c1/c9 dry-runs that touch only vectors can be sanity-checked locally.
- `origin/robustness_experiments` has NOT been merged into
  `experiments/paper-hardening`. Merging it in (or at least cherry-picking the
  `null` and `nonsense` personas + `r*` scripts) is a prerequisite for the
  council plan's baseline comparisons.

## Merge plan

A clean, low-risk route:

1. Branch `integration/council-first-pass` off `experiments/paper-hardening`.
2. `git merge origin/robustness_experiments` (expect conflicts in `run.sh`,
   `persona_steering/config.py`, `persona_steering/utils.py`, `persona_steering/wandb_utils.py`).
3. Land the council scaffolding (c1/c2/c9/c11 scripts + `persona_steering/council.py`)
   already on this worktree.
4. On the pod, run `c9_residualization.py` first (the gate) against the existing
   activation tree; then `c1_assistant_centroid.py` once the `assistant_default`
   baseline vectors exist; then pilot `c2_shared_specific.py` and `c11_persona_extrapolation.py`.

## First-pass council plan — status

See `run_plan.md` for the full plan. Scaffolding now in place:

| Experiment | Script | Ready to run? |
|---|---|---|
| C9 gate | `pipeline/c9_residualization.py` | yes, pure numpy/torch on existing activations |
| C1 Assistant ≈ centroid | `pipeline/c1_assistant_centroid.py` | yes once `assistant_default_*.pt` vectors exist (need one pipeline pass with empty system prompt) |
| C2 shared+specific behavior | `pipeline/c2_shared_specific.py` | yes, requires GPU and Claude API credits; `--pilot` flag for cheap subset |
| C11 persona extrapolation | `pipeline/c11_persona_extrapolation.py` | yes; small-scale GPU + Claude API |

All four write artifacts to `outputs/{model}/council/{c9,c1,c2,c11}/`. Every
script ends with one summary JSON and one figure that directly answers its
question.

## Decision gates ahead

1. **C9 verdict.** If residualization collapses persona vectors (`cos_resid > 0.95`
   on most traits), reframe the paper around "existing trait vectors are
   dominated by persona baseline drift." If it passes (`cos_resid < 0.9`), the
   manifold story stays and C1/C2/C11 proceed.
2. **C2 pilot.** 3 traits × 5 personas × 3 conditions × 20 generations
   (≈ 900 generations) before committing to the full 4 800-generation sweep.
3. **Merge robustness R-series.** Once council C1/C2 start citing R3 / R4 as
   the robustness baseline, the two branches need to share a commit tree.
