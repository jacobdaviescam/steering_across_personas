# Causal-figures pipeline (X-series)

Adds the three paper-grade experiments — behavioural context-sensitivity (Fig 1),
null-vs-context-trained probes (Fig 2), and the causal alpha sweep (Fig 3) —
on top of the existing `pipeline/0..9` infrastructure. Lives on the
`experiments/causal-figures` branch.

All new outputs go under `outputs/{model}/v2/` so the pre-existing v1
results stay intact for comparison.

---

## Method: CAA as main, IV as behavioural probe

The project settled on CAA (Contrastive Activation Addition) as the main
method for trait vectors. The causal-figures pipeline runs on CAA for
everything that involves trait vectors or probes, **with one exception**:
the X1 context classifier measures *behavioural* context-sensitivity
from free text, and CAA produces only A/B answer-token activations
(no long-form responses). So:

| Script | Method | Why |
|---|---|---|
| x1 classifier | **IV responses** (`outputs/{model}/v2/responses/`) | Needs free-form text; CAA has none. The classifier is a behavioural instrument for Fig 1 / Fig 3, not a trait probe. |
| x2 probes | **CAA activations** (`outputs/{model}/v2/caa_activations/`) | Accepts both key formats (`q{id}` CAA and `v{i}_q{j}` IV) — just point `--activations-dir` at the CAA dir. |
| x3a neutral gen | n/a | Persona system prompt only, no trait manipulation. |
| x3b directions | **CAA vectors** as orth basis | Set `--vectors-dir outputs/{model}/v2/caa_vectors` so the null trait vector used in `u_C - proj(u_C on v_T_null)` is the CAA one. |
| x3c causal sweep | **CAA** for trait basis & probes | `--null-trait-vectors-dir outputs/{model}/v2/caa_vectors`, `--probes-dir outputs/{model}/v2/caa_probes/probes_pkl`. |

This means both pipelines must be run under v2: `pipeline/1`→`pipeline/2`
→`pipeline/3` for IV responses/activations/vectors (consumed by x1), and
`pipeline/0c`→`pipeline/2c`→`pipeline/3c` for CAA (consumed by x2 / x3b
/ x3c). Only x1 depends on IV; everything downstream is CAA.

---

## 0 — Why v2

The v1 persona descriptions named the eight measured traits explicitly
("risk is defining", "deep empathy", "assertiveness manifests as
dominance", etc.) which is circular: we were instructing the trait we then
measured. v2 personas convey the same role through vocation, behaviour,
and concrete detail without the trait words or their direct synonyms. Any
results comparing v1 → v2 belong in the appendix.

The diff lives in `data/personas/*.yaml`. Rebuild **all** generations and
activations from scratch under `v2/` — old responses are not transferable
because the system prompts changed.

---

## 1 — Pre-work

Two things must exist before any task runs:

1. **Null + nonsense generations** through the regular pipeline (`null` and
   `nonsense` are persona slugs; the existing `pipeline/1_generate.py` will
   include them automatically when no `--personas` filter is given).
2. **Neutral question pool** at `data/prompts/neutral.json` — already
   committed, 100 generic questions used to extract context directions
   without trait contamination.

Generate v2 IV responses (for the x1 classifier) and v2 CAA activations
+ vectors (for x2 / x3b / x3c):

```bash
# --- IV side: responses for x1 classifier ---
python pipeline/1_generate.py \
    --model google/gemma-2-27b-it \
    --output-dir outputs/gemma-2-27b-it/v2/responses \
    --n-questions 20

# --- CAA side: activations + vectors for x2 / x3b / x3c ---
python pipeline/2c_caa_activations.py \
    --model google/gemma-2-27b-it \
    --output-dir outputs/gemma-2-27b-it/v2/caa_activations

python pipeline/3_vectors.py \
    --activations-dir outputs/gemma-2-27b-it/v2/caa_activations \
    --output-dir outputs/gemma-2-27b-it/v2/caa_vectors
```

Skip IV activations/vectors unless you also want the IV-vs-CAA appendix
comparison — x1 only needs the IV *responses*, not the activations.

---

## 2 — Task 1: context classifier (Fig 1)

Trains an SBERT-frozen + linear-head classifier to predict which context
generated a response. Per-trait accuracy is the behavioural
context-sensitivity score.

```bash
python pipeline/x1_context_classifier.py \
    --responses-dir outputs/gemma-2-27b-it/v2/responses \
    --output-dir outputs/gemma-2-27b-it/v2/classifier
```

**Controls** (runs in the appendix):

```bash
# chance baseline
python pipeline/x1_context_classifier.py ... --shuffle-labels \
    --output-dir outputs/gemma-2-27b-it/v2/classifier_chance

# lexical-shortcut control
python pipeline/x1_context_classifier.py ... --mask-entities \
    --output-dir outputs/gemma-2-27b-it/v2/classifier_masked
```

Outputs: `head.pt`, `metrics.json` (per-trait + confusion matrix),
`predictions.jsonl`, `splits.json` (held-out question IDs — Task 2 reuses
these for split alignment).

Expected sanity headline: mean accuracy well above 1/12 ≈ 0.083; honesty
likely lowest, risk-taking likely highest. Null tends to be hardest to
classify — that's correct, not a bug.

---

## 3 — Task 2: probe regimes (Fig 2)

Three regimes per trait:
- **A** — train on null-context activations only
- **B** — train on 11 contexts (leave one out), eval on the held-out one
- **B-parity** — subsample B's training set to match A's size
- **C** (appendix) — full 12×12 train-context × eval-context AUROC matrix

```bash
python pipeline/x2_probe_regimes.py \
    --activations-dir outputs/gemma-2-27b-it/v2/caa_activations \
    --output-dir outputs/gemma-2-27b-it/v2/caa_probes \
    --classifier-splits outputs/gemma-2-27b-it/v2/classifier/splits.json \
    --layer 22
```

Outputs: `probes_pkl/{trait}_A_null.pkl` (null probes — Task 3 will
re-load these), `auroc_matrix_{trait}.npy`, `metrics.json`.

Headline: null-trained AUROC drops 10–20 pp off-diagonal; multi-context
holds within ~5 pp of within-context ceiling.

---

## 4 — Task 3: causal sweep (Fig 3)

Four sub-steps. The first two are one-shot; the third (`x3c`) is the
expensive sweep.

### 4a — Neutral-prompt generations

```bash
python pipeline/x3a_neutral_generations.py \
    --model google/gemma-2-27b-it \
    --output-dir outputs/gemma-2-27b-it/v2/neutral_responses \
    --n-prompts 50
```

### 4b — Activations on neutral responses

Reuses `pipeline/2_activations.py`, just pointed at the neutral dir:

```bash
python pipeline/2_activations.py \
    --model google/gemma-2-27b-it \
    --responses-dir outputs/gemma-2-27b-it/v2/neutral_responses \
    --output-dir outputs/gemma-2-27b-it/v2/neutral_activations
```

### 4c — Context directions + per-trait orthogonalization

```bash
python pipeline/x3b_context_directions.py \
    --neutral-activations-dir outputs/gemma-2-27b-it/v2/neutral_activations \
    --vectors-dir outputs/gemma-2-27b-it/v2/caa_vectors \
    --output-dir outputs/gemma-2-27b-it/v2/causal/directions \
    --layer 22
```

Outputs `u_{C}.pt` (raw context direction) and `u_{C}_{T}_orth.pt`
(orthogonalised against null trait vector). Pairs with
`||u_orth|| / ||u_C|| < 0.5` are flagged in `directions_summary.json` —
they indicate context/trait entanglement and are reported separately.

### Two-probe test

The sweep evaluates **both** the null-trained probe (from Regime A) and
the context's **within-context probe** (from Regime C diagonal, saved by
x2 as `{trait}_within_{context}.pkl`). Expected signature:

- `auroc_null` **decreases** as α rises — null probe loses its grip because
  activations leave the region it was trained on.
- `auroc_within` **increases** as α rises — activations move into context
  C's native region, where C's own probe was trained.

Together these pin down the causal claim: steering isn't just "breaking"
a probe, it's relocating activations to a different, context-specific
region.

### 4d — Alpha sweep — pilot first, scale on success

**Pilot** (1 trait × 1 context × 5 alphas × 5 pairs, ~150 generations):

```bash
python pipeline/x3c_causal_sweep.py \
    --model google/gemma-2-27b-it \
    --directions-dir outputs/gemma-2-27b-it/v2/causal/directions \
    --null-trait-vectors-dir outputs/gemma-2-27b-it/v2/caa_vectors \
    --classifier-dir outputs/gemma-2-27b-it/v2/classifier \
    --probes-dir outputs/gemma-2-27b-it/v2/caa_probes/probes_pkl \
    --output-dir outputs/gemma-2-27b-it/v2/causal \
    --pilot --traits honesty --contexts therapist
```

Inspect `outputs/.../causal/metrics/sweep_results.json`. Expected pilot
headline: monotone rising classifier P(C), monotone falling probe AUROC
across the alpha axis. **If either trend is missing or flat — stop and
debug before scaling.** Common debug paths: tighten the eliciting pairs,
re-pilot at a different alpha range, or check that the orthogonalised
direction has reasonable norm.

**Full main run** (all traits × all personas × full alpha grid):

```bash
python pipeline/x3c_causal_sweep.py \
    --model google/gemma-2-27b-it \
    --directions-dir outputs/gemma-2-27b-it/v2/causal/directions \
    --null-trait-vectors-dir outputs/gemma-2-27b-it/v2/caa_vectors \
    --classifier-dir outputs/gemma-2-27b-it/v2/classifier \
    --probes-dir outputs/gemma-2-27b-it/v2/caa_probes/probes_pkl \
    --output-dir outputs/gemma-2-27b-it/v2/causal \
    --conditions main \
    --alphas 0 1 2 4 8 12 16 24 32 \
    --n-pairs 20
```

**Controls** (smaller subset to keep budget down — see design doc:
3 traits × 3 contexts is enough for the appendix):

```bash
python pipeline/x3c_causal_sweep.py \
    ... \
    --conditions rand trait \
    --traits honesty assertiveness empathy \
    --contexts therapist tech_ceo drill_sergeant \
    --alphas 0 1 2 4 8 16 \
    --output-dir outputs/gemma-2-27b-it/v2/causal_controls
```

---

## 5 — Figures

```bash
python pipeline/x4_figures.py \
    --classifier-dir outputs/gemma-2-27b-it/v2/classifier \
    --probes-dir outputs/gemma-2-27b-it/v2/caa_probes \
    --sweep-results outputs/gemma-2-27b-it/v2/causal/metrics/sweep_results.json \
    --output-dir outputs/gemma-2-27b-it/v2/figures
```

Pass `--no-fig3` while the sweep is still running.

---

## Weights & Biases

All x-scripts call `init_run` / `log_metrics` / `log_summary` /
`log_artifact` / `finish_run` via `persona_steering.wandb_utils`. Runs are
grouped under `method:causal-figures` — filter that tag to see this
pipeline's runs. Each script tags itself with its step name
(`x1_classifier`, `x2_probes`, `x3a_neutral_gen`, `x3b_directions`,
`x3c_causal_sweep`, `x4_figures`) so you can compare across models.

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=persona-steering   # default if unset
export WANDB_UPLOAD_ARTIFACTS=true       # opt-in: push heads/probes/directions
```

Without `WANDB_API_KEY`, every wandb call is a silent no-op — the
scripts still run normally and results land in `outputs/{model}/v2/...`.
Setting `WANDB_DISABLED=true` forces the no-op path even with a key.

Per-step logging:

| Script | W&B metrics | Summary | Artifact (if `WANDB_UPLOAD_ARTIFACTS=true`) |
|---|---|---|---|
| x1 | per-epoch train/loss + val/accuracy; per-trait accuracy | overall_accuracy, best_val | `{model}-x1-classifier` |
| x2 | — | per-trait A_mean / B_mean / Bparity_mean | `{model}-x2-probes` |
| x3a | files_done / total | — | `{model}-x3a-neutral-responses` |
| x3b | — | n_entangled_pairs, entanglement_threshold | `{model}-x3b-directions` |
| x3c | per (cond,trait,ctx): p_context, auroc, coherence | n_sweep_points, conditions | `{model}-x3c-causal` |
| x4 | figure PNGs | — | `{model}-x4-figures` |

---

## Shared decisions

These are pinned across all three tasks:

| Setting | Value |
|---|---|
| Response length cap | 256 tokens |
| Activation extraction layer | 22 (mid-layer for Gemma-2-27b-it) |
| Held-out questions | 20 per trait, by question ID |
| Random seed | 42 |
| LLM judge model | claude-sonnet-4-20250514 |
| SBERT backbone | all-mpnet-base-v2 |

---

## Compute budget — full run

| Stage | Approx. cost |
|---|---|
| v2 generations + activations | one-off, ~existing pipeline cost |
| Task 1 classifier | minutes on CPU/MPS |
| Task 2 probes | minutes on CPU |
| Task 3a neutral generations | ~600 generations |
| Task 3c full sweep (main) | ~32k generations + ~32k Claude calls |
| Task 3c controls | ~6k generations + ~6k Claude calls |

Use `--skip-judge` on x3c to drop Claude API cost during early iteration —
coherence will be unavailable for those runs but the phase-portrait shape
still works.

---

## Pitfalls worth re-reading before each scale-up

- **Null is the hardest context to classify.** The default-Assistant
  voice sits near the centre of the activation manifold; expect Task 1
  accuracy on null to be lower than on persona contexts.
- **Saturation at high alpha.** Outputs go degenerate above a
  context-dependent alpha. Use coherence to clip the sweep — failures at
  degenerate alpha don't prove causality, they prove breakage.
- **Orthogonalisation collapse.** When `||u_orth|| / ||u_C|| < 0.5` the
  context direction is dominated by trait structure. These pairs stay in
  the data but get reported separately. Check
  `directions_summary.json:entangled_pairs`.
- **Variance across prompt pairs.** 20 pairs × 2 directions = 40 samples
  for the AUROC estimate. If error bars are wide, expand the pair pool
  in `data/prompts/eliciting_pairs.json`.
