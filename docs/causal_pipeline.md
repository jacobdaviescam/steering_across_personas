# Causal-figures pipeline (X-series)

Adds the three paper-grade experiments — behavioural context-sensitivity (Fig 1),
null-vs-context-trained probes (Fig 2), and the causal alpha sweep (Fig 3) —
on top of the existing `pipeline/0..9` infrastructure. Lives on the
`experiments/causal-figures` branch.

All new outputs go under `outputs/{model}/v2/` so the pre-existing v1
results stay intact for comparison.

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

Generate v2 responses + activations + null-trait vectors first:

```bash
# v2 generations (all 12 contexts × 8 traits × 100 questions × pos/neg × 5 variants)
python pipeline/1_generate.py \
    --model google/gemma-2-27b-it \
    --output-dir outputs/gemma-2-27b-it/v2/responses \
    --n-questions 20

# v2 activations
python pipeline/2_activations.py \
    --model google/gemma-2-27b-it \
    --responses-dir outputs/gemma-2-27b-it/v2/responses \
    --output-dir outputs/gemma-2-27b-it/v2/activations

# v2 IV trait vectors (needed by x3b for null-trait basis)
python pipeline/3_vectors.py \
    --activations-dir outputs/gemma-2-27b-it/v2/activations \
    --output-dir outputs/gemma-2-27b-it/v2/vectors
```

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
    --activations-dir outputs/gemma-2-27b-it/v2/activations \
    --output-dir outputs/gemma-2-27b-it/v2/probes \
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
    --vectors-dir outputs/gemma-2-27b-it/v2/vectors \
    --output-dir outputs/gemma-2-27b-it/v2/causal/directions \
    --layer 22
```

Outputs `u_{C}.pt` (raw context direction) and `u_{C}_{T}_orth.pt`
(orthogonalised against null trait vector). Pairs with
`||u_orth|| / ||u_C|| < 0.5` are flagged in `directions_summary.json` —
they indicate context/trait entanglement and are reported separately.

### 4d — Alpha sweep — pilot first, scale on success

**Pilot** (1 trait × 1 context × 5 alphas × 5 pairs, ~150 generations):

```bash
python pipeline/x3c_causal_sweep.py \
    --model google/gemma-2-27b-it \
    --directions-dir outputs/gemma-2-27b-it/v2/causal/directions \
    --null-trait-vectors-dir outputs/gemma-2-27b-it/v2/vectors \
    --classifier-dir outputs/gemma-2-27b-it/v2/classifier \
    --probes-dir outputs/gemma-2-27b-it/v2/probes/probes_pkl \
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
    --null-trait-vectors-dir outputs/gemma-2-27b-it/v2/vectors \
    --classifier-dir outputs/gemma-2-27b-it/v2/classifier \
    --probes-dir outputs/gemma-2-27b-it/v2/probes/probes_pkl \
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
    --probes-dir outputs/gemma-2-27b-it/v2/probes \
    --sweep-results outputs/gemma-2-27b-it/v2/causal/metrics/sweep_results.json \
    --output-dir outputs/gemma-2-27b-it/v2/figures
```

Pass `--no-fig3` while the sweep is still running.

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
