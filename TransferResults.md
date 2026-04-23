# Probe Transfer Results (x5 + x6)

Documentation of the cross-context probe transfer experiment and the correlation between vector dissimilarity and probe transfer. Runs on Gemma 2 27B IT, layer 22, using v2 data (corrected persona prompts with no explicit trait mentions).

Two probe-training methods are evaluated side-by-side:

- **CAA → IV:** probe trained on CAA activations, evaluated on IV activations
- **IV → IV:** probe trained on IV activations, evaluated on IV activations

---

## 1. What was asked

From the team meeting preceding this work, two questions:

1. **Do probes trained on one persona's data still work on a different persona's data?** Train a probe to detect "honest vs deceptive" using farmer's activations; test whether it still works on politician's activations. Full 12×12 pairwise matrix ("full circle") across all persona pairs.
2. **Is there any shared variance between how different two personas' trait vectors are, and how well the probe transfers between them?** Hypothesis: more different vectors → worse probe transfer. Suggested analysis: cosine similarity between vectors on one axis, probe effectiveness on the other.

x5 answers question 1. x6 answers question 2.

---

## 2. Data and design decisions

### Personas (12)

10 concrete personas plus 2 baselines. From `persona_steering.config.PERSONA_SLUGS`:

`con_artist, drill_sergeant, farmer, kindergarten_teacher, nonsense, null, politician, professor, street_hustler, surgeon, tech_ceo, therapist`

- `null` = no system prompt (model default)
- `nonsense` = gibberish system prompt (control for "having any system prompt at all")

### Traits (8)

`assertiveness, confidence, deference, empathy, honesty, impulsivity, risk_taking, warmth`

### Why both CAA→IV and IV→IV?

There are three conceivable training/eval method combinations. Each has different trade-offs:

| Combination | What we get | Why we (don't) use it |
|---|---|---|
| **CAA → CAA** | Probe trained and tested on contrastive A/B answer-token activations. | **Useless — saturates at 1.0 everywhere.** The CAA scenario presents the model with a two-answer choice, and the activation at the answer token trivially encodes which letter was chosen. Any probe hits near-perfect AUROC. Confirmed by Jacob's `auroc_matrix_*.npy` files (all cells = 1.0). Gives no signal for probe-transfer questions. |
| **CAA → IV** | Probe trained on clean contrastive signal, tested on free-form text responses. | Was run. Real numbers. But mixes a persona shift with a method shift (probe trained on CAA but tested on IV). The within-vs-cross comparison still isolates the persona effect (both diagonal and off-diagonal share the method shift), but absolute AUROCs are lower than pure same-method evaluation. |
| **IV → IV** | Probe trained and tested on IV activations. | Was run as a cross-check. No method shift. Any within-vs-cross gap is purely the persona effect. Caveat: diagonal cells have train/test leakage (probe trained on persona X's IV activations, tested on same X activations) → diagonals are optimistically inflated. Off-diagonal cells have no leakage. |

Running both gives us two independent views of question 1. If the off-diagonal pattern is similar between CAA→IV and IV→IV, we have confidence the finding is not a method artifact.

### Metric choice

**AUROC** — Area Under the ROC Curve. Inherited from Jacob's `pipeline/x2_probe_regimes.py` which uses `roc_auc_score` throughout. AUROC answers: "if I pick a random positive sample and a random negative sample, what's the probability the probe scores the positive one higher?" 1.0 = perfect, 0.5 = random guessing.

### Vector distance choice

**1 − cos(v_i, v_j)** — simple cosine distance. Cosine similarity was explicitly mentioned as the candidate metric in the meeting. Subtracting from 1 converts similarity to a distance so "more different" is a larger number, which means the hypothesized correlation is negative (more different → worse transfer).

### Which vectors go on the x-axis of x6?

For each probe-training method, we use the same-method vectors on the x-axis:

- **CAA→IV probe-transfer matrix** is correlated against **CAA trait vectors** (`v2/caa_vectors/`).
- **IV→IV probe-transfer matrix** is correlated against **IV trait vectors** (`v2/vectors/`).

This keeps the x-axis consistent with how the probe was constructed: the probe was trained on some method's activations, so the trait vector that geometrically characterises that method's representation is the right reference.

### Train/test leakage

- **CAA→IV diagonal:** probe was trained on CAA data; evaluation uses IV data. Different data sources → no direct leakage. Diagonal AUROCs are clean.
- **IV→IV diagonal:** probe was trained on IV data; evaluation uses all IV data (same distribution). If Jacob used a held-out split during training, some of the test data overlaps with training data. Diagonal AUROCs are optimistically inflated.
- **Off-diagonal cells** (both methods): probe trained on persona A's data, tested on persona B's data. No leakage either way. The x6 correlation is unaffected in both methods.

---

## 3. Data sources

All paths relative to `/workspace/steering_across_personas/` on the pod.

| Input | Path | Notes |
|---|---|---|
| IV activations (eval side for both methods) | `outputs/gemma-2-27b-it/v2/activations/{persona}_{trait}_{pos,neg}.pt` | Dict of 100 entries per file keyed `v{variant}_q{question}`, tensor shape `(46, 4608)`. We use layer 22. |
| CAA probes | `outputs/gemma-2-27b-it/v2/caa_probes/probes_pkl/{trait}_within_{context}.pkl` | Trained by Jacob's `x2_probe_regimes.py` on CAA activations. |
| IV probes | `outputs/gemma-2-27b-it/v2/iv_probes/probes_pkl/{trait}_within_{context}.pkl` | Trained by Jacob's `x2_probe_regimes.py` on IV activations. |
| CAA vectors (x-axis for CAA→IV x6) | `outputs/gemma-2-27b-it/v2/caa_vectors/{persona}_{trait}.pt` | From `3_vectors.py` on CAA activations. |
| IV vectors (x-axis for IV→IV x6) | `outputs/gemma-2-27b-it/v2/vectors/{persona}_{trait}.pt` | From `3_vectors.py` on IV activations. Generate with `python pipeline/3_vectors.py --activations-dir outputs/gemma-2-27b-it/v2/activations --output-dir outputs/gemma-2-27b-it/v2/vectors` if not present. |

---

## 4. Scripts

### `pipeline/x5_probe_cross_transfer.py`

**What it does:** evaluates every within-context probe on every persona's IV activations to produce a 12×12 AUROC matrix per trait. Works with either CAA-trained or IV-trained probes — point `--probes-dir` at the appropriate directory. The W&B tag and heatmap titles auto-detect the probe method from the path.

**Algorithm:**
1. Load all pos and neg IV activations at layer 22 from `v2/activations/` for each (context, trait). Label pos as 1, neg as 0.
2. For each trait:
   - For each train_context:
     - Load `{trait}_within_{train_context}.pkl` → unpickle probe and scaler.
     - For each eval_context:
       - Apply scaler to eval_context's activations, run `probe.decision_function(X)` to get scores.
       - Compute `roc_auc_score(y_true, scores)` → one AUROC value. Store at `mat[train, eval]`.
   - Save matrix as `cross_transfer_{trait}.npy` (12×12).
   - Plot heatmap as `cross_transfer_{trait}.png`.
3. Save per-trait mean diagonal (within) and mean off-diagonal (cross) to `cross_transfer_summary.json`.

**Commands:**

CAA → IV:
```
python pipeline/x5_probe_cross_transfer.py --activations-dir outputs/gemma-2-27b-it/v2/activations --probes-dir outputs/gemma-2-27b-it/v2/caa_probes/probes_pkl --output-dir outputs/gemma-2-27b-it/v2/caa_probes --layer 22
```

IV → IV:
```
python pipeline/x5_probe_cross_transfer.py --activations-dir outputs/gemma-2-27b-it/v2/activations --probes-dir outputs/gemma-2-27b-it/v2/iv_probes/probes_pkl --output-dir outputs/gemma-2-27b-it/v2/iv_probes --layer 22
```

**Outputs (per run):**
- `cross_transfer_{trait}.npy` (× 8) — raw 12×12 matrices
- `cross_transfer_{trait}_contexts.json` (× 8) — context ordering + per-cell values
- `cross_transfer_{trait}.png` (× 8) — heatmaps
- `cross_transfer_summary.json` — per-trait mean within/cross

### `pipeline/x6_correlation.py`

**What it does:** correlates vector distance with probe transfer, using x5's matrices. Auto-detects probe method from the `--matrix-dir` path.

**Algorithm:**
1. For each trait, load `cross_transfer_{trait}.npy` and context ordering.
2. For every ordered pair (i, j) with i ≠ j:
   - Load vectors `{context_i}_{trait}.pt` and `{context_j}_{trait}.pt` from `--vectors-dir`. Take layer 22.
   - Compute `x = 1 − cos(vec_i, vec_j)`.
   - Read `y = mat[i, j]`.
3. Compute per-trait Pearson r and p. Compute aggregate Pearson r and p.
4. Plot one aggregate scatter + 8-panel per-trait scatter grid.

**Commands:**

CAA → IV correlation (CAA vectors on x-axis):
```
python pipeline/x6_correlation.py --matrix-dir outputs/gemma-2-27b-it/v2/caa_probes --vectors-dir outputs/gemma-2-27b-it/v2/caa_vectors --output-dir outputs/gemma-2-27b-it/v2/x6_correlation_caa --layer 22
```

IV → IV correlation (IV vectors on x-axis):
```
python pipeline/x6_correlation.py --matrix-dir outputs/gemma-2-27b-it/v2/iv_probes --vectors-dir outputs/gemma-2-27b-it/v2/vectors --output-dir outputs/gemma-2-27b-it/v2/x6_correlation_iv --layer 22
```

**Outputs (per run):**
- `summary.json` — aggregate and per-trait Pearson stats
- `scatter.png` — aggregate scatter
- `scatter_per_trait.png` — 8-panel per-trait grid

---

## 5. Results — CAA → IV

### Per-trait summary (x5)

From `v2/caa_probes/cross_transfer_summary.json`:

| Trait | Within (mean diagonal) | Cross (mean off-diagonal) | Drop |
|---|---|---|---|
| assertiveness | 0.848 | 0.841 | +0.007 |
| empathy | 0.837 | 0.791 | +0.046 |
| risk_taking | 0.915 | 0.919 | −0.004 |
| honesty | 0.745 | 0.733 | +0.011 |
| confidence | 0.744 | 0.725 | +0.019 |
| deference | 0.585 | 0.558 | +0.027 |
| warmth | 0.907 | 0.893 | +0.014 |
| impulsivity | 0.710 | 0.686 | +0.024 |

**Overall:** mean within = 0.786, mean cross = 0.768. Average drop = 0.018 AUROC points.

**W&B:** https://wandb.ai/persona-steering/persona-steering/runs/f1318usl

### Correlation (x6)

From `v2/x6_correlation_caa/summary.json`:

- **n = 1056**, **Pearson r = −0.121**, **p = 8.4 × 10⁻⁵**

| Trait | Pearson r | p |
|---|---|---|
| confidence | **−0.506** | **5.8 × 10⁻¹⁰** |
| assertiveness | **−0.344** | **5.5 × 10⁻⁵** |
| deference | −0.143 | 0.10 |
| warmth | −0.103 | 0.24 |
| empathy | −0.093 | 0.29 |
| impulsivity | −0.086 | 0.33 |
| honesty | −0.024 | 0.79 |
| risk_taking | +0.073 | 0.41 |

**W&B:** https://wandb.ai/persona-steering/persona-steering/runs/g6cbbmfg

---

## 6. Results — IV → IV

*To be filled after running both scripts with the IV-probe paths.*

### Per-trait summary (x5)

| Trait | Within (mean diagonal) | Cross (mean off-diagonal) | Drop |
|---|---|---|---|
| assertiveness | — | — | — |
| empathy | — | — | — |
| risk_taking | — | — | — |
| honesty | — | — | — |
| confidence | — | — | — |
| deference | — | — | — |
| warmth | — | — | — |
| impulsivity | — | — | — |

**Note:** diagonal values will be optimistically inflated because of train/test leakage (the IV probe was trained on the same persona's IV activations it's now being evaluated on). Off-diagonal values are clean.

**W&B:** *to be filled*

### Correlation (x6)

*To be filled.*

**W&B:** *to be filled*

---

## 7. Plot legends (apply to both methods)

### Heatmap — how to read

Each heatmap is one trait for one probe method. Saved as `cross_transfer_{trait}.png`.

- **Rows (y-axis):** which context the probe was trained on (12 contexts alphabetically).
- **Columns (x-axis):** which context's IV activations the probe was tested on (same 12).
- **Cell value:** AUROC of that specific probe on that specific eval context. Printed to 2 decimals.
- **Color:** RdYlGn from 0.5 (red, random) to 1.0 (green, perfect).
- **Title:** probe method (CAA or IV), trait name, plus within (mean diagonal), cross (mean off-diagonal), and drop.

**What to look for:** if probes fail to transfer across contexts, the diagonal should be visibly greener than the off-diagonal cells.

### Aggregate scatter (x6)

File: `scatter.png`. All ~1000 points pooled across traits.

- **x-axis, "Vector distance (1 − cos)":** `1 − cos(vec_i, vec_j)` where i and j are distinct contexts.
- **y-axis, "Probe transfer (AUROC)":** AUROC of probe trained on i, tested on j's IV data.
- **Dots:** one per (i, j) pair per trait, with i ≠ j.
- **Dashed line:** linear least-squares fit with Pearson r and p in the legend.

### Per-trait scatter grid (x6)

File: `scatter_per_trait.png`. Same data split by trait, 2×4 grid.

- **Each panel:** x and y axes as above, zoomed to that trait's range.
- **Dots:** ~132 points per panel (12 × 11 ordered pairs).
- **Title:** trait name, per-trait Pearson r and p.

---

## 8. Headline findings (CAA → IV, pending IV → IV)

**Question 1 — do probes transfer?** Yes, with minimal loss. Within-vs-cross AUROC drop is under 0.05 on every trait under CAA → IV. The hypothesis that "a probe trained on farmer won't work on politician" is not supported.

**Question 2 — vector difference vs probe transfer?** Weakly yes, in the hypothesized direction. Aggregate r = −0.12 (p < 0.001). Effect is concentrated in confidence (r = −0.51) and assertiveness (r = −0.34); other traits show little to no relationship.

**Implication:** representational deviations across personas (measured geometrically) do not translate strongly into probe transfer failures. The persona-specific variance in the activations appears to live in a subspace largely orthogonal to the direction the probe uses. This decouples "representations differ" (shown by R4/R5 in the robustness work) from "probes fail to transfer."

The IV → IV results will test whether this pattern is a property of the probes themselves or an artifact of the CAA → IV method shift.

---

## 9. Caveats

1. **CAA→IV method shift.** Every AUROC under CAA → IV combines a method shift with a persona shift. Within-vs-cross comparison isolates the persona effect, but absolute AUROCs are lower than pure same-method evaluation.
2. **IV→IV diagonal leakage.** Diagonals are inflated because the probe was trained on the same persona's IV activations it's being tested on. Off-diagonal values drive the x6 correlation and are not affected.
3. **Only Gemma 2 27B IT.** The v2 pipeline has only been run on Gemma 2 so far.
4. **Small per-trait n.** Each per-trait correlation uses ~132 points. Two traits hit p < 0.001; four are p > 0.2.
