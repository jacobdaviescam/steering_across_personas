# Robustness Results (CAA, 10 personas x 8 traits, Gemma-2-27B-IT, layer 22)

---

## R1: Bootstrap Stability

### What problem does it solve?

Same as for IV: are the CAA vectors stable under resampling, or driven by a few lucky contrastive pairs?

### Exactly what happens, step by step

Same procedure as IV. For each of the 80 persona x trait combos, resample the ~50 CAA contrastive activation pairs with replacement, recompute the contrastive vector 50 times, and measure pairwise cosine similarity between the bootstrap vectors.

Note: CAA has fewer pairs per combo (~50, from ~50 contrastive scenarios) compared to IV (~525). This means each bootstrap resample draws from a smaller pool, so stability could in principle be lower.

### Results

- **Pairwise stability: 0.989 +/- 0.006** -- essentially identical to IV (0.990). Despite having ~10x fewer pairs per combo, the CAA vectors are just as stable.
- **Full-data alignment: 0.994 +/- 0.003** -- again, matches IV.

### The graphs

**Graph 1: Bootstrap Stability Heatmap** (`bootstrap_stability_heatmap.png`)

- **Rows**: 10 personas
- **Columns**: 8 traits
- **Cell value**: mean pairwise cosine across 50 bootstrap resamples
- **Color scale**: RdYlGn, range 0.8 to 1.0

**Graph 2: Bootstrap by Trait** (`bootstrap_by_trait.png`)

- **x-axis**: 8 traits
- **y-axis**: mean pairwise cosine
- **Plot type**: boxplot, one box per trait

### What this means for the paper

CAA vectors are just as stable as IV vectors despite having far fewer contrastive pairs. This is consistent with Nathaniel's comment that CAA requires ~O(100) samples -- even with ~50, the vectors are reliable. The robustness findings from IV carry over to CAA.

---

## R2: Convergence

### What problem does it solve?

Same as IV: how many contrastive pairs are needed for a stable CAA vector?

### Exactly what happens, step by step

Same procedure as IV. Compute vectors from subsets of N pairs (N = 1, 2, 5, 10, 20, 50, ~499), measure cosine to the reference vector (computed from all ~50 pairs). Note: the maximum N is ~50 for CAA (vs ~525 for IV), so the convergence curve covers a smaller range.

### Results

**How many pairs are needed? (mean across all 80 persona x trait combos):**

| N pairs used | Similarity to reference vector | Std |
|---|---|---|
| 1 | 0.438 | 0.220 |
| 2 | 0.517 | 0.275 |
| 5 | 0.695 | 0.182 |
| 10 | 0.831 | 0.104 |
| 20 | 0.904 | 0.055 |
| 50 | ~0.99 | ~0.01 |

**Per-trait at N=20 (fastest to slowest converging):**

| Trait | Similarity at N=20 |
|---|---|
| impulsivity | 0.941 |
| empathy | 0.940 |
| risk_taking | 0.919 |
| assertiveness | 0.919 |
| honesty | 0.910 |
| confidence | 0.895 |
| deference | 0.887 |
| warmth | 0.822 |

**Transfer matrix stability:** Cluster structure stabilizes at N=50 for CAA (vs N=20 for IV). At N=20, ARI is still 0.000 -- the clustering hasn't locked in yet. By N=50, ARI=1.0.

### The graphs

**Graph 1: Convergence Curves** (`convergence_curves.png`)

- **x-axis**: N (number of contrastive pairs), log scale
- **y-axis**: cosine similarity to reference vector (0 to 1)
- **Lines**: one colored line per trait, bold black mean line

**Graph 2: Transfer Stability** (`transfer_stability.png`)

- **Left panel**: ARI vs N -- jumps to 1.0 at N=50 (later than IV's N=20)
- **Right panel**: Frobenius distance vs N

### What this means for the paper

CAA converges at roughly the same rate as IV for individual vectors (N=20 gets you to 0.90). But the higher-level structure (which personas cluster together) takes longer to stabilize -- N=50 vs N=20. This makes sense: CAA vectors have more context-specific noise (lower shared variance), so the relative structure between personas takes more data to pin down. This is consistent with the paper's claim that CAA captures more context-entangled information.

---

## R3: Syntactic Invariance

### Not applicable to CAA

R3 tests sensitivity to instruction phrasing by splitting activations across the 5 instruction variants. CAA does not use instruction variants -- it uses pre-written contrastive scenarios. There are no variants to split by, so R3 produces no results for CAA.

This is itself an informative difference between the methods: IV's context dependence could in principle be confounded with instruction-phrasing sensitivity (R3 shows it largely isn't, but the concern exists). CAA doesn't have this confound at all -- any context dependence it shows is purely from the persona, not from instruction wording.

---

## R4: General vs Context-Dependent

### What problem does it solve?

Same as IV: for each trait, how close is each persona's vector to the general (mean across personas) direction?

### Exactly what happens, step by step

Same procedure as IV. Load all 80 CAA vectors, compute the general vector per trait, measure cosine between each persona's vector and the general direction.

### Results

**Traits ranked by context dependence (most dependent first):**

| Trait | Mean cosine | Std | Most different | Most similar |
|---|---|---|---|---|
| deference | 0.735 | 0.170 | politician (0.389) | street_hustler |
| impulsivity | 0.740 | 0.187 | politician (0.321) | con_artist |
| risk_taking | 0.786 | 0.124 | farmer (0.548) | drill_sergeant |
| warmth | 0.839 | 0.129 | drill_sergeant (0.564) | street_hustler |
| honesty | 0.842 | 0.157 | politician (0.468) | tech_ceo |
| empathy | 0.845 | 0.119 | drill_sergeant (0.590) | street_hustler |
| assertiveness | 0.867 | 0.131 | con_artist (0.571) | drill_sergeant |
| confidence | 0.879 | 0.094 | therapist (0.706) | surgeon |

**Comparison to IV:**

| Trait | IV cosine | CAA cosine | Gap |
|---|---|---|---|
| confidence | 0.929 | 0.879 | 0.050 |
| assertiveness | 0.942 | 0.867 | 0.075 |
| empathy | 0.903 | 0.845 | 0.058 |
| honesty | 0.962 | 0.842 | 0.120 |
| warmth | 0.911 | 0.839 | 0.072 |
| risk_taking | 0.858 | 0.786 | 0.072 |
| impulsivity | 0.876 | 0.740 | 0.136 |
| deference | 0.916 | 0.735 | 0.181 |

CAA shows more context dependence for every trait, with gaps ranging from 0.050 (confidence) to 0.181 (deference). The outlier personas are also more extreme under CAA -- politician's deference drops to 0.389 (vs 0.842 under IV), and politician's impulsivity to 0.321.

### The graphs

**Graph 1: General vs Contextual Heatmap** (`general_vs_contextual_heatmap.png`)

- **Rows**: 10 personas
- **Columns**: 8 traits
- **Cell value**: cosine to general vector
- **Color scale**: RdYlGn, 0.5 to 1.0

What to look for: more yellow/red cells than the IV version. The overall pattern should be similar (same traits are most/least context-dependent) but shifted toward lower values.

**Graph 2: Trait Context Dependence** (`trait_context_dependence.png`)

- **y-axis**: 8 traits ranked by mean cosine
- **x-axis**: mean cosine to general
- **Color**: red/yellow/green by value

What to look for: all bars shifted left compared to IV. The ordering may differ slightly -- notably honesty drops from most-universal (IV: 0.962) to mid-pack (CAA: 0.842), suggesting that honesty's apparent universality under IV is partly an artifact of IV's tight controls.

### What this means for the paper

This is the key IV-vs-CAA comparison. CAA consistently shows more context dependence, confirming the paper's prediction that CAA captures more context-entangled structure. The gap is not uniform -- deference (0.181) and impulsivity (0.136) show the largest IV-CAA gaps, while confidence (0.050) shows the smallest. This suggests deference and impulsivity are traits where the context interaction is large but mostly hidden from IV's controlled extraction.

The most striking finding: **honesty drops from 0.962 (most universal under IV) to 0.842 (mid-pack under CAA)**. Under IV, honesty looks nearly context-independent. Under CAA, it's as context-dependent as warmth. This suggests honesty has a strong context-independent core (which IV captures) surrounded by a substantial context-modulated component (which CAA reveals). The paper's framing of "layered structure" is directly supported.

---

## R5: Context Similarity

### What problem does it solve?

Same as IV: do semantically related personas produce similar CAA vectors?

### Exactly what happens, step by step

Same procedure as IV. Build 10x10 cosine similarity matrices per trait, compute mean across traits, run permutation test with the 5 labeled persona pairs.

### Results

**Per-trait mean pairwise similarity (off-diagonal):**

| Trait | Mean | Std | Min | Max |
|---|---|---|---|---|
| confidence | 0.751 | 0.184 | 0.136 | 0.960 |
| assertiveness | 0.728 | 0.201 | 0.201 | 0.948 |
| empathy | 0.686 | 0.225 | 0.061 | 0.943 |
| honesty | 0.681 | 0.255 | -0.021 | 0.939 |
| warmth | 0.674 | 0.237 | -0.040 | 0.932 |
| risk_taking | 0.578 | 0.321 | -0.232 | 0.962 |
| impulsivity | 0.503 | 0.399 | -0.346 | 0.959 |
| deference | 0.502 | 0.375 | -0.281 | 0.958 |

**Semantic coherence:**

| Labeled pair | Mean similarity |
|---|---|
| therapist <-> kindergarten_teacher | 0.874 |
| con_artist <-> street_hustler | 0.872 |
| drill_sergeant <-> surgeon | 0.760 |
| professor <-> tech_ceo | 0.600 |
| politician <-> con_artist | 0.559 |

- **Labeled pairs mean: 0.733**
- **Random pairs mean: 0.638**
- **p-value: 0.065** -- approaching significance (vs IV's 0.091)

### The graphs

**Graph 1: Mean Similarity Heatmap** (`similarity_heatmap_mean.png`)

- **Rows and columns**: 10 personas
- **Cell value**: mean cosine similarity across all 8 traits
- **Color scale**: RdYlGn

What to look for: more variation than IV. Some cells may be yellow or even red, especially for deference and impulsivity where pairwise similarities go negative.

**Graph 2: Semantic Coherence** (`semantic_coherence.png`)

- **Histogram**: distribution of random 5-pair means
- **Vertical line**: labeled pairs mean (0.733)
- **p-value** shown

What to look for: the vertical line should be further into the right tail than IV's version, since the gap between labeled (0.733) and random (0.638) is larger.

**Graph 3: Persona Dendrogram** (`persona_dendrogram.png`)

- Same format as IV but may show different groupings due to CAA's different similarity structure

### What this means for the paper

CAA's semantic coherence test is closer to significance (p=0.065 vs IV's 0.091) despite having lower overall similarities. This is because CAA has a larger gap between labeled (0.733) and random (0.638) pairs -- the signal is bigger even though everything is noisier. The two strongest pairs (therapist-teacher: 0.874, con_artist-hustler: 0.872) are the same under both methods, confirming these are genuine representational clusters.

The negative minimum similarities (deference: -0.28, impulsivity: -0.35) are notable -- some persona pairs have CAA vectors pointing in *opposite* directions for these traits. This never happens under IV (where the minimum is 0.39). It suggests that for some persona pairs, the CAA extraction captures fundamentally opposed manifestations of the trait.
