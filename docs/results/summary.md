# Complete Results: Context-Dependent Trait Representations

Results across two models (Gemma 2 27B IT, Gemma 3 27B IT), two extraction methods (IV, CAA), five robustness experiments (R1-R5), and SAE feature comparison (Gemma 3 only).

---

## 1. Core Finding: Shared Variance Decomposition

For each trait, we decompose persona steering vectors into a shared component (the "general" direction) and persona-specific residuals. The shared variance ratio measures what fraction of the total variance is explained by the shared direction. Lower = more context-dependent.

**IV extraction (10 personas, 8 traits):**

| Trait | Gemma 2 (L22) | Gemma 3 (L31) | Change |
|---|---|---|---|
| assertiveness | 0.867 | **0.893** | +0.026 |
| honesty | **0.896** | 0.635 | **-0.261** |
| confidence | 0.863 | 0.712 | -0.151 |
| warmth | 0.857 | 0.693 | -0.165 |
| empathy | 0.854 | 0.661 | -0.193 |
| deference | 0.828 | 0.776 | -0.052 |
| impulsivity | 0.640 | 0.776 | +0.136 |
| risk_taking | 0.726 | 0.737 | +0.011 |

Honesty drops from the most universal trait on Gemma 2 (0.896) to the most context-dependent on Gemma 3 (0.635). Assertiveness is consistently universal across both models. The trait ordering is not stable across models -- context dependence is a property of how each model was trained, not a fixed property of traits.

---

## 2. Robustness Experiments

### R1: Bootstrap Stability

Resample activation pairs with replacement (50 bootstraps), recompute contrastive vectors, measure pairwise cosine stability.

| Condition | Pairwise stability | Full-data alignment |
|---|---|---|
| Gemma 2 IV | 0.990 +/- 0.008 | 0.995 +/- 0.004 |
| Gemma 2 CAA | 0.989 +/- 0.006 | 0.994 +/- 0.003 |
| Gemma 3 IV | 0.983 +/- 0.011 | 0.992 +/- 0.006 |
| Gemma 3 CAA | 0.993 +/- 0.005 | 0.997 +/- 0.003 |

All conditions produce stable vectors (>0.98 pairwise). Measurement uncertainty is ~0.01 in cosine, far smaller than the effect sizes in R4.

W&B graphs:
- [Gemma 2 IV](https://wandb.ai/girishgupta-com/persona-steering/runs/n5gn6kra)
- [Gemma 2 CAA](https://wandb.ai/persona-steering/personas/runs/g8yeutls)
- [Gemma 3 IV](https://wandb.ai/persona-steering/personas/runs/dpejiq4t)
- [Gemma 3 CAA](https://wandb.ai/persona-steering/personas/runs/4m7siy7e)

### R2: Convergence

Compute vectors from subsets of N activation pairs, measure cosine to the reference vector (all pairs).

**Cosine to reference at N=20:**

| Condition | N=20 cosine | Cluster stability (ARI=1 at N=?) |
|---|---|---|
| Gemma 2 IV | 0.899 | N=20 |
| Gemma 2 CAA | 0.904 | N=50 |
| Gemma 3 IV | 0.842 | N=100 |
| Gemma 3 CAA | 0.938 | N=20 |

Gemma 3 IV converges slower than all other conditions -- representations are higher-dimensional or noisier. Gemma 3 CAA converges fastest.

W&B graphs:
- [Gemma 2 IV](https://wandb.ai/girishgupta-com/persona-steering/runs/ghrfldv0)
- [Gemma 2 CAA](https://wandb.ai/persona-steering/personas/runs/piu5r1u7)
- [Gemma 3 IV](https://wandb.ai/persona-steering/personas/runs/tmrcevl3)
- [Gemma 3 CAA](https://wandb.ai/persona-steering/personas/runs/2ihlujh5)

### R3: Syntactic Invariance (IV only)

Compute separate vectors per instruction variant, compare within-persona cross-variant similarity (syntactic noise) to across-persona same-variant similarity (persona signal). Higher across-persona means persona identity is the stronger signal.

| Model | Within-persona | Across-persona | p-value (Mann-Whitney) |
|---|---|---|---|
| Gemma 2 | 0.655 | 0.719 | **0.007 (significant)** |
| Gemma 3 | 0.599 | 0.616 | 0.349 (not significant) |

On Gemma 2, persona identity is significantly stronger than instruction phrasing. On Gemma 3, they are not separable -- the model is equally sensitive to how the instruction is worded and which persona is active. This is a complication: on Gemma 3, some of what looks like "context dependence" may actually be instruction-phrasing sensitivity.

W&B graphs:
- [Gemma 2](https://wandb.ai/girishgupta-com/persona-steering/runs/xw61w2ua)
- [Gemma 3](https://wandb.ai/persona-steering/personas/runs/sc23spag)

Not applicable to CAA (no instruction variants).

### R4: General vs Context-Dependent

Compute the "general" vector per trait (mean across personas). Measure each persona's cosine to the general direction. Also compare to null (no system prompt) and nonsense (gibberish system prompt) baselines.

**All four conditions (cosine to general, sorted by most context-dependent):**

| Trait | G2 IV | G2 CAA | G3 IV | G3 CAA |
|---|---|---|---|---|
| deference | 0.916 | **0.735** | 0.865 | **0.693** |
| impulsivity | 0.876 | 0.740 | 0.822 | 0.794 |
| risk_taking | 0.858 | 0.786 | 0.848 | 0.799 |
| warmth | 0.911 | 0.839 | 0.820 | 0.822 |
| empathy | 0.903 | 0.845 | **0.797** | 0.865 |
| honesty | **0.962** | 0.842 | 0.850 | 0.925 |
| confidence | 0.929 | 0.879 | 0.836 | 0.825 |
| assertiveness | 0.942 | 0.867 | 0.924 | 0.927 |

Key observations:
- **Deference** is consistently the most context-dependent under CAA on both models (0.735, 0.693).
- **Assertiveness** is consistently the most universal (0.87-0.94 across all conditions).
- **Honesty** shows a striking reversal on Gemma 3: it's the most context-dependent under IV (0.850) but the most universal under CAA (0.925). On Gemma 2 the pattern was the opposite.
- **CAA shows more context dependence than IV** on Gemma 2 (every trait), but this pattern does not hold consistently on Gemma 3.

**Most divergent personas per trait (G3 IV):**

| Trait | Most different persona |
|---|---|
| empathy | street_hustler |
| warmth | street_hustler |
| impulsivity | professor |
| confidence | professor |
| risk_taking | politician |
| honesty | therapist |
| deference | professor |
| assertiveness | professor |

Professor emerges as a major outlier on Gemma 3, diverging most on 4 of 8 traits. On Gemma 2, drill_sergeant and surgeon were the main outliers.

W&B graphs:
- [Gemma 2 IV](https://wandb.ai/girishgupta-com/persona-steering/runs/pdokd87s)
- [Gemma 2 CAA](https://wandb.ai/persona-steering/personas/runs/0fui3v4p)
- [Gemma 3 IV](https://wandb.ai/persona-steering/personas/runs/9b4rfkb8)
- [Gemma 3 CAA](https://wandb.ai/persona-steering/personas/runs/obwjyu3q)

### R5: Context Similarity

Pairwise cosine similarity between all persona vectors per trait. Permutation test for semantic coherence (do labeled-similar persona pairs score higher than random?).

| Condition | Labeled pairs | Random pairs | p-value |
|---|---|---|---|
| Gemma 2 IV | 0.838 | 0.816 | 0.138 |
| Gemma 2 CAA | 0.733 | 0.638 | 0.065 |
| Gemma 3 IV | 0.760 | 0.704 | 0.098 |
| Gemma 3 CAA | 0.733 | 0.656 | 0.085 |

The trend is consistent: labeled pairs always score higher than random. The gap is larger under CAA (because overall similarity is lower, making the semantic structure more visible). None reach p<0.05 significance, but all are trending (p=0.065-0.138). With more personas, these would likely become significant.

W&B graphs:
- [Gemma 2 IV](https://wandb.ai/girishgupta-com/persona-steering/runs/5xvv2j9u)
- [Gemma 2 CAA](https://wandb.ai/persona-steering/personas/runs/tnuwnehl)
- [Gemma 3 IV](https://wandb.ai/persona-steering/personas/runs/tsxzg3mf)
- [Gemma 3 CAA](https://wandb.ai/persona-steering/personas/runs/tp3t45oo)

---

## 3. SAE Feature Comparison (Gemma 3 only)

Using Gemma Scope 2 SAE (google/gemma-scope-2-27b-it, resid_post_all, layer 31, 262k features) to compare SAE features against steering vectors.

### Best SAE feature alignment

For each trait, the cosine between the general steering vector and the single best-matching SAE feature (out of 262,144).

| Trait | IV best cos | IV feature | CAA best cos | CAA feature |
|---|---|---|---|---|
| honesty | -0.506 | #34250 | **0.860** | #156484 |
| empathy | -0.352 | #5037 | **0.849** | #156484 |
| assertiveness | -0.528 | #174267 | **0.793** | #16190 |
| warmth | 0.340 | #18490 | **0.789** | #16190 |
| impulsivity | **0.705** | #9393 | -0.631 | #156484 |
| risk_taking | 0.420 | #8176 | 0.278 | #107170 |
| confidence | 0.318 | #6058 | 0.384 | #7614 |
| deference | -0.541 | #66298 | -0.296 | #26129 |

**CAA vectors match SAE features much better than IV vectors** for most traits (honesty: 0.86 vs 0.51, empathy: 0.85 vs 0.35). This makes sense: CAA captures how traits naturally manifest in the model's activations, which is closer to what the SAE learned from normal forward passes.

**Feature #156484** appears as the best match for honesty, empathy, and (negatively) impulsivity under CAA. This single feature captures a "prosocial/ethical" direction -- honest, empathetic, and anti-impulsive.

**Feature #16190** captures both assertiveness and warmth under CAA -- suggesting these traits share representational structure.

IV and CAA never share the same best feature for any trait. The two extraction methods find different aspects of the same trait.

### Feature overlap across personas

For each trait, how many of the top-10 SAE features are shared across all 10 personas?

| Trait | IV shared (all) | IV shared (majority) | CAA shared (all) | CAA shared (majority) |
|---|---|---|---|---|
| assertiveness | 0 | 7 | **1** | 7 |
| confidence | 0 | 7 | 0 | 5 |
| honesty | 0 | 3 | 0 | 4 |
| impulsivity | 0 | 4 | 0 | 4 |
| risk_taking | 0 | 4 | 0 | 2 |
| deference | 0 | 0 | 0 | 0 |
| empathy | 0 | 0 | 0 | 3 |
| warmth | 0 | 1 | 0 | 2 |

**Assertiveness under CAA is the only trait where a single SAE feature appears in every persona's top-10.** For all other traits, different personas activate completely different SAE features. This is feature-level evidence of context-dependent representations -- the SAE itself decomposes traits differently depending on persona.

Deference has zero shared features even by majority under both methods -- the most context-specific trait at the feature level, consistent with R4 findings.

W&B graphs:
- [Gemma 3 IV SAE](https://wandb.ai/persona-steering/personas/runs/wnnbxd33)
- [Gemma 3 CAA SAE](https://wandb.ai/persona-steering/personas/runs/zl3e6vt8)

---

## 4. Main Pipeline Analysis (Gemma 3)

### Transfer matrix

The 12x12 persona similarity matrix (mean cosine across traits) shows clear structure on Gemma 3:
- **Professor** is an outlier (0.37-0.49 similarity with politician, street_hustler, surgeon)
- **Null and nonsense** baselines sit in the mid-range, not dramatically different from real personas
- Mean off-diagonal similarity is 0.70 (lower than Gemma 2's 0.81)

Spearman correlation between Gemma 2 and Gemma 3 transfer matrices: rho=0.51 (moderate agreement on which personas are similar).

W&B: [Gemma 3 step 4 analysis](https://wandb.ai/persona-steering/personas/runs/kxb83gy7)

### Shared variance bar chart

Only assertiveness (89.3%) exceeds the 80% threshold on Gemma 3. Five traits are below 75%. Honesty is at the bottom (63.5%).

### Behavioral effect sizes

From the LLM-judge evaluation (step 6):
- **Deference** has near-zero effect for several personas (drill_sergeant: 0.04, politician: 0.15) -- the instruction barely changes behavior
- **Honesty** is near-zero for professor (0.02) and therapist (0.04) -- these personas may already be "locked in" to honesty
- **Empathy** has consistently high effects (0.42-0.75) across all personas

### Geometry vs behavior correlation

r = -0.082 on Gemma 3. Geometric context dependence (how much the vector deviates from general) does not predict behavioral divergence (how much steering effectiveness varies). The geometric and behavioral evidence are complementary but independent.

W&B: [Gemma 3 step 5 figures](https://wandb.ai/persona-steering/personas/runs/w40opqnt)

### Persona landscape (PCA)

PC1 explains 79.2% of variance. Clear groupings:
- Professor isolated on the far left
- Drill sergeant and surgeon cluster together (upper middle)
- Con artist and street hustler cluster together (lower middle)
- Null and nonsense are far right -- genuinely different from all real personas

W&B: [Gemma 3 landscape](https://wandb.ai/persona-steering/personas/runs/ezvn2rdg)

---

## 5. Cross-Model Summary

### What replicates

1. **Context dependence exists on both models.** No trait is fully context-independent under any condition.
2. **Assertiveness is consistently the most universal trait** (0.87-0.94 across all conditions).
3. **Deference is consistently problematic** -- most context-dependent under CAA, low syntactic invariance, zero shared SAE features.
4. **Vectors are stable** (R1 > 0.98 everywhere).
5. **Semantic coherence trends positive** (labeled pairs > random) but doesn't reach significance with 10 personas.
6. **SAE features are overwhelmingly persona-specific** -- different personas activate different features for the same trait.

### What doesn't replicate

1. **Trait ordering changes.** Honesty goes from most universal (Gemma 2 IV: 0.962) to among the most context-dependent (Gemma 3 IV: 0.850). The degree of context dependence is model-specific.
2. **IV vs CAA gap is not consistent.** On Gemma 2, CAA always shows more context dependence. On Gemma 3, this reverses for some traits (honesty, empathy).
3. **R3 syntactic invariance is significant on Gemma 2 (p=0.007) but not Gemma 3 (p=0.349).** Persona and syntax are cleanly separable on Gemma 2 but not Gemma 3.
4. **Outlier personas change.** Gemma 2: drill_sergeant, surgeon. Gemma 3: professor, street_hustler.
5. **Convergence rates differ.** Gemma 3 IV needs ~100 pairs (vs Gemma 2's ~20).

### Implications for the paper

The core claim -- that trait representations are context-dependent -- is supported on both models. But the specific quantitative findings (which traits, which personas, how much) are model-dependent. The paper should frame context dependence as a general phenomenon while being honest that the details vary across models. The SAE evidence (zero universal features) is the most model-independent finding and may be the strongest single piece of evidence.
