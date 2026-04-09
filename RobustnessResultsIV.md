# Robustness Results (IV, 10 personas x 8 traits, Gemma-2-27B-IT, layer 22)

---

## R1: Bootstrap Stability

### What problem does it solve?

Each steering vector is built from ~525 (5 instruction variants x ~105 questions) activation pairs. But what if 10 of those ~525 pairs are weird outliers, and they're pulling the vector in a direction it wouldn't otherwise point? If that's the case, our downstream results (R2-R5, and the paper's core claims) would be unreliable -- they'd change if we happened to sample different questions.

R1 answers: **if we rebuilt each vector from a different random sample of the same data, would we get the same direction?**

### Exactly what happens, step by step

1. **Load activations.** For each of the 80 persona x trait combos (e.g., farmer x honesty), load the positive and negative activation files. Each file is a dict mapping keys like `v2_q7` to a tensor of shape `(n_layers, hidden_dim)` -- one tensor per response. There are ~525 (5 instruction variants x ~105 questions) positive and ~525 (5 instruction variants x ~105 questions) negative tensors per combo.

2. **Bootstrap resample.** 50 times, do the following:
   - Draw ~525 indices *with replacement* from the ~525 positive activations (so some appear twice, some are left out)
   - Draw ~525 indices *with replacement* from the ~525 negative activations
   - For each selected activation, take the layer-22 slice and clean NaN/Inf values
   - Compute: `bootstrap_vector = mean(selected_positive) - mean(selected_negative)`
   - Extract layer 22 from the result -- one 4608-dimensional vector

   After 50 iterations, you have 50 bootstrap vectors for this persona x trait.

3. **Pairwise cosine.** Compute cosine similarity between every pair of the 50 bootstrap vectors. That's C(50,2) = 1225 cosine values. Take the mean -- this is the **pairwise stability** for this persona x trait. If it's 0.99, the 50 vectors are all pointing in nearly the same direction regardless of which samples were drawn.

4. **Full-data alignment.** Also load the "real" vector (computed from all ~525 pairs, stored in `vectors/farmer_honesty.pt`). Compute cosine between each of the 50 bootstrap vectors and this real vector. Take the mean -- this is **full-data alignment**. It tells you: how close is any given bootstrap resample to the vector you'd get with all the data?

5. **Aggregate.** Collect pairwise stability and full-data alignment across all 80 persona x trait combos. Report the overall mean +/- std.

### Results

- **Pairwise stability: 0.990 +/- 0.008** -- across all 80 combos, the mean pairwise cosine between bootstrap vectors is 0.990. The worst combo is still above ~0.97.
- **Full-data alignment: 0.995 +/- 0.004** -- bootstrap vectors are even closer to the full-data vector than to each other (because the full-data vector is the best estimate, and bootstraps scatter around it).

### The graphs

**Graph 1: Bootstrap Stability Heatmap** (`bootstrap_stability_heatmap.png`)


- **Rows**: 10 personas (con_artist, drill_sergeant, farmer, ...)
- **Columns**: 8 traits (assertiveness, confidence, deference, ...)
- **Cell value**: mean pairwise cosine across the 50 bootstrap vectors for that persona x trait
- **Color scale**: RdYlGn (red = low stability, green = high), range 0.8 to 1.0
- **Each cell has the number printed on it** (e.g., 0.993)

What to look for: any red or yellow cells would indicate an unstable vector -- a persona x trait where the direction is sensitive to which samples you draw. If the whole heatmap is green, everything is stable.

**Graph 2: Bootstrap by Trait** (`bootstrap_by_trait.png`)


- **x-axis**: 8 traits
- **y-axis**: mean pairwise cosine (range 0.7 to 1.0)
- **Plot type**: boxplot -- one box per trait, showing the distribution of pairwise stability across the 10 personas for that trait

What to look for: are some traits systematically less stable than others? If deference's box sits lower than honesty's, it means deference vectors are noisier -- different data samples give more different directions.

### What this means for the paper

You can trust the vectors. When R4 says "surgeon's risk-taking vector has cosine 0.713 to the general direction," that 0.713 is a real measurement, not noise. The bootstrap stability of 0.99 means the measurement uncertainty is about +/-0.01 in cosine -- far smaller than the effect sizes reported in R4 (which are 0.05-0.25).

---

## R2: Convergence

### What problem does it solve?

R1 showed the vectors are stable with all ~525 (5 instruction variants x ~105 questions) pairs. But how many pairs do you actually *need*? If vectors converge at N=20, collecting ~525 is overkill. If they haven't converged at N=100, you need more data. This also tells you: if you wanted to extract vectors for a new context, what's the minimum data budget?

### Exactly what happens, step by step

1. **Load activations and the reference vector.** Same as R1 -- load all ~525 (5 instruction variants x ~105 questions) positive and ~525 (5 instruction variants x ~105 questions) negative activation tensors per persona x trait. Also load the "reference" vector -- the one computed from all ~525 pairs. This is the best estimate of the true direction.

2. **Subset sampling.** For each subset size N in [1, 2, 5, 10, 20, 50, 100, ~525]:
   - Randomly sample N activations *without replacement* from the positive set, and N from the negative set
   - Compute the contrastive vector: `mean(sampled_positive) - mean(sampled_negative)` at layer 22
   - Measure cosine similarity between this N-pair vector and the reference vector. This answers: "how close is my cheap vector (built from N pairs) to the best vector (built from all ~525 pairs)?" A value of 0.90 means the N-pair vector captures 90% of the direction.

3. **Per-trait aggregation.** Average convergence curves across the 10 personas for each trait, revealing which traits converge faster/slower.

### Results

**How many pairs are needed? (mean across all 80 persona x trait combos):**

| N pairs used | Similarity to reference vector | Std |
|---|---|---|
| 1 | 0.428 | 0.244 |
| 2 | 0.533 | 0.274 |
| 5 | 0.689 | 0.173 |
| 10 | 0.823 | 0.107 |
| **20** | **0.899** | **0.064** |
| 50 | 0.952 | 0.030 |
| 100 | 0.980 | 0.013 |
| 520 | 1.000 | 0.000 |

**Per-trait at N=20 (fastest to slowest converging):**

| Trait | Similarity at N=20 |
|---|---|
| empathy | 0.943 |
| assertiveness | 0.940 |
| warmth | 0.938 |
| honesty | 0.930 |
| confidence | 0.904 |
| impulsivity | 0.872 |
| risk_taking | 0.834 |
| deference | 0.832 |

### The graphs

**Graph 1: Convergence Curves** (`convergence_curves.png`)

- **x-axis**: N (number of activation pairs used), log scale
- **y-axis**: cosine similarity to reference vector (0 to 1)
- **Lines**: one colored line per trait (mean +/- std error bars across 10 personas), plus a bold black line for the mean across all traits

What to look for: how quickly each line rises toward 1.0. Traits that rise fast (empathy, warmth) have "cleaner" representations -- fewer samples needed to pin down the direction. Traits that rise slowly (risk_taking, deference) are noisier or higher-dimensional.

**Graph 2: Transfer Stability** (`transfer_stability.png`)

This graph shows a secondary analysis: at each N, the script also rebuilds the full 10x10 persona similarity matrix (which personas have similar vectors?) using the N-pair vectors, and checks whether the overall structure matches the reference. The left panel shows when the persona groupings snap into place (at N=20), the right panel shows the matrix error shrinking.

### What this means for the paper

20 activation pairs per persona x trait is the practical minimum for reliable vectors. The ~525 pairs used in the actual pipeline are massive overkill -- the vectors converged long before that. Traits that converge slowly (risk_taking, deference) are the same ones that show the most context dependence in R4, suggesting their representations are genuinely more complex, not just noisier.

---

## R3: Syntactic Invariance

### What problem does it solve?

The extraction method uses 5 different instruction phrasings per trait (e.g., "be very assertive" vs "speak with authority and confidence"). These are meant to be semantically equivalent. But what if the model's representation is sensitive to the *wording* rather than the *meaning*? If so, what we're calling "context dependence" in R4 might actually be "instruction-phrasing dependence."

R3 disentangles the two by asking: **does persona identity or instruction phrasing explain more of the variation?**

### Exactly what happens, step by step

1. **Split activations by variant.** The activation keys follow the format `v{i}_q{j}` (variant i, question j). For each persona x trait, split the ~525 (5 instruction variants x ~105 questions) activations into 5 groups by variant index, each containing ~105 activations.

2. **Compute per-variant vectors.** For each variant independently, compute the contrastive vector: `mean(variant_positive) - mean(variant_negative)` at layer 22. This gives 5 vectors per persona x trait, one per instruction phrasing.

3. **Within-persona, across-variant similarity.** For each persona x trait, compute pairwise cosine between all C(5,2) = 10 pairs of per-variant vectors. Take the mean. This measures: *holding the persona and trait fixed, how much does the vector change when you rephrase the instruction?*

4. **Across-persona, within-variant similarity.** For each trait and variant index, compute pairwise cosine between the 10 personas' vectors (all computed from the same variant). Take the mean. This measures: *holding the instruction phrasing fixed, how much does the vector change across personas?*

5. **Compare.** If across-persona > within-persona, persona identity is the stronger signal. If within-persona > across-persona, instruction phrasing matters more.

### Results

- **Within-persona across-variant: 0.659 +/- 0.139** -- moderate. Same persona + same trait, different instruction wording -> cosine ~0.66.
- **Across-persona within-variant: 0.726 +/- 0.088** -- higher. Different personas, same instruction wording -> cosine ~0.73.

**Per-trait within-persona cross-variant similarity (worst to best):**

| Trait | Mean | Std |
|---|---|---|
| deference | 0.406 | 0.070 |
| risk_taking | 0.589 | 0.115 |
| impulsivity | 0.630 | 0.088 |
| confidence | 0.654 | 0.071 |
| assertiveness | 0.728 | 0.050 |
| empathy | 0.749 | 0.078 |
| honesty | 0.751 | 0.050 |
| warmth | 0.765 | 0.102 |

### The graphs

**Graph 1: Syntactic Invariance by Trait** (`syntactic_by_trait.png`)


- **y-axis**: 8 traits, ordered by mean cross-variant similarity
- **x-axis**: cosine similarity (0 to 1), with error bars showing std across 10 personas
- **Plot type**: horizontal bar chart

What to look for: deference is far worse than everything else (0.41) -- different phrasings of "be deferential" produce nearly orthogonal vectors. This means deference may not have a coherent single representation in this model. Warmth, honesty, and empathy are the most robust (~0.75).

**Graph 2: Invariance Comparison** (`invariance_comparison.png`)


- **Two boxes**: "Within-persona (across variants)" and "Across-persona (same variant)"
- **y-axis**: cosine similarity
- **Plot type**: side-by-side boxplots showing the full distribution

What to look for: the right box (across-persona) should sit higher than the left box (within-persona). This confirms that persona identity explains more variance than instruction phrasing. The gap between the two boxes is the evidence that "context dependence" is real and not just "instruction noise."

### What this means for the paper

The context dependence measured in R4 is not an artifact of instruction phrasing -- persona identity is the stronger signal (0.73 > 0.66). However, there IS real syntactic sensitivity, especially for deference (0.41). The 5-variant averaging in the main pipeline is doing important denoising work: each individual variant is noisy, but averaging across 5 washes out the syntactic noise and leaves the persona signal. Deference's terrible invariance (0.41) is itself a finding -- it may not have a coherent single representation in this model, supporting the team's discussion about dropping it.

---

## R4: General vs Context-Dependent

### What problem does it solve?

This is the core experiment for the paper's central claim. For each trait, there's a "general" direction (the average across all personas). How close is each persona's individual vector to this general direction? If they're all close (cosine ~1.0), the trait is universal and a single vector suffices. If they diverge, the trait is context-dependent.

### Exactly what happens, step by step

1. **Load all 80 vectors.** One per persona x trait at layer 22 (4608-dimensional).

2. **Compute general vectors.** For each trait, average the 10 persona vectors: `general_honesty = mean(farmer_honesty, politician_honesty, ..., con_artist_honesty)`. This is the "context-free" representation of the trait.

3. **Measure deviation.** For each persona x trait, compute:
   - **Cosine to general**: how aligned is this persona's vector with the general direction? 1.0 = identical, 0.0 = orthogonal.
   - **Specificity ratio**: project the persona vector onto the general direction, compute the residual (the orthogonal component). The ratio of residual magnitude to total magnitude tells you what fraction of the vector is persona-specific.

4. **Per-trait summary.** For each trait, report mean cosine-to-general across the 10 personas, plus which persona is most different (lowest cosine) and most similar (highest cosine).

5. **Per-persona summary.** For each persona, report mean cosine-to-general across the 8 traits, plus which trait they diverge most on.

6. **Cluster bias.** Build the 10x10 transfer matrix (pairwise persona similarity averaged across traits). Run agglomerative clustering. Check if the general vector is equidistant from all clusters or biased toward one.

### Results

**Traits ranked by context dependence (most dependent first):**

| Trait | Mean cosine | Std | Most different | Most similar |
|---|---|---|---|---|
| risk_taking | 0.858 | 0.064 | surgeon (0.713) | street_hustler (0.953) |
| impulsivity | 0.876 | 0.055 | therapist (0.770) | street_hustler (0.950) |
| empathy | 0.903 | 0.051 | drill_sergeant (0.816) | tech_ceo (0.967) |
| warmth | 0.911 | 0.061 | drill_sergeant (0.754) | con_artist (0.961) |
| deference | 0.916 | 0.032 | con_artist (0.842) | farmer (0.950) |
| confidence | 0.929 | 0.031 | politician (0.871) | con_artist (0.964) |
| assertiveness | 0.942 | 0.017 | therapist (0.912) | surgeon (0.971) |
| honesty | 0.962 | 0.012 | therapist (0.938) | professor (0.976) |

**Clustering:** Only 1 cluster found per trait with 10 personas.

### The graphs

**Graph 1: General vs Contextual Heatmap** (`general_vs_contextual_heatmap.png`)


- **Rows**: 10 personas
- **Columns**: 8 traits
- **Cell value**: cosine similarity between that persona's vector and the general vector for that trait
- **Color scale**: RdYlGn, range 0.5 to 1.0
- **Each cell has the number printed on it**

What to look for: green cells (high cosine, close to general) vs yellow/red cells (low cosine, divergent from general). The pattern should show: honesty column is all green (universal), risk_taking column has more yellow (context-dependent). Individual red/yellow cells identify the specific persona x trait outliers (e.g., surgeon x risk_taking, drill_sergeant x warmth).

**Graph 2: Trait Context Dependence** (`trait_context_dependence.png`)


- **y-axis**: 8 traits, ordered by mean cosine to general (most context-dependent at top)
- **x-axis**: mean cosine to general vector, with error bars showing std across personas
- **Color**: red = most context-dependent, green = most universal
- **Annotations**: "most diff: [persona]" next to each bar

What to look for: the spread from risk_taking (~0.86) to honesty (~0.96) is the core finding. The annotations tell you which persona drives each trait's context-dependence.

### What this means for the paper

This is the central result. The spectrum from honesty (0.962) to risk_taking (0.858) directly supports the claim that "the mapping from traits to representations is context-dependent." The effect size is substantial: a surgeon's risk-taking vector has cosine 0.713 to the general direction, meaning it points in a meaningfully different direction. The outliers are interpretable (drill_sergeant on warmth, surgeon on risk_taking), not random. And R1 established that the measurement uncertainty is ~0.01, so these effects are well above noise.

---

## R5: Context Similarity

### What problem does it solve?

R4 tells you *how much* each persona deviates from the general direction, but not *which personas are similar to each other*. R5 asks: do semantically related personas (therapist and kindergarten teacher, both caring roles) produce similar steering vectors? If yes, the representation space has meaningful structure that tracks real-world semantic relationships. If no, the context-dependence might be unstructured noise.

### Exactly what happens, step by step

1. **Load all 80 vectors.** Same as R4.

2. **Per-trait similarity matrix.** For each trait, compute the 10x10 cosine similarity matrix between all persona pairs. Cell (i,j) = cosine(persona_i's vector, persona_j's vector) for that trait. Diagonal is always 1.0.

3. **Mean similarity matrix.** Average the 8 per-trait matrices to get an overall "how similar are these two personas across all traits" measure.

4. **Semantic coherence test.** Define 5 human-labeled pairs of personas that *should* be similar based on their real-world roles:
   - therapist <-> kindergarten_teacher (caring/nurturing)
   - con_artist <-> street_hustler (street-smart, deceptive)
   - drill_sergeant <-> surgeon (high-authority, decisive)
   - professor <-> tech_ceo (intellectual authority)
   - politician <-> con_artist (strategic/manipulative)

   Compute the mean similarity of these 5 labeled pairs from the mean similarity matrix. Then run a permutation test: 10,000 times, randomly pick 5 pairs and compute their mean similarity. The p-value is the fraction of random draws that equal or exceed the labeled mean.

5. **Hierarchical clustering.** Convert the mean similarity matrix to a distance matrix (1 - similarity). Run average-linkage agglomerative clustering. Plot a dendrogram showing which personas group together.

### Results

**Per-trait mean pairwise similarity (off-diagonal):**

| Trait | Mean | Std | Min | Max |
|---|---|---|---|---|
| honesty | 0.918 | 0.030 | 0.821 | 0.966 |
| assertiveness | 0.874 | 0.046 | 0.766 | 0.951 |
| confidence | 0.849 | 0.074 | 0.625 | 0.952 |
| deference | 0.821 | 0.066 | 0.637 | 0.931 |
| warmth | 0.812 | 0.107 | 0.532 | 0.961 |
| empathy | 0.796 | 0.098 | 0.542 | 0.944 |
| impulsivity | 0.742 | 0.102 | 0.482 | 0.905 |
| risk_taking | 0.710 | 0.111 | 0.390 | 0.882 |

**Semantic coherence:**

| Labeled pair | Mean similarity |
|---|---|
| therapist <-> kindergarten_teacher | 0.898 |
| con_artist <-> street_hustler | 0.879 |
| politician <-> con_artist | 0.817 |
| professor <-> tech_ceo | 0.816 |
| drill_sergeant <-> surgeon | 0.779 |

- **Labeled pairs mean: 0.838**
- **Random pairs mean: 0.816**
- **p-value: 0.138** -- not significant at 0.05

### The graphs

**Graph 1: Mean Similarity Heatmap** (`similarity_heatmap_mean.png`)


- **Rows and columns**: 10 personas
- **Cell value**: mean cosine similarity across all 8 traits
- **Color scale**: RdYlGn, range -0.2 to 1.0
- **Each cell has the number printed on it**

What to look for: blocks of high similarity between semantically related personas. Therapist and kindergarten_teacher should show a bright green cell. The diagonal is always 1.0.

**Graph 2: Per-Trait Heatmaps** (`similarity_heatmap_{trait}.png`)

8 separate heatmaps (one per trait), same format as above but showing only that trait's pairwise similarities. Compare across traits: honesty should be uniformly high (all personas similar), risk_taking should show more variation.

**Graph 3: Semantic Coherence** (`semantic_coherence.png`)


- **Histogram**: distribution of mean similarity for 10,000 random 5-pair draws
- **Vertical line**: the labeled pairs' mean similarity (0.838)
- **p-value** shown on the plot

What to look for: how far to the right the vertical line sits from the bulk of the histogram. If it's in the tail, labeled pairs are significantly more similar than random. Here it's in the right portion but not clearly in the tail (p=0.138).

**Graph 4: Persona Dendrogram** (`persona_dendrogram.png`)


- **Leaf nodes**: 10 personas
- **Branch height**: distance (1 - similarity) at which personas merge
- **Linkage**: average

What to look for: do semantically related personas merge early (low branch height)? E.g., therapist and kindergarten_teacher merging before either merges with drill_sergeant would indicate the representation space respects semantic structure.

### What this means for the paper

The trend goes the right way: labeled-similar pairs (0.838) score higher than random (0.816), and the two highest-scoring pairs (therapist-teacher at 0.898, con_artist-hustler at 0.879) are the ones with the strongest semantic relationship. But with only 10 personas giving 45 unique pairs and 5 labeled pairs, the permutation test doesn't reach significance (p=0.138). The range within traits is telling -- risk_taking has persona pairs as low as 0.39 (nearly orthogonal vectors) and as high as 0.88, showing that context can *dramatically* change the representation for some traits. The dendrogram provides a visual check of whether the hierarchical structure matches semantic intuition.
