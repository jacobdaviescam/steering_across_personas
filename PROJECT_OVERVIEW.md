# Persona-Conditional Steering Vectors: Project Overview

**Jacob Davies, Rhea Karty** | ERA:AI Fellowship, Winter 2026

## Research Question

Steering vectors extracted from a single baseline persona are widely used for monitoring and controlling model behaviour (Chen et al., 2025; Lu et al., 2026). But models are deployed across diverse operational modes, each a distinct persona state. **Do steering vectors transfer across personas, or do personas fundamentally reshape trait representations?**

## Method

We induce **10 persona archetypes** via system prompts (Farmer, Politician, Therapist, Drill Sergeant, Street Hustler, Professor, Tech CEO, Kindergarten Teacher, Surgeon, Con Artist) and extract steering vectors for **8 traits** (assertiveness, empathy, risk-taking, honesty, confidence, deference, warmth, impulsivity).

**Two extraction methods:**

- **Instruction-Variant (IV):** 5 pos/neg instruction pairs x 20 sampled questions = 100 contrastive pairs per persona x trait. Same question under opposing instructions isolates the trait signal from content.
- **Contrastive Activation Addition (CAA):** Standard contrastive pair method from the literature.

**Primary model:** Gemma-2-27B-IT, layer 22/46. **Training trajectory model:** OLMo-2-1124-7B across 7 checkpoints (pretrain 1%, 10%, 50%, base, SFT, DPO, Instruct).

---

## Result 1: Geometric Similarity (Cosine Similarity of Steering Vectors)

Cross-persona cosine similarity heatmaps for each trait reveal that **steering vectors differ substantially across personas** --- the same trait is encoded differently depending on identity.

![Per-trait cosine similarity heatmaps (IV method)](figures/placeholder_iv_cosine_heatmaps.png)
*Figure 1: Cross-persona cosine similarity heatmaps for each trait under the Instruction-Variant method. Red = similar encoding; Blue = distinct representations.*

![Per-trait cosine similarity heatmaps (CAA method)](figures/placeholder_caa_cosine_heatmaps.png)
*Figure 2: Cross-persona cosine similarity heatmaps for each trait under the CAA method.*

Key patterns:
- **Honesty** shows the highest cross-persona similarity, consistent with RLHF training collapsing representational diversity for heavily optimised traits.
- **Risk-taking and impulsivity** show the most persona-specific variation.
- Intuitively similar persona-trait pairs share similar vectors (e.g. kindergarten teacher and therapist on empathy/warmth), while unexpected pairs diverge (e.g. tech CEO and surgeon on risk).
- The CAA method produces **more divergent** vectors across personas than IV, with lower overall cosine similarities and more distinct per-persona structure.

## Result 2: Shared vs Persona-Specific Variance Decomposition

We decompose each trait's steering vectors into a shared component (common across all personas) and persona-specific residuals. The shared variance ratio measures what fraction of the total signal is universal.

| Trait | IV Shared Variance | CAA Shared Variance |
|-------|-------------------|---------------------|
![Shared variance bar chart comparing IV and CAA](figures/placeholder_shared_variance_bars.png)
*Figure 3: Shared vs persona-specific variance for each trait under both extraction methods. The 80% threshold line marks the boundary between high- and low-universality traits.*

| Honesty | 92.8% | 72.7% |
| Assertiveness | 88.5% | 78.0% |
| Confidence | 86.4% | 76.7% |
| Warmth | 85.1% | 67.4% |
| Deference | 83.9% | 50.2% |
| Empathy | 83.8% | 68.6% |
| Impulsivity | 75.2% | 54.2% |
| Risk-taking | 72.9% | 56.9% |

**No trait exceeds 93% shared variance under either method** --- all encode substantial persona-specific signal. The IV method consistently yields higher shared variance than CAA, suggesting IV extracts a "cleaner" trait signal that is more universal, while CAA captures more persona-entangled information.

Under IV, the 80% threshold separates traits into two groups: those heavily shaped by RLHF (honesty, assertiveness, confidence, warmth, deference, empathy --- all above 80%) and those less constrained by training (impulsivity, risk-taking --- below 80%).

Under CAA, only assertiveness and confidence exceed 75%, and deference and impulsivity fall below 55%, indicating that CAA vectors are substantially more persona-conditioned.

## Result 3: Behavioural Validation (Self vs Cross-Persona Steering)

We apply steering vectors and evaluate their behavioural effect using a Claude LLM-as-judge (0--1 trait scores). Self-steering uses a persona's own vector; cross-steering uses vectors from other personas.

| Trait | Self-Steer (mean) | Cross-Steer (mean) | Gap |
|-------|-------------------|---------------------|-----|
![Self vs cross steering bar chart](figures/placeholder_self_vs_cross.png)
*Figure 4: Mean trait scores for self-steering vs cross-steering across all 8 traits.*

| Confidence | 0.896 | 0.892 | +0.004 |
| Assertiveness | 0.861 | 0.856 | +0.005 |
| Empathy | 0.851 | 0.848 | +0.003 |
| Warmth | 0.795 | 0.787 | +0.008 |
| Honesty | 0.797 | 0.776 | +0.021 |
| Risk-taking | 0.649 | 0.618 | +0.031 |
| Impulsivity | 0.565 | 0.526 | +0.039 |
| Deference | 0.484 | 0.454 | +0.030 |

Self-steering consistently outperforms cross-steering, confirming that vectors are persona-conditioned. The gap is largest for traits with low shared variance (impulsivity, risk-taking, deference), consistent with the geometric analysis.

![Behavioural transfer heatmaps per trait](figures/placeholder_behavioural_transfer_heatmaps.png)
*Figure 5: Behavioural transfer scores by trait (source persona x target persona). Yellow/green = high trait expression; purple = low.*

**Qualitative finding --- source persona "residues":** Cross-steering introduces biases and quirks from the source persona. For example, applying a "confidence" vector extracted from the Tech CEO to the Farmer produces grandiose, empire-building language, while the same trait from the Therapist produces validating, emotionally grounded responses. The trait direction is the same, but the *flavour* carries the source persona's identity.

## Result 4: Activation Oracle

We test whether an LLM can identify the trait and persona encoded in a steering vector by presenting the vector's effect on model activations and asking the model to classify it. This tests how semantically interpretable the vectors are.

### Trait Identification

| Metric | IV Vectors | CAA Vectors |
|--------|-----------|-------------|
| Closed-set accuracy | 26.3% (21/80) | 31.3% (25/80) |
| Open-ended accuracy | 41.3% (33/80) | 21.3% (17/80) |

Per-trait accuracy varies dramatically:

| Trait | IV Accuracy | CAA Accuracy |
|-------|------------|--------------|
| Empathy | 100% | 90% |
| Confidence | 100% | 60% |
| Assertiveness | 10% | 100% |
| All others | 0% | 0% |

![Oracle trait confusion matrices](figures/placeholder_oracle_trait_confusion.png)
*Figure 6: Trait confusion matrices for the activation oracle under IV (left) and CAA (right) vectors.*

Only a few traits (empathy, confidence, assertiveness) are reliably identifiable from their vectors. Others (honesty, deference, warmth, impulsivity, risk-taking) are consistently misclassified --- often confused with empathy or confidence, suggesting these traits may not have cleanly separable semantic signatures in activation space.

### Persona Identification

Persona identification is much harder:

| Metric | IV Vectors | CAA Vectors |
|--------|-----------|-------------|
| Closed-set accuracy | 17.5% (14/80) | 12.5% (10/80) |
| Open-ended accuracy | 2.5% (2/80) | 0% (0/80) |

The drill sergeant is the most identifiable persona (62.5% IV, 75% CAA). Most other personas are at or near chance. Many are misclassified as drill sergeant, suggesting a dominant "authoritative" attractor in activation space.

![Oracle persona confusion matrices](figures/placeholder_oracle_persona_confusion.png)
*Figure 7: Persona confusion matrices for the activation oracle under IV (left) and CAA (right) vectors.*

## Result 5: Training Trajectory (OLMo-2-1124-7B)

We apply the same extraction method across 7 OLMo training checkpoints to track how persona-conditioning of steering vectors evolves through the training pipeline.

### Shared Variance Across Training

| Stage | Mean Shared Variance | Range |
|-------|---------------------|-------|
| Pretrain 1% | 95.7% | 94.7--96.9% |
| Pretrain 10% | 96.5% | 95.1--97.8% |
| Pretrain 50% | 96.0% | 94.9--97.6% |
| Base (100%) | 96.8% | 95.9--97.9% |
| SFT | 85.5% | 79.5--89.4% |
| DPO | 84.2% | 77.6--88.7% |
| Instruct | 83.5% | 76.1--88.2% |

![Shared variance across training stages](figures/placeholder_variance_trajectory.png)
*Figure 8: Shared variance ratio across OLMo training stages for all 8 traits. Dashed line marks the pre-training/post-training boundary. Note the sharp drop at SFT.*

**Key finding:** During pre-training, steering vectors are nearly universal across personas (~96% shared variance). A **sharp drop occurs at SFT**, where shared variance falls by ~11 percentage points. DPO and Instruct produce further modest decreases. This demonstrates that **post-training (not pre-training) creates persona-conditional trait representations**.

### Transfer Matrix Distances

Frobenius distances between transfer matrices at different stages confirm two distinct regimes:

- **Pre-training stages** are close to each other (Frobenius distances 0.07--0.17).
- **Post-training stages** are close to each other (SFT-DPO: 0.16, DPO-Instruct: 0.09).
- **The gap between pre- and post-training is large** (base to SFT: 1.44, base to Instruct: 1.67).

The Spearman correlations tell the same story: post-training stages are nearly identical in their transfer structure (rho > 0.99), while pre- vs post-training correlations are moderate (rho ~ 0.55--0.64).

![Transfer matrix distance heatmap across training stages](figures/placeholder_transfer_matrix_distances.png)
*Figure 9: Frobenius distances between cross-persona transfer matrices at each training stage. Two distinct clusters visible: pre-training and post-training.*

---

## Summary of Key Findings

1. **Steering vectors are persona-conditioned.** The same trait is encoded differently depending on the active persona, in both direction and magnitude.

2. **RLHF-trained traits show more universality.** Honesty (heavily optimised during training) has the highest shared variance; impulsivity and risk-taking (less constrained) have the lowest.

3. **Extraction method matters.** IV produces more universal vectors (higher shared variance) than CAA, suggesting that the method of extraction affects how much persona-specific information is captured.

4. **Self-steering outperforms cross-steering.** Behavioural evaluation confirms the geometric findings --- persona-matched vectors are more effective, especially for persona-specific traits.

5. **Post-training creates persona-conditioning.** The OLMo trajectory shows that pre-trained models have near-universal steering vectors; the sharp differentiation occurs at SFT and deepens through DPO.

6. **Cross-steering introduces source persona "residues."** Applying one persona's vector to another introduces the source persona's behavioural flavour, not just the intended trait.

## Implications for Safety

Safety and alignment work implicitly assumes steering vectors are model-global --- that a sycophancy suppression vector extracted from the default assistant works equally for a chat assistant, an autonomous agent, or a roleplayed character. Our results challenge this assumption: persona-specific representations mean that a single intervention may not generalise across deployment contexts. Safety teams may need **persona-aware interventions** rather than one-size-fits-all steering vectors.

## Next Steps

- Extend to safety-critical traits (sycophancy, refusal, power-seeking) with the prediction that refusal will be highly shared (like honesty)
- Measure trait coupling --- side-effects of single-trait steering on other traits
- Validate on additional model families (OLMo 3, Llama)
- Improve behavioural evaluations with better datasets and judges
- Robustness checks: bootstrap steering vectors, vary number of instruction variants

## References

- Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509.
- Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. arXiv:2601.10387.
