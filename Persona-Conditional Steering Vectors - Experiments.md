---
tags: #ai-safety #interpretability #research #draft #project
date: 2026-02-27
---

# Persona-Conditional Steering Vectors: Experimental Design

Companion to [[The Persona Landscape]]. This document describes a concrete experimental programme to test whether steering vectors are persona-dependent, designed to be implemented as a public research repo.

## Core Question

**Do steering vectors for the same trait change depending on which persona the model is currently operating under?**

If we induce different personas and then extract steering vectors for specific behavioural traits, are those vectors the same across personas - or does each persona have its own version?

This is a direct test of whether the persona landscape has *curvature*. If steering vectors are universal, the landscape is flat (Euclidean) and safety interventions generalise freely. If they're persona-dependent, the landscape is curved and safety interventions must be calibrated per-persona.

## Experimental Protocol

### Step 1: Induce Personas Along the Assistant Axis

Using the methodology from the [Assistant Axis paper](https://www.anthropic.com/research/assistant-axis) and [Persona Vectors](https://arxiv.org/abs/2507.21509):

**Persona induction methods:**
- **System prompt conditioning**: Use system prompts that reliably position the model at different points along the assistant axis
- **Activation injection**: Use the assistant axis direction itself to steer the model to specific positions (more precise, avoids confounds from prompt wording)
- **Few-shot examples**: Provide in-context examples characteristic of each persona

**Target personas:**
- **Persona A** - Non-assistant (base model / pre-trained character): Induced by moving away from the assistant end of the assistant axis. Characteristics: less helpful, less structured, more like raw internet text completion.
- **Persona B** - Assistant: The default post-trained helpful assistant persona. Standard system prompt.
- **Persona X** - Arbitrary positions along the assistant axis: Sample multiple points between A and B to test whether effects are continuous or discrete.

**Validation**: Confirm persona induction by measuring position along the assistant axis using the probes from the Assistant Axis paper. Each persona must produce a reliably distinct activation signature before proceeding.

### Step 2: Extract Steering Vectors for a Set of Traits (T)

For each induced persona, extract steering vectors for a fixed set of behavioural traits using contrastive activation pairs.

**Trait set T** (start with well-studied traits, expand later):
- **Honesty / deception** - willingness to provide truthful information
- **Sycophancy / directness** - tendency to agree with the user vs push back
- **Verbosity / conciseness** - output length and detail level
- **Compliance / refusal** - willingness to follow instructions vs refuse
- **Confidence / hedging** - certainty in responses
- **Formality / casualness** - register and tone

**Extraction method**: Use the contrastive activation addition approach (Turner et al., 2023; Rimsky et al., 2023). For each trait:
1. Create matched prompt pairs that differ only on the target trait
2. Run forward passes through the model under each induced persona
3. Take the mean difference in residual stream activations at a target layer as the steering vector
4. Repeat across multiple prompt pairs and average to reduce noise

This produces a matrix of steering vectors: for each persona P and trait T, we have a vector `P_T`.

### Step 2.5: Compare Steering Vectors Across Personas

**Analysis**: For each trait T, compare the steering vectors extracted under different personas.

**Metrics:**
- **Cosine similarity**: `cos(A_T, B_T)` - are the vectors pointing in the same direction?
- **Magnitude ratio**: `||A_T|| / ||B_T||` - does the same trait shift require more activation distance from one persona?
- **Residual after projection**: project `A_T` onto `B_T` and measure the orthogonal residual - this is the persona-specific component

**Prediction 0**: We anticipate that these vectors are **tangibly different**, but the closer together the personas are in the landscape, the less different their steering vectors will be. Specifically:
- `cos(A_T, B_T)` will be significantly less than 1.0 for most traits (not identical)
- `cos(A_T, B_T)` will be significantly greater than 0.0 for most traits (not orthogonal - there's a shared component)
- For Persona X at intermediate positions along the assistant axis, `cos(A_T, X_T)` should correlate with the distance between A and X on the assistant axis

If Prediction 0 fails (vectors are identical across personas), the landscape is flat and the rest of the experiment is less interesting but still informative. If it holds, we proceed to causal tests.

### Step 3: Cross-Persona Steering Effectiveness

This is the key causal test. Apply each persona's steering vectors to *other* personas and measure effectiveness.

**Protocol:**
1. Steer Persona A using `A_T` (same-persona steering - baseline)
2. Steer Persona A using `B_T` (cross-persona steering)
3. Steer Persona B using `B_T` (same-persona steering - baseline)
4. Steer Persona B using `A_T` (cross-persona steering)
5. Repeat for all persona pairs including intermediate Persona X positions

**Effectiveness measurement:**
- Behavioural evaluation: Does the model's output actually shift on the target trait? Use LLM-as-judge and/or trait classifiers to score outputs.
- Activation evaluation: Does the steering vector move the model's activations in the intended direction? Measure via trait probes.
- Side-effect evaluation: Does cross-persona steering produce unintended trait changes? Measure all traits, not just the target.

**Prediction 1**: The steering vector for Persona A will have **less effect when applied to Persona B** than to Persona A (and vice versa). Same-persona steering will consistently outperform cross-persona steering. This demonstrates that steering vectors are persona-local, not universal.

**Prediction 2**: Steering vectors for personas that are **closer together** in the landscape will be **more effective** when cross-applied than vectors from distant personas. Effectiveness should decay as a function of inter-persona distance along the assistant axis.

**Quantitative target**: If same-persona steering produces an effect size of `d`, cross-persona steering should produce `d * f(distance)` where `f` is a decreasing function of persona distance. Characterising `f` tells us about the curvature of the landscape.

### Step 4 (Extension): Steering Direction vs Inter-Persona Direction

**Question**: Is the steering vector for a trait the same as the direction between personas that differ on that trait?

**Protocol:**
1. Compute the inter-persona axis: the direction from Persona A to Persona B in activation space
2. For traits where A and B differ (e.g., if A is less compliant than B), compute the steering vector for that trait
3. Measure the angle between the inter-persona axis and the steering vector

**If they align**: Personas *are* their behaviours. Moving along the assistant axis *is* the same as steering on compliance, helpfulness, etc.

**If they diverge**: There is a "persona identity" component separate from any single behavioural trait. You can change behaviour without changing persona. This would be significant - it would mean post-training creates a persona identity that's more than the sum of its behavioural parts.

## Resource Requirements

### Compute

**Models**: Open-weights models with accessible residual streams. Recommended:
- **Primary**: Gemma 2 27B or Qwen 3 32B (used in the Assistant Axis paper, enabling direct comparison)
- **Validation**: Llama 3.3 70B (larger model, tests scale dependence)
- **Lightweight iteration**: Gemma 2 9B or Llama 3.1 8B (fast iteration during development)

**GPU requirements:**
- **Steering vector extraction** (forward passes only, no training):
  - 27B-32B models: 1x A100 80GB or 2x A100 40GB (model fits in ~54-64GB in fp16, or ~27-32GB in int8)
  - 70B model: 2x A100 80GB (model in fp16) or 1x A100 80GB (int8 quantisation, some precision loss)
  - 8-9B models: 1x A100 40GB or even 1x RTX 4090 24GB (int8)
- **Batch processing**: Extraction requires many forward passes (N_personas × N_traits × N_prompt_pairs × N_layers). With ~5 personas, ~6 traits, ~100 prompt pairs, ~32 layers = ~96,000 forward passes per model. At ~0.5s per forward pass on A100 for 27B model → ~13 hours. Parallelisable across GPUs.
- **Steering effectiveness evaluation**: Similar scale - need forward passes with steering applied, plus generation for behavioural evaluation.

**Estimated total compute**: ~50-100 A100-hours for the full experimental programme on one model. ~200-400 A100-hours for three models.

**Cloud cost estimate** (at ~$2/A100-hour on Lambda/RunPod spot):
- Minimum viable experiment (one 27B model): ~$100-200
- Full programme (three models): ~$400-800
- With iteration and debugging buffer: ~$1,000-1,500

### Data

**Contrastive prompt pairs**: Need ~100 matched pairs per trait. Can be generated synthetically:
- Use an LLM to generate prompt pairs that differ only on the target trait
- Validate a subset manually to ensure quality
- Reuse existing datasets where available (e.g., TruthfulQA for honesty, sycophancy benchmarks)

**Persona induction prompts**: Need system prompts and few-shot examples for each target persona. The Assistant Axis paper provides the character archetype taxonomy to draw from.

**Evaluation data**: Separate held-out prompts for measuring steering effectiveness. ~200 evaluation prompts per trait.

### Software

**Core dependencies:**
- `transformers` (HuggingFace) - model loading and forward passes
- `nnsight` or `transformer_lens` - activation extraction and intervention
- `torch` - tensor operations
- `baukit` or custom hooks - residual stream access at specific layers

**Evaluation:**
- LLM-as-judge for behavioural scoring (can use API models - Claude or GPT-4)
- Custom trait classifiers (train on labelled data or use existing)

**Analysis:**
- Standard scientific Python: `numpy`, `scipy`, `matplotlib`, `seaborn`
- Dimensionality reduction: PCA, UMAP for visualisation of persona/steering spaces

### Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup & infrastructure | 1 week | Model loading, activation extraction pipeline, persona induction validation |
| Step 1: Persona induction | 1 week | Validate persona positions along assistant axis |
| Step 2: Steering vector extraction | 1-2 weeks | Extract vectors for all persona × trait combinations |
| Step 2.5: Vector comparison | 1 week | Analysis of cosine similarities, magnitudes, residuals |
| Step 3: Cross-persona effectiveness | 2 weeks | Full steering experiments + behavioural evaluation |
| Step 4: Direction comparison | 1 week | Inter-persona axis vs steering direction analysis |
| Analysis & writeup | 2 weeks | Statistical analysis, figures, paper draft |
| **Total** | **~9-11 weeks** | |

Could be compressed to ~5-6 weeks with dedicated full-time effort and fast GPU access.

## Expected Outputs

1. **Transfer matrix**: For each trait, a matrix showing steering effectiveness between all persona pairs. This is the core result.
2. **Curvature map**: How steering transfer decays as a function of inter-persona distance. Characterises the geometry of the persona landscape.
3. **Shared vs persona-specific decomposition**: For each trait's steering vector, the proportion that's universal vs persona-local.
4. **Steering ≠ persona direction**: Evidence for or against the claim that behavioural steering is the same as persona movement.

## Implications

**If predictions hold (steering vectors are persona-dependent):**
- Current safety interventions based on steering vectors may be less robust than assumed - they work for the persona they were calibrated on but degrade for others
- Safety evaluations should test interventions across multiple persona positions
- The persona landscape has meaningful curvature, making it a richer object of study than a simple vector space
- The Assistant Axis is not just a descriptive tool but a *necessary coordinate* for correctly calibrating interventions

**If predictions fail (steering vectors are universal):**
- The persona landscape is locally flat - good news for safety, interventions generalise
- Still valuable: confirms that steering vectors are robust to persona variation
- Suggests personas are "surface-level" and don't affect the model's deeper representational geometry

## References

- [Anthropic, "The Assistant Axis"](https://www.anthropic.com/research/assistant-axis)
- [Persona Vectors (arXiv, 2025)](https://arxiv.org/abs/2507.21509)
- Turner et al., "Activation Addition: Steering Language Models Without Optimization" (2023)
- Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition" (2023)
- [Anthropic PSM paper](https://alignment.anthropic.com/2026/psm/)
