# Persona-Conditional Steering Vectors — Project Overview

**Jacob Davies, Rhea Karty** · ERA:AI Fellowship, Winter 2026

This is the consolidated project overview: the research question, method, headline
results, experimental design, and robustness battery. For how to *run* each
experiment see [experiments.md](experiments.md); for the causal X-series pipeline
see [causal_pipeline.md](causal_pipeline.md); for full results see
[results/](results/).

---

## Research question

Steering vectors extracted from a single baseline persona are widely used for
monitoring and controlling model behaviour (Chen et al., 2025; Lu et al., 2026).
But models are deployed across diverse operational modes, each a distinct persona
state. **Do steering vectors transfer across personas, or do personas
fundamentally reshape trait representations?**

Framed geometrically: does the persona landscape have *curvature*? If steering
vectors are universal, the landscape is flat (Euclidean) and safety interventions
generalise freely. If they are persona-dependent, the landscape is curved and
interventions must be calibrated per-persona.

## Method

We induce **persona archetypes** via system prompts and extract steering vectors
for **8 traits** (assertiveness, empathy, risk-taking, honesty, confidence,
deference, warmth, impulsivity).

The 10 core persona archetypes are: Farmer, Politician, Therapist, Drill
Sergeant, Street Hustler, Professor, Tech CEO, Kindergarten Teacher, Surgeon,
Con Artist. The published activation dataset extends this to **17 personas**
(adding control personas `null`/`nonsense` and extensions such as
`pathological_liar`, `six_year_old`, `sociopath`, `actor_in_rehearsal`,
`contrarian_deceiver`) — see the dataset card for the full list.

**Two extraction methods:**

- **Instruction-Variant (IV):** 5 pos/neg instruction pairs × 20 sampled
  questions = 100 contrastive pairs per persona × trait. The same question under
  opposing instructions isolates the trait signal from content.
- **Contrastive Activation Addition (CAA):** the standard contrastive-pair
  method from the literature (Turner et al., 2023; Rimsky et al., 2023).

Contrastive vector = mean(pos activations) − mean(neg activations).

**Primary model:** Gemma-2-27B-IT, layer 22/46. **Training-trajectory model:**
OLMo-2-1124-7B across 7 checkpoints (pretrain 1/10/50%, base, SFT, DPO,
Instruct). Configs for Gemma-3-27B and Gemma-4-E4B also exist (see
`persona_steering/config.py` and [results/](results/)).

---

## Headline findings

1. **Steering vectors are persona-conditioned.** The same trait is encoded
   differently depending on the active persona, in both direction and magnitude.

2. **RLHF-trained traits show more universality.** Honesty (heavily optimised in
   training) has the highest shared variance; impulsivity and risk-taking (less
   constrained) the lowest.

3. **Extraction method matters.** IV produces more universal vectors (higher
   shared variance) than CAA, suggesting IV extracts a cleaner trait signal while
   CAA captures more persona-entangled information.

4. **Self-steering outperforms cross-steering.** Behavioural evaluation confirms
   the geometric findings — persona-matched vectors are more effective,
   especially for persona-specific traits.

5. **Post-training creates persona-conditioning.** The OLMo trajectory shows
   pre-trained models have near-universal steering vectors (~96% shared
   variance); a sharp differentiation occurs at SFT (−11 pp) and deepens through
   DPO/Instruct.

6. **Cross-steering introduces source-persona "residues."** Applying one
   persona's vector to another carries the source persona's behavioural flavour,
   not just the intended trait.

### Shared vs persona-specific variance (Gemma-2-27B-IT, layer 22)

| Trait | IV shared variance | CAA shared variance |
|---|---|---|
| Honesty | 92.8% | 72.7% |
| Assertiveness | 88.5% | 78.0% |
| Confidence | 86.4% | 76.7% |
| Warmth | 85.1% | 67.4% |
| Deference | 83.9% | 50.2% |
| Empathy | 83.8% | 68.6% |
| Impulsivity | 75.2% | 54.2% |
| Risk-taking | 72.9% | 56.9% |

No trait exceeds 93% shared variance under either method — all encode substantial
persona-specific signal. Under IV the 80% line separates RLHF-shaped traits
(honesty, assertiveness, confidence, warmth, deference, empathy) from less
constrained ones (impulsivity, risk-taking).

### Self vs cross-persona behavioural steering

| Trait | Self-steer | Cross-steer | Gap |
|---|---|---|---|
| Confidence | 0.896 | 0.892 | +0.004 |
| Assertiveness | 0.861 | 0.856 | +0.005 |
| Empathy | 0.851 | 0.848 | +0.003 |
| Warmth | 0.795 | 0.787 | +0.008 |
| Honesty | 0.797 | 0.776 | +0.021 |
| Risk-taking | 0.649 | 0.618 | +0.031 |
| Impulsivity | 0.565 | 0.526 | +0.039 |
| Deference | 0.484 | 0.454 | +0.030 |

The cross-steering gap is largest for low-shared-variance traits, consistent with
the geometric analysis. Full per-trait results and multi-model comparison are in
[results/summary.md](results/summary.md).

### Training trajectory (OLMo-2-1124-7B)

| Stage | Mean shared variance |
|---|---|
| Pretrain 1% | 95.7% |
| Pretrain 10% | 96.5% |
| Pretrain 50% | 96.0% |
| Base (100%) | 96.8% |
| SFT | 85.5% |
| DPO | 84.2% |
| Instruct | 83.5% |

Pre-training stages cluster tightly (Frobenius distance 0.07–0.17); post-training
stages cluster tightly (0.09–0.16); the gap between them is large (base→Instruct
1.67). **Post-training, not pre-training, creates persona-conditional trait
representations.**

---

## Experimental design

The conceptual programme has four steps; the implemented pipeline (steps 0–9 plus
the E/X series) operationalises them.

1. **Induce personas** via system prompts (5 prompt variants per persona for
   robustness), and validate each produces a distinct activation signature.
2. **Extract steering vectors** for each trait under each persona (IV and CAA).
3. **Compare vectors across personas** — cosine similarity, magnitude ratio, and
   the orthogonal residual after projection (the persona-specific component).
   *Prediction:* `cos(A_T, B_T)` significantly below 1 (not identical) but above
   0 (a shared component exists).
4. **Cross-persona steering effectiveness** (the key causal test) — apply each
   persona's vector to other personas; measure behavioural shift (LLM-as-judge),
   activation shift (probes), and side-effects on other traits. *Prediction:*
   same-persona steering beats cross-persona steering, and effectiveness decays
   with inter-persona distance.

**Core outputs:** per-trait transfer matrices; the curvature map (transfer decay
vs distance); shared/specific decomposition; and the test of whether the steering
direction equals the inter-persona direction.

## Robustness battery (R1–R5)

Five focused robustness checks on the canonical 10×8 grid at layer 22, with `null`
and `nonsense` personas as baselines. Full writeups:
[results/robustness_iv.md](results/robustness_iv.md) and
[results/robustness_caa.md](results/robustness_caa.md).

| Exp | What it measures | Headline (IV) |
|---|---|---|
| R1 | 50× bootstrap-resampled vectors; pairwise cosine + full-data alignment | pairwise 0.990 ± 0.008; full-data 0.995 ± 0.004 |
| R2 | Vector↔reference cosine vs N pairs ∈ {1,2,5,10,20,50,100,525} | convergence curve |
| R3 | Within-persona-across-variant vs across-persona-within-variant cosine (Mann-Whitney) | within 0.655 ± 0.140; across 0.719 ± 0.088; p = 0.0072 |
| R4 | Each persona vector vs the general (all-persona-mean) vector, plus null/nonsense baselines | per-trait context-dependence ordering |
| R5 | Same-context-pair vs random-pair cosine significance | labelled 0.838; random 0.805; p = 0.091 (trend) |

CAA mirrors the IV pattern (R1 stability 0.989 ± 0.006 despite ~10× fewer pairs).

## Extended experiments (E-series and beyond)

Operational details and commands in [experiments.md](experiments.md):

- **E2** bootstrap CIs on shared variance · **E3** IV–CAA geometric decomposition
- **E4** probe transfer across contexts (safety-relevant: monitoring-probe decay)
- **E5** SAE feature divergence · **E7** SAE sparse codes as persona fingerprints
- **E6** basin geometry (cosine-to-default decays along persona gradients)
- **X-series** causal-figures pipeline — see [causal_pipeline.md](causal_pipeline.md)

---

## Implications for safety

Safety and alignment work implicitly assumes steering vectors are model-global —
that, e.g., a sycophancy-suppression vector extracted from the default assistant
works equally for a chat assistant, an autonomous agent, or a roleplayed
character. These results challenge that assumption: persona-specific
representations mean a single intervention may not generalise across deployment
contexts. Safety teams may need **persona-aware interventions** rather than
one-size-fits-all steering vectors, and safety evaluations should test
interventions across multiple persona positions.

## Open directions

- Extend to safety-critical traits (sycophancy, refusal, power-seeking);
  prediction: refusal is highly shared, like honesty.
- Measure trait coupling — side-effects of single-trait steering on other traits.
- Validate on additional model families (already have Gemma-3, Gemma-4-E4B, OLMo;
  extend to Llama).
- Improve behavioural evaluations with better datasets and judges.

## References

- Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). *Persona
  Vectors: Monitoring and Controlling Character Traits in Language Models.*
  arXiv:2507.21509.
- Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). *The
  Assistant Axis: Situating and Stabilizing the Default Persona of Language
  Models.* arXiv:2601.10387.
- Turner, A., et al. (2023). *Activation Addition: Steering Language Models
  Without Optimization.*
- Rimsky, N., et al. (2023). *Steering Llama 2 via Contrastive Activation
  Addition.*
- Anthropic. *The Assistant Axis.* https://www.anthropic.com/research/assistant-axis
