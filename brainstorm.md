# Persona-Conditional Representations: A Brainstorm

## Core claim

The Linear Representation Hypothesis (LRH) and Representation Engineering (RepEng) treat a concept as a single direction in activation space. But an LLM is trained on text produced by many different subjective experiences — many personas. Its internal representation of a concept is therefore not a single linear direction but a structured object that varies with the active persona. A single "concept direction" is a readout of that object at a particular persona point, and the model's approximation to a universal concept direction is some aggregation function over persona-specific directions.

This reframes two existing pictures and connects them:

- **RepEng / LRH** gives us "concepts are linear directions."
- **Persona Selection Model** gives us "the model maintains a superposition of simulacra; conditioning collapses it."

The synthesis: a concept in the model is not a vector but an operator that maps (persona state) → (direction in activation space). Standard steering vectors are point estimates of this operator evaluated at the Assistant persona.

## Two versions of the claim

**Weak version.** Trait concepts are linear *within* a persona, but the direction translates/rotates across personas. LRH holds locally; the global object is a union of linear subspaces indexed by persona. Probes transfer imperfectly because they are evaluated at one persona and applied at another.

**Strong version.** There is no single "true" linear direction. The representation is a smooth object — a manifold — on which each persona is a point or region. A trait is a vector field on the manifold; the Assistant direction is that field's value at the Assistant point; the "universal" direction is a Fréchet mean of the field weighted by the persona prior from training.

The strong version is the more interesting contribution and is what our work seems to be reaching toward.

## Geometry: union of subspaces vs. manifold

**Union of linear subspaces (UoS).** Each persona p gets its own subspace S_p ⊂ ℝ^d. The union has no smooth structure at junctions. There is no well-defined "between" two personas. This is the geometry of sparse coding and classical subspace clustering.

**Manifold.** A single smooth surface M. Every point has a tangent space. Continuous movement between personas is well-defined. LRH holds locally (each tangent space is linear) but not globally (the tangent spaces at different personas are not the same linear subspace).

**Why manifold is the better default.**

- *Training-time:* the model saw billions of continuously varying humans-in-text, not ten discrete archetypes. Gradient descent on continuous inputs tends to produce continuous internal structure.
- *Prompt-time:* blended prompts ("a farmer who is secretly a con artist") produce graded, coherent behavior rather than bistable snapping between modes.
- *Mathematical:* manifolds locally look Euclidean, which recovers LRH within a persona for free.

The likely picture is **manifold with cluster structure**: ten dense regions on a smooth surface, each locally well-approximated by an affine subspace. The failure mode of naive LRH is assuming those local subspaces are all the same subspace in ℝ^d, translated.

## The clean mathematical reframe

Treat each trait as a **vector field on the persona manifold**. At every point p ∈ M, the trait picks out a direction v_trait(p) in the tangent space T_p M.

- Our 10 persona-specific steering vectors are samples of this vector field at 10 points.
- The Assistant trait direction is the field's value at the Assistant point.
- The "true" universal direction is a Fréchet mean over the manifold, weighted by the training prior over personas.

Under this view:

- LRH is right locally — at each persona, the trait is a linear direction in the local tangent space.
- LRH is wrong globally — no single linear direction works everywhere, because the manifold is curved.
- UoS is a degenerate special case: zero curvature, disjoint subspaces, no interpolation.

Steering vectors as currently constructed are operator point estimates. This explains why they leak across contexts, why they degrade under jailbreaks (different persona → different operator value), and why persona-conditional vectors should outperform a generic one on in-persona behavior.

## Implications

- **Steering.** Persona-conditioned steering should dominate persona-agnostic steering on in-persona behavior. The generic trait vector is a marginalized quantity that is optimal on average but suboptimal for any specific persona.
- **Probing.** Linear probes measure the operator at the training persona. Transfer failures across contexts are curvature, not noise.
- **Robustness.** Alignment interventions that assume a single concept direction will be undermined by persona shifts (including adversarial ones). The relevant invariant to preserve is the vector field, not any single vector.
- **Philosophy.** The "true" direction is a property of the aggregation function and the training prior, not of the world. It is the model's best linear summary of a fundamentally higher-dimensional structure.

---

# Experiments

The experiments below are ordered from cheapest/most decisive to more involved. The first three use our existing 10-persona × 8-trait data; the later ones require some new generation.

## E1. Assistant ≈ centroid?

**Question.** Is the Assistant trait vector well-approximated by the Euclidean mean of the 10 persona trait vectors?

**Method.** For each trait t, compute m_t = mean over personas of v_{p,t}. Compare m_t to the Assistant's trait vector v_{assistant, t} (extracted with no persona system prompt, default Assistant behavior) via cosine similarity and relative L2 error. Compare to two baselines: (a) cosine between Assistant vector and the single closest persona vector; (b) cosine between two random persona vectors.

**Prediction.**
- Strong manifold / aggregation claim: Assistant ≈ centroid with higher similarity than any individual persona.
- Weak LRH: all vectors are nearly collinear; centroid and individuals are all similar. Uninformative.
- UoS: Assistant lives in its own subspace; centroid is no better than chance.

**Why it matters.** Directly tests the "Assistant is a marginalization over personas" claim. Single plot, high leverage.

## E2. Shared + specific decomposition

**Question.** Is the persona-specific residual behaviorally load-bearing?

**Method.** Decompose v_{p,t} = u_t + w_{p,t} with u_t = mean over p. Steer with (a) u_t alone, (b) u_t + w_{p,t}, (c) u_t + w_{p',t} for a different persona p' (wrong-persona residual), at matched magnitude. Score outputs with Claude as judge on in-persona trait expression AND persona coherence.

**Prediction.**
- If (a) ≈ (b) > (c): shared direction carries the trait signal; residuals are noise. Weak LRH wins.
- If (b) > (a) ≫ (c): residuals are load-bearing and persona-specific. Manifold view wins.
- If (b) ≈ (c) > (a): residuals help but aren't persona-specific (they're just "more signal"). Ambiguous.

**Why it matters.** Tells us whether the manifold structure has behavioral content or is just a geometric curiosity.

## E3. Euclidean vs. Fréchet residual as curvature proxy

**Question.** Does the deviation between the Assistant vector and the Euclidean centroid grow with persona spread?

**Method.** For each trait, measure (a) the residual r_t = ||v_{assistant,t} − m_t||, (b) the spread s_t = mean pairwise distance among persona vectors. Regress r across traits on s.

**Prediction.**
- Flat manifold: r_t is small regardless of s_t. Euclidean averaging is valid.
- Curved manifold: r_t scales with s_t. Euclidean averaging systematically misses.

**Why it matters.** This is the quickest diagnostic for whether we need manifold machinery at all, or whether linear aggregation suffices in practice.

## E4. Interpolation test — the decisive UoS vs. manifold test

**Question.** Is the trait direction continuous between personas?

**Method.** For a pair of personas (e.g. Farmer, Politician), construct a sequence of system prompts that smoothly blend between them. One option: convex combinations of embeddings of the two system prompts; another: natural-language blends ("a former farmer now running for office," "a farmer who is increasingly political," etc.) rated by Claude for blend fraction. At each blend level, extract the trait vector.

**Prediction.**
- UoS: trait vector stays near Farmer, snaps to near Politician, possibly with a high-variance region at the transition.
- Manifold: trait vector traces a smooth curve in ℝ^d between the endpoints.

Quantify by measuring cosine similarity to each endpoint along the path and looking for smoothness vs. step-function behavior.

**Why it matters.** This is the cleanest geometric test. A smooth curve is strong evidence against UoS.

## E5. Parallel transport and curvature

**Question.** Is the manifold genuinely curved, or is it flat (directions differ only by translation)?

**Method.** Take the trait vector at Farmer. Transport it to Politician along a smooth prompt-interpolation path (track how the tangent space rotates). Compare transported vector to the directly-measured trait vector at Politician. Repeat along a different path.

**Prediction.**
- Flat: transported = measured, and path-independent. LRH holds with a global linear direction plus persona-specific offset.
- Curved: transported ≠ measured, and path-dependent. Genuine non-linearity.

**Why it matters.** Distinguishes "one true direction that looks different because of offsets" from "no single direction exists."

## E6. Dimensionality and structure of the persona cloud

**Question.** What is the intrinsic dimension of a single persona's activation cloud, and of the 10-persona set?

**Method.** For each persona, collect many activation samples (across questions, instruction variants, token positions). Compute local intrinsic dimension (e.g. via MLE on nearest-neighbor distances, or participation ratio of local PCA). Compute the intrinsic dimension of the combined cloud. Compute the intrinsic dimension of the 10 trait-vector endpoints for each trait (are they 2D, 3D, or scattered in 10D?).

**Prediction.**
- Manifold: uniform local dimension across personas, with a higher global dimension driven by persona variation.
- UoS: per-persona dimensions similar but global dimension equals sum (disjoint subspaces).
- Low-dim trait subspace (e.g. 2–3D): all 10 personas' trait vectors lie in a small subspace, which *is* the trait structure we are trying to characterize.

**Why it matters.** Tells us the shape of the object. If the trait vectors for one trait live in a clean 2–3D subspace, that subspace is the object of interest; "the" direction is a 1D slice of it.

## E7. Cross-trait entanglement varies with persona

**Question.** Do traits interact differently under different personas?

**Method.** For each persona, compute the 8×8 cosine-similarity matrix among its trait vectors. Compare matrices across personas (e.g. Frobenius distance between matrices, or per-entry variance across personas).

**Prediction.**
- LRH with persona-independent traits: all 10 matrices nearly identical.
- Persona-dependent traits: off-diagonal entries vary systematically (e.g. empathy and assertiveness are anti-correlated for Drill Sergeant, correlated for Therapist).

**Why it matters.** If trait interactions vary with persona, traits aren't persona-independent primitives — their meaning is shaped by the character's coherence constraints. This is perhaps the deepest challenge to naive LRH.

## E8. Alignment with the assistant axis

**Question.** Does the manifold's principal axis align with the assistant axis from the reference repo?

**Method.** Run PCA on the pooled persona activations. Compute cosine similarity between the top principal components and the assistant axis vector.

**Prediction.**
- If the assistant axis is the "average-persona direction," the top PC should align with it, connecting our manifold picture to the existing assistant-axis framing.

**Why it matters.** Ties the new object (persona manifold) to an existing well-studied object (the assistant axis), strengthening the theoretical claim.

---

# Suggested ordering

1. **E1, E3, E7** use existing data, are cheap, and together diagnose whether we need the full manifold apparatus.
2. **E2** tests whether the structure is behaviorally real. Also uses existing data.
3. **E4, E5** require new generation but are the decisive geometric tests.
4. **E6, E8** contextualize the result.

If E1 + E3 + E7 all support the manifold picture, E4 becomes the headline experiment.

---

# Council Review

A Brainstorming council (Visionary, Contrarian, Pragmatist, Connector) plus a Methods Expert stress-tested the brainstorm over three rounds. Key outcomes.

## Convergence points

**1. Retreat from "manifold" as the headline claim.** The defensible contribution is that the trait vector field on persona-space is non-trivial and behaviorally load-bearing. Manifold-vs-UoS is a downstream geometric question tested by experiment, not the framing. What we actually have today is "ten clusters in activation space with internal linear structure and systematic inter-cluster variation" — weaker than a manifold but defensible and falsifiable.

**2. Persona-mean residualization is a mandatory control.** The strongest boring explanation is that persona prompts shift the activation baseline, and trait vectors measured relative to that baseline are actually identical. Subtracting persona means before extracting trait vectors is a gating experiment that must pass before any geometric claim is made.

**3. The behavioral payoff (E2) gates the paper; the interpolation and extrapolation tests (E4 + E11) gate the geometric claim.** Everything else is supporting.

**4. Trait vectors should be treated as distributional objects, not points.** Each extraction has statistical uncertainty. Cross-persona comparisons should be significance tests, not cosine inspections. The tangent space at a persona is better modeled as the activation covariance under that persona, not as a point estimate.

**5. Rhetorical reframe worth adopting.** Existing RepEng results are *implicitly Assistant-conditioned*. Under persona shift they do not generalize. This reframe alone is a contribution and belongs near the top of any write-up.

## Additional experiments the council surfaced

### E9. Persona-mean residualization (mandatory control)

**Question.** Does the persona-trait vector variation survive removal of persona-specific activation baselines?

**Method.** For each persona p, compute the mean activation across all inputs (not just trait-relevant ones). Subtract this mean from all activations for persona p before extracting trait vectors. Re-run E1–E3 on the residualized vectors.

**Prediction.**
- Trait vectors collapse to near-identical directions across personas: boring explanation wins. The apparent manifold was baseline drift.
- Trait vectors still differ systematically: persona-conditional trait structure is real, not a baseline artifact.

**Why it matters.** This is the first experiment a skeptical reviewer will demand. Running it early avoids building a story on an artifact.

### E10. Prompt-content matching (confound control)

**Question.** Does the manifold structure survive when persona prompts are matched for surface-level properties?

**Method.** Generate length-, vocabulary-, and register-matched paraphrases of each persona prompt (Claude can produce these). Re-extract persona-trait vectors. Test whether inter-persona differences persist.

**Prediction.**
- Structure persists: personas encode semantic identity, not lexical surface.
- Structure collapses: we were measuring prompt-form variation dressed up as persona.

**Why it matters.** Second-line skeptic defense. If E9 passes, E10 is the next thing to rule out.

### E11. Persona-space extrapolation (decisive geometry test, cheap)

**Question.** Is persona-space linearly navigable?

**Method.** Apply (v_Politician − v_Farmer) to Farmer's activations and generate. Score outputs with Claude for (a) Farmer-ness decrease, (b) Politician-ness increase, (c) overall coherence.

**Prediction.**
- Smooth partial shift toward Politician with preserved coherence: persona-space supports linear extrapolation, manifold picture is vindicated.
- Incoherent Franken-persona output: persona structure is discrete or highly non-linear; closer to UoS.

**Why it matters.** Arguably cleaner and cheaper than E4's smooth interpolation. A single cross-persona extrapolation test may be more decisive than a path of blended prompts.

### E12. Cross-model universality

**Question.** Does the persona-cloud structure replicate across models?

**Method.** Extract persona activations on a second model family (e.g. Llama-3 or Qwen alongside Gemma). Run PCA on pooled persona activations. Compare top PCs — do "warmth," "assertiveness," "honesty" axes emerge in similar relative positions?

**Prediction.**
- Similar structure across model families: persona manifold is a property of how language models represent human variation, not of one model's weights.
- Model-specific structure: the phenomenon is real but limited in generality.

**Why it matters.** Separates "interesting property of this model" from "general feature of LLM representation." The latter is a much stronger paper.

### E13. SAE feature variance under persona shift

**Question.** Do supposedly monosemantic SAE features show systematic cross-persona variance consistent with the vector-field view?

**Method.** Using a publicly available SAE for a model we can run, identify trait-relevant features (e.g. a "deception" feature). Measure activation variance of these features within each persona vs. across personas. Compare to non-trait control features.

**Prediction.**
- Across-persona variance > within-persona variance for trait features: SAE features are fibers of a bundle, collapsed to their mean by the SAE objective. Predicted by the manifold theory.
- No systematic difference: either SAEs successfully factor out persona, or the manifold theory is wrong at the feature level.

**Why it matters.** Connects the work to the SAE / interpretability literature and produces a pre-registerable prediction that distinguishes our account from standard accounts.

## Methodological upgrades the council flagged

- **Statistical extraction.** Replace mean-difference trait vectors with LDA-style or Fisher-information-based extraction that returns both a direction and an uncertainty ellipsoid. Enables significance testing on cross-persona comparisons.
- **Power analysis.** Ten personas is likely underpowered for some of the cross-cluster tests. Either expand to 20–30 personas for key experiments or pre-register exactly which claims the current N supports.
- **Pre-registered null.** Specify, before running the geometric experiments, what "no manifold" would look like in the data. Otherwise post-hoc interpretation is too flexible.
- **Unlabeled persona recovery.** Can persona clusters be recovered from activation covariances without knowing which prompt generated them? If yes, persona-space has intrinsic structure, not just label-induced structure. This is a strong form of the base-space reality check.

## Updated experiment ordering

1. **E9, E10** (controls). Must pass before any geometric claim is made.
2. **E1, E2, E3, E7** (existing data, core diagnostics). E2 is the load-bearing behavioral test.
3. **E11** (cheap extrapolation). Decisive for discrete vs. continuous persona-space.
4. **E4, E5** (interpolation and curvature). Expensive but headline-worthy if earlier results hold.
5. **E6, E8** (dimensionality and assistant-axis alignment). Contextualize.
6. **E12, E13** (generality and SAE connection). Strengthen the broader claim.

## Revised abstract-shaped summary

Existing representation-engineering work treats concepts as single directions in activation space. Because language models simulate many personas, a concept is better modeled as a vector field over persona-space, with the standard "concept direction" being a point estimate of that field at the Assistant persona. We show that trait directions extracted under ten distinct persona conditions differ systematically, that the differences are behaviorally load-bearing under steering, and that the geometry of persona-space supports continuous but not trivially linear structure. We argue that existing RepEng results are implicitly Assistant-conditioned and identify the conditions under which they fail to generalize.
