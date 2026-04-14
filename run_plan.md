# First-Pass Run Plan: E9 → E1 → E2 → E11

Focused plan for the four experiments the council flagged as load-bearing: one control (E9), two diagnostics on existing data (E1, E2), and one cheap decisive geometric test (E11). Written to be executable against the current `outputs/gemma-2-27b-it/` tree.

## Guiding principles

1. **E9 is a gate.** If it fails, the geometric story is an artifact and the downstream experiments change meaning. Run E9 first and inspect before proceeding.
2. **Reuse existing activations where possible.** Don't regenerate anything that's already in `outputs/gemma-2-27b-it/activations/`.
3. **One layer first, sweep later.** Use layer 22 (middle of a 46-layer model, matches existing analysis default in `run.sh`). Only sweep if results at layer 22 are ambiguous.
4. **Small, named output dir per experiment.** All new artifacts under `outputs/gemma-2-27b-it/council/{e9,e1,e2,e11}/` so nothing collides with existing analyses.
5. **Every experiment ends with a single summary figure or JSON that directly answers its question.**

## Shared setup

Create a small helper module `persona_steering/council.py` with:

- `load_persona_activations(persona, trait, direction, layer)` → tensor of shape (n_samples, hidden)
- `load_trait_vector(persona, trait, layer)` → tensor of shape (hidden,)
- `compute_assistant_vector(trait, layer)` → trait vector extracted with no persona prompt (the default Assistant); likely already exists in `outputs/.../vectors/` under a distinguished persona slug — verify, generate if missing
- `score_generations(texts, trait)` → list of floats using `evaluation.py`'s Claude judge

If the Assistant-baseline trait vector is not already in the vectors tree, generating it is a prerequisite for E1. That's one additional run through `1_generate.py` + `2_activations.py` + `3_vectors.py` with an empty/default system prompt under a new persona slug `assistant_default`.

All four experiments operate on one layer (22) for the first pass. Outputs go to `outputs/gemma-2-27b-it/council/`.

---

## E9. Persona-mean residualization (gate)

**Question.** Does cross-persona trait-vector variation survive removal of persona-specific activation baselines?

**Script.** `pipeline/c9_residualization.py`

**Inputs.**
- `outputs/gemma-2-27b-it/activations/{persona}_{trait}_{pos|neg}.pt` for all 10 personas × 8 traits × 2 directions
- Layer 22 activations

**Procedure.**
1. For each persona p, pool all activations across all traits and directions (this is p's persona baseline sample). Compute `mu_p = mean(pooled_activations_p)`.
2. For each persona p, trait t, direction d, load activations `A_{p,t,d}`, subtract `mu_p` to get residuals `R_{p,t,d} = A_{p,t,d} - mu_p`.
3. Extract residualized trait vectors: `v_residual_{p,t} = mean(R_{p,t,pos}) - mean(R_{p,t,neg})`.
4. Also load the original (non-residualized) trait vectors `v_orig_{p,t}` from `outputs/.../vectors/`.
5. For each trait t, compute:
   - Mean pairwise cosine among the 10 original persona vectors: `cos_orig_t`
   - Mean pairwise cosine among the 10 residualized persona vectors: `cos_resid_t`
   - Cosine between residualized centroid and original centroid
6. Save summary JSON and one figure: bar chart of `cos_orig_t` vs `cos_resid_t` across traits, with a horizontal line at cosine = 0.95 for "effectively collapsed."

**Success criteria.**
- **Gate passed (manifold story survives):** for at least 6 of 8 traits, `cos_resid_t < 0.9`. The persona vectors remain measurably non-parallel after residualization.
- **Gate failed (boring explanation wins):** `cos_resid_t > 0.95` for most traits. Persona variation was baseline drift. Stop and reframe the project.
- **Ambiguous:** intermediate values. Proceed to E1 but flag the reduced manifold signal.

**Outputs.**
- `outputs/gemma-2-27b-it/council/e9/residualized_vectors.pt` (dict keyed by `(persona, trait)`)
- `outputs/gemma-2-27b-it/council/e9/summary.json` — per-trait cos_orig, cos_resid, verdict
- `outputs/gemma-2-27b-it/council/e9/fig_residualization.png`

**Expected runtime.** Minutes. Pure numpy on existing activations.

**Decision point after E9.** If gate fails, stop and discuss reframing before running E1–E11. If gate passes, all subsequent experiments use both original and residualized vectors and compare.

---

## E1. Assistant ≈ centroid?

**Question.** Is the Assistant trait vector well-approximated by the Euclidean mean of the 10 persona trait vectors?

**Script.** `pipeline/c1_assistant_centroid.py`

**Prerequisites.**
- Assistant-baseline vectors in `outputs/.../vectors/assistant_default_{trait}.pt`. Generate if missing (one pass through the pipeline with empty persona system prompt).

**Procedure.**
1. For each trait t:
   - Load 10 persona vectors `v_{p,t}` (both original and E9-residualized).
   - Load Assistant vector `v_assistant_t`.
   - Compute centroid `m_t = mean_p(v_{p,t})`.
   - Compute:
     - `cos(v_assistant_t, m_t)` — centroid similarity
     - `max_p cos(v_assistant_t, v_{p,t})` — best single persona
     - `mean pairwise cos(v_{p,t}, v_{p',t})` — random persona baseline
     - `||v_assistant_t - m_t|| / ||v_assistant_t||` — relative residual
2. Aggregate across traits into a summary table.
3. Optional: fit a weighted mean `m_t = sum w_p v_{p,t}` with non-negative weights summing to 1 that maximizes cosine to Assistant. Save learned weights. Skewed weights suggest non-uniform persona prior.

**Success criteria.**
- **Strong aggregation signal:** median across traits of `cos(v_assistant, centroid) > cos(v_assistant, best_single_persona)`, and > 0.6 in absolute terms.
- **Weak LRH regime:** all cosines are similar and high (>0.9). The centroid claim is not distinguishable from "all vectors are nearly parallel." Report but do not over-claim.
- **UoS-like:** Assistant cosine to centroid no better than to a random persona. The Assistant is its own subspace. Unlikely given prior literature but worth checking.

**Outputs.**
- `outputs/gemma-2-27b-it/council/e1/summary.json`
- `outputs/gemma-2-27b-it/council/e1/fig_centroid_comparison.png` — per-trait bar chart of the three cosine quantities
- `outputs/gemma-2-27b-it/council/e1/weighted_mean_weights.json` (if optional step run)

**Expected runtime.** Minutes if Assistant vectors exist. ~1–2 GPU-hours if the Assistant-baseline pipeline needs to run (generation + activation extraction + vector computation).

---

## E2. Shared + specific decomposition (behavioral)

**Question.** Is the persona-specific residual behaviorally load-bearing?

**Script.** `pipeline/c2_shared_specific.py`

**Procedure.**
1. For each trait t:
   - Compute shared `u_t = mean_p(v_{p,t})` and specific residuals `w_{p,t} = v_{p,t} - u_t`.
   - Normalize all steering vectors to the same L2 magnitude `alpha` (match existing `eval_alpha2` convention where appropriate).
2. Three steering conditions per (persona, trait) pair:
   - **(a) Shared only:** steer with `u_t` while generating under persona p's system prompt.
   - **(b) Full (shared + correct specific):** steer with `v_{p,t} = u_t + w_{p,t}`.
   - **(c) Mismatched specific:** steer with `u_t + w_{p',t}` for a different persona p' (pick the furthest-away persona by baseline cosine).
3. Generate N responses per condition (start with N=20, scale if signal is weak). Use existing `8_steered_generation.py` infra.
4. Score each generation with two Claude-as-judge calls:
   - **Trait expression:** "On a scale of 0 to 1, how much does this response exhibit [trait]?"
   - **Persona coherence:** "On a scale of 0 to 1, how consistent is this response with a [persona description]?"
5. For each trait, report per-condition means and 95% CIs on both axes. Primary contrasts:
   - (b) − (a) on trait expression: does adding the specific residual increase trait signal?
   - (b) − (c) on persona coherence: does the *correct* residual preserve persona better than a mismatched one?

**Success criteria.**
- **Manifold view vindicated:** (b) > (a) on trait expression AND (b) > (c) on persona coherence, both with non-overlapping CIs on a majority of traits.
- **Weak LRH vindicated:** (a) ≈ (b) ≈ (c) on trait expression; the residuals contribute nothing measurable.
- **Residuals are signal but not persona-specific:** (b) ≈ (c) > (a). Interesting but doesn't support the persona-conditional story; reframe.

**Outputs.**
- `outputs/gemma-2-27b-it/council/e2/generations.jsonl` — all generations with condition labels
- `outputs/gemma-2-27b-it/council/e2/scored.jsonl` — with Claude judge scores
- `outputs/gemma-2-27b-it/council/e2/summary.json` — per-trait per-condition means + CIs + contrasts
- `outputs/gemma-2-27b-it/council/e2/fig_e2_contrasts.png` — forest plot of (b)−(a) and (b)−(c) per trait

**Expected runtime.** Dominant cost. 10 personas × 8 traits × 3 conditions × 20 generations = 4,800 generations. At gemma-2-27b-it speeds roughly 1–3 hours on a reasonable GPU setup. Claude judging adds Claude API cost (9,600 judge calls at 2 per generation) — budget a few hours of wall time and whatever that costs at current pricing.

**Cost control option.** Restrict to 3 traits spanning the expected difficulty range (e.g. assertiveness, empathy, honesty) and 5 personas for a pilot. Full sweep only if the pilot shows effect.

---

## E11. Persona-space extrapolation

**Question.** Is persona-space navigable by linear extrapolation?

**Script.** `pipeline/c11_persona_extrapolation.py`

**Procedure.**
1. Compute persona-difference vectors `d_{p→p'} = mu_{p'} - mu_p` from the persona baselines computed in E9. (These are persona-*identity* vectors, not trait vectors.)
2. Pick 3–5 persona pairs that span the space: e.g. Farmer→Politician, Therapist→Drill Sergeant, Kindergarten Teacher→Con Artist, Professor→Street Hustler.
3. For each pair (p, p'):
   - Under persona p's system prompt, generate responses to a fixed set of 20 neutral questions (e.g. a subset from an existing question file) in three conditions:
     - **Baseline:** no steering.
     - **Small extrapolation:** steer with `α · d_{p→p'}` at a small α (e.g. the magnitude you use for trait steering).
     - **Large extrapolation:** α scaled 2–3×.
4. Score each generation with Claude on three axes:
   - **Source-persona resemblance** (should decrease)
   - **Target-persona resemblance** (should increase)
   - **Coherence / fluency** (should remain high for manifold; drop for UoS)

**Success criteria.**
- **Manifold navigable:** small α produces a measurable shift toward p' with coherence preserved; large α continues the shift with some coherence degradation but nothing catastrophic.
- **Discrete / UoS:** coherence collapses before any meaningful persona shift appears; outputs become Franken-persona gibberish.
- **Intermediate:** partial persona shift with partial coherence loss. Consistent with a manifold that has meaningful curvature between these particular personas.

**Outputs.**
- `outputs/gemma-2-27b-it/council/e11/generations.jsonl`
- `outputs/gemma-2-27b-it/council/e11/scored.jsonl`
- `outputs/gemma-2-27b-it/council/e11/summary.json` — per-pair scores on the three axes at each α
- `outputs/gemma-2-27b-it/council/e11/fig_e11_extrapolation.png` — three-panel plot showing source/target/coherence vs α for each pair

**Expected runtime.** 5 pairs × 20 questions × 3 conditions = 300 generations. Small. Claude judging is 900 calls at 3 per generation. An hour or two end-to-end.

---

## Integrated sequencing

Suggested calendar-style ordering for a single concentrated push:

1. **Day 1 morning.** Write `persona_steering/council.py` helpers. Generate Assistant-baseline vectors if missing (kick off the pipeline, let it run in background).
2. **Day 1 afternoon.** Run E9. Inspect. Decision point: proceed or reframe.
3. **Day 1 evening → Day 2 morning.** Run E1 on both original and residualized vectors.
4. **Day 2.** Pilot E2 (3 traits × 5 personas × 3 conditions × 20 gens). Inspect the pilot before committing to the full sweep.
5. **Day 3.** Full E2 sweep if pilot shows effect. In parallel, run E11 (small, fits alongside).
6. **Day 4.** Write up findings into a single analysis doc. Decide whether E4, E5, E12, E13 follow.

## Risks and mitigations

- **E9 fails.** Pre-decide that if cos_resid is > 0.95 on most traits, the paper reframes as "existing trait vectors are dominated by persona baseline drift; here is how to disentangle them." Still a contribution, different story.
- **E2 shows no behavioral difference between (a) and (b).** Either the residuals are not load-bearing, or steering magnitudes were wrong, or the Claude judge is insensitive at this layer. Triage in order: sweep steering magnitude, try layer 30 and layer 15, inspect generations manually for whether trait differences are visible to a human.
- **E11 produces gibberish even at small α.** Check that `α` is calibrated to the same per-token activation shift magnitude as trait steering. If gibberish persists at very small α, persona-space is discrete and the UoS view is closer to right — that's itself a publishable finding.
- **Claude judge noise.** For E2 especially, noise in the judge can wash out a real effect. Pre-commit to running 3 independent judge passes per generation and taking the median, or use a rubric prompt that forces more structured output.

## What this first pass does and does not buy you

It buys you:
- A defensible answer to whether the manifold story survives the hardest boring explanation.
- Evidence on whether the Assistant is really a centroid.
- Behavioral evidence on whether persona-specific residuals matter.
- A first geometric test of whether persona-space is continuously navigable.

It does not buy you:
- A full curvature characterization (requires E4, E5).
- Cross-model universality (E12).
- Connection to SAE features (E13).
- Prompt-content confound control (E10) — recommended as a follow-up if E9 passes.

If all four experiments come back positive, the paper spine the council sketched becomes directly supportable and the remaining experiments are enrichment rather than defense.
