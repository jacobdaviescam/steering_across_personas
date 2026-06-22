# Mentee Project Catalogue

Nine scoped projects extending the paper (*persona-conditional trait representations*,
[icml2026/main.tex](../icml2026/main.tex)). Each was scoped against the repo's actual
state and adversarially reviewed for overlap with completed work and feasibility for a
junior researcher at 4–10 weeks part-time. Two tracks:

- **Robustness** — is the result what it seems? (targets the paper's stated limitations
  and the open problems in [findings.md](findings.md))
- **Extension** — what can the result *do*? (SOTA-improvement and safety-measurement
  projects)

Every project starts with a **CPU-only on-ramp** from the HF dataset
([girishgupta/persona-steering-activations](https://huggingface.co/datasets/girishgupta/persona-steering-activations)):
reproduce a headline number from cached activations before touching a GPU.

## Prerequisites (mentor tasks, before assignment)

These came up repeatedly in review — several projects silently depend on artifacts that
exist only on the RunPod volume, not on HF:

1. **Upload the v2 naturalistic artifacts to HF**: judged free-form responses, response
   activations, probe pickles, x5 transfer matrices, splits. Blocks the on-ramps of
   *probe-hardening*, *ensembles*, and *failure-predictor*.
2. **Upload OLMo stage vectors (and activations if findable) to HF** — blocks
   *training-origin*'s bootstrap CIs.
3. **Resolve the judge-API gap**: `evaluation.py` calls `anthropic.Anthropic()` directly
   but pods only have an OpenRouter key ([PROJECT_STATUS.md](../PROJECT_STATUS.md)).
   Needed by any project that re-runs judges on a pod.

## Dependency map

- **P1 cross-model** produces the Gemma-3 v2 extraction that **P9 SAE-anatomy** and the
  second-model arms of **P6 failure-predictor** reuse. Start P1 early.
- **P4 probe-hardening** sharpens the ground truths that **P5 ensembles** and
  **P6 failure-predictor** benchmark against; they can run in parallel but should share
  the residualized-probe code.

---

## Track A — Robustness

### P1 · Minimal decisive cross-model replication — `intermediate`, priority 8/10

**One-liner:** Upgrade the v1 cross-model hints to paper-grade v2 evidence: do the three
core claims (geometric conditioning vs controls, steering residue, per-trait ordering)
hold on Gemma-3-27B-IT — and does the trait ordering *change*, as the post-training
hypothesis predicts?

**Why:** "Single model family" is the paper's first stated limitation and the most
predictable reviewer objection. Existing Gemma-3 numbers ([results/summary.md](results/summary.md))
are v1 (trait-mentioning persona prompts, pre-CAA pipeline) and the residue test has
*never* run on another model. The v1 hint that honesty flips from most-stable (0.896) to
least-stable (0.635) directly bears on the post-training-emphasis hypothesis.

**Scoped plan** (reviewer-adjusted: one primary arm, one reduced secondary):
1. *CPU on-ramp:* reproduce the σ-spread figure and residue-classifier accuracy from HF
   Gemma-2 data; write the 1–2 page "replication battery" doc (8 contexts × 8 traits).
2. *Pre-flight (~2 GPU-h):* Gemma-3 vLLM load (multimodal wrapper), `ProbingModel`
   layer paths, layer choice via `e1_layer_sweep.py` — the exact checklist in
   [PROJECT_STATUS.md](../PROJECT_STATUS.md). Standalone PR.
3. *Primary arm (~40–50 GPU-h):* v2 CAA+IV extraction on Gemma-3; claim-1 battery
   (`r3` phrasing control — v1 failed this at p=0.349, v2 must settle it; nonsense
   control; bootstrap floor); claim-2 residue run (`8_steered_generation.py` at α=2 +
   SBERT classifier).
4. *Secondary arm (reduced):* OLMo-2-7B-Instruct residue-classifier-only + v2
   re-extraction sanity check (geometry already in the paper appendix).
5. *Writeup:* 3-models × claims replication grid + cross-model trait-ordering table;
   supersede the v1 numbers in `docs/results/summary.md`; upload vectors to HF.

**Compute:** ~$120–250 GPU; negligible API. **Relation to issues:** supersedes #27,
absorbs the scoped core of #28.
**Null is fine:** if Gemma-3 conflates persona with phrasing even on v2 prompts, that
bounds the claim's generality — reportable either way.

### P2 · Persona-induction robustness (few-shot core, LoRA stretch) — `advanced`, priority 8/10

**One-liner:** Is the geometric effect a property of the *persona* or of the *system
prompt*? Induce the same personas via few-shot examples (core) and LoRA adapters
(stretch) and test whether same-persona cross-induction cosine beats the across-persona
floor (~0.78).

**Why:** Second stated limitation of the paper. Connects to the emergent-misalignment
literature: if weight-level personas shift trait vectors *more* than prompted ones,
default-trained probes are least reliable exactly where fine-tuning moves models.
Nothing in the repo tests this (verified: only `10_oracle.py` touches PEFT, load-only).

**Scoped plan** (reviewer-adjusted: few-shot is the deliverable, LoRA is stretch):
1. CPU on-ramp: reproduce baseline geometry (across-persona ~0.78, paraphrase ~0.85,
   bootstrap ~0.99) from HF data.
2. Expand `few_shot_examples` in persona YAMLs to ~8/persona (Claude-generated,
   fidelity-gated with the existing SBERT 12-way classifier — zero API cost).
3. Few-shot CAA extraction 10×8 (~25–35 GPU-h) + three-way cosine analysis with
   bootstrap CIs.
4. *Stretch:* QLoRA personas for 2 personas (con_artist outlier + farmer anchor),
   de-risked first on Gemma-2-9B; extraction under null system prompt (persona lives in
   weights only); null-probe AUROC on each induction mode.

**Compute:** ~70–110 GPU-h full plan; few-shot-only ~35 GPU-h.
**Null is fine:** "trait geometry is prompt-bound" would itself be a notable,
mitigation-relevant finding.

### P3 · Layer pervasiveness + coherence-controlled steering — `advanced`, priority 8/10

**One-liner:** Map persona-conditioning across all 46 layers from the existing HF
activations (pure CPU), then fix the α≥4 mode-collapse confound with a norm-matched,
LLM-judged coherence gate and rerun the causal probe-degradation sweep under it.

**Why:** All paper results sit at layer 22 (coarse 10-layer sweep in the appendix only),
and [findings.md](findings.md) documents that generations degenerate at α≥4 — exactly
where the headline AUROC drop (0.83→0.59) is largest. No steering paper has published a
full-depth persona-conditioning profile with paraphrase/nonsense control bands.

**Scoped plan** (reviewer-adjusted: core/stretch split):
1. *Core, CPU:* 46-layer sweep of per-trait cross-persona cosine + shared variance with
   control bands at every layer (loop `4_analysis.py`; activations have all layers);
   localization statistic (depth interval holding ≥50% of above-control variance);
   per-layer probe transfer in the one non-saturated regime (CAA→IV).
2. *Core, GPU (~15–25 h):* productionize the `coherence_judge()` already sketched in
   `x3c_causal_sweep.py`; per-cell max-coherent-α; rerun steer-then-probe at layer 22
   gated to coherent α only — does 0.83→0.59 survive?
3. *Stretch:* repeat gated sweep at 2–3 other layers.

**Compute:** core CPU + ~15–25 GPU-h.
**Null is fine:** "the AUROC drop is a coherence artifact" is precisely the audit the
limitations section calls for, and a flat depth profile validates the single-layer
methodology.

### P4 · Hardening (or refuting) the probe-degradation claim — `intermediate`, priority 8/10

**One-liner:** The paper's probe section is explicitly preliminary and
[findings.md](findings.md) documents three open problems — valence-axis saturation,
three-way domain shift, non-monotonic distance bins. Fix all three; end with a verdict:
hardened, per-trait-conditional, or refuted.

**Scoped plan** (reviewer-adjusted):
0. *Week-0 audit:* check which naturalistic artifacts are actually on HF vs pod-only
   (see Prerequisites) — this gates the on-ramp.
1. CPU: reproduce the x2 saturation result and per-trait within-vs-null table.
2. CPU: **residualized probes** (project out the pooled valence axis before training) —
   the design findings.md proposes but never ran — as residualized IV-holdout probes.
3. CPU: replace the binned |r| table with a continuous mixed-effects analysis +
   per-trait permutation tests + judge-noise ceiling (judge self-agreement).
4. **Pre-register pass/fail criteria** before GPU spend.
5. GPU (~20–30 h): domain-shift-controlled `x3c` rerun — probes trained on α=0
   eval-distribution activations, norm-matched random + trait-direction controls.

**Compute:** mostly CPU; one bounded GPU phase.
**Null is fine:** "probe degradation was a distribution-mismatch artifact" is a clean,
publishable negative that the paper's own framing invites. Best case, the weakest
section becomes the strongest.

---

## Track B — Extensions

### P5 · Persona-conditional ensembles & routing (the paper's mitigation, built) — `advanced`, priority 8/10

**One-liner:** Build the probe/steering-vector family + inference-time router the
Discussion proposes but never tests; benchmark against null-probe, pooled-average, and
within-context oracle on held-out personas. This is the **beat-the-single-extraction-SOTA**
project.

**Scoped plan** (reviewer-adjusted: endpoints moved to where degradation actually exists):
1. CPU week 1: routing module (hard nearest-centroid via `x3b` context directions, soft
   ensemble, uniform average, oracle) + a ~1-week LOPO sanity check on instruction
   activations with the *pre-registered expectation of a near-null* (within-vs-cross gap
   there is ~0.02).
2. **Primary endpoints from week 3:** (a) naturalistic free-form |r|-by-distance-bin —
   does routing recover the far-bin from ~0.40 toward the close-bin 0.59? (b) causal
   α-sweep — does the routed probe hold near 0.83 where the null probe falls to 0.59?
3. Steering-vector routing under held-out personas (null vs average vs routed vs
   matched-norm control), trait-judge scored.
4. Report **fraction of oracle–null gap closed**, incl. on the 5 extension personas
   never in the repertoire.

**Compute:** CPU for routing/benchmark; ~30–50 GPU-h + judge spend for naturalistic and
causal endpoints (needs Prerequisite 1, else regenerate).
**Risk worth knowing:** common-mode dominance may cap the oracle–null gap for 5/8 traits
(coordinate with P4's residualized probes).

### P6 · A calibrated pre-deployment "probe health check" — `intermediate`, priority 8/10

**One-liner:** Turn σ_T(c) from a suggestion in the Discussion into a validated,
calibrated predictor of probe failure — benchmarked against cheaper label-free
statistics — and ship it as a CLI tool. This is the **measure-something-useful-for-safety**
project.

**Why:** The paper ends: "evaluate which measures are predictive of probe and steering
vector failure." The correlational core exists (`x6`: distance-vs-transfer r≈−0.12 to
−0.24; naturalistic ρ≈−0.39) but nothing validates a *decision rule* on held-out
contexts.

**Scoped plan** (reviewer-adjusted):
0. Prerequisite 1 (artifact upload) or rewrite on-ramp to retrain probes via `x2`.
1. CPU: predictor battery per cell — σ_T(c), magnitude ratio, persona-direction norm,
   activation distance, label-free probe-score KS shift (`h1_health_statistics.py`).
2. CPU: two ground truths (probe-judge |r|; CAA→IV transfer AUROC — drop saturated
   IV→IV), failure labels defined *within-trait* relative to the trait's own ceiling.
3. Leave-one-persona-out + leave-one-trait-out predictor comparison with bootstrap CIs;
   freeze a threshold at a stated operating point (e.g. ≤10% false-trust).
4. Evaluate frozen threshold on ≥30 extension-persona cells never used in calibration
   (~10–20 GPU-h for their free-form generations + judge).
5. Ship `persona_steering/health_check.py` CLI; *stretch:* second-model validation on
   P1's Gemma-3 data.

**Null is fine:** "no cheap statistic predicts failure" directly answers the paper's
closing question and warns against σ-based sanity checks.

### P7 · Realistic deployment contexts — `intermediate`, priority 8/10

**One-liner:** Swap theatrical archetypes for ~12 realistic contexts — coding-agent /
computer-use scaffolds, jailbreak & companion roleplay, non-English, long multi-turn —
and produce a ranked deployment-risk table: which real contexts move trait
representations furthest, and does probe accuracy fall where σ says it should?

**Why:** The archetypes were a method-validation set. The deployment contexts that
motivate monitoring (the paper's own Discussion point) are agentic and adversarial ones.
A DAN-prompt row sitting below the farmer row in σ would be a directly safety-relevant,
headline-able result.

**Scoped plan** (reviewer-adjusted: probe eval on free-form, *not* CAA-format):
1. CPU on-ramp: reproduce the per-trait σ figure for the 17 existing personas.
2. Author the battery as `data/personas/` YAMLs (5 paraphrase variants each, 4 families);
   refusal-rate gate for jailbreak prompts; language-matched null baseline for
   non-English.
3. CAA extraction battery × 8 traits (~40–80 GPU-h, forward passes only) + σ with
   bootstrap CIs and paraphrase-vs-cross check per context.
4. **Probe eval on free-form generations** (n-series runbook: generate → judge → probe)
   — the CAA-format regime is documented to saturate and must not be used.
5. Multi-turn drift at depths 1/5/15/30 for 2 contexts (needs new prefix-assembly on
   top of `2c`; demoted to stretch if heavy).
6. Deployment-risk ranking figure + σ-vs-AUROC-drop correlation.

**Relation to issues:** complements #25 (safety traits × realistic contexts is the
natural follow-on). **Null is fine:** "realistic contexts shift less than archetypes"
bounds deployment risk — exactly what a safety team wants to know.

### P8 · Training-stage origin & a targeted-SFT mitigation test — `advanced`, priority 7/10

**One-liner:** The per-trait OLMo emergence figure already exists — harden it (bootstrap
CIs, cross-family Spearman vs Gemma's ordering), test the supervision-dose hypothesis
against the open tulu-3 SFT mixture, and run the first deliberate **trait-stabilization
training intervention**.

**Scoped plan** (reviewer-adjusted: intervention is the core):
0. Week-0: locate OLMo stage activations on the pod; upload vectors (+activations if
   found) to HF; hard go/no-go on intermediate SFT checkpoints (else weight
   interpolation, clearly labelled).
1. CPU (~1 wk): harden the existing `variance_trajectory` figure; Spearman of OLMo vs
   Gemma per-trait orderings.
2. CPU+API: per-trait supervision frequency in `tulu-3-sft-mixture` (lexicons validated
   on ~500 Claude-judged examples) regressed on per-trait stability.
3. **Core:** trait-consistency SFT data via `0_generate_data.py` extensions → LoRA-SFT
   OLMo-2-7B-SFT → re-run t1–t3 → Δρ on target trait vs ≥3 control traits, with
   persona-separability and capability guard metrics.

**Compute:** ~$30–80 API + bounded 7B fine-tuning GPU.
**What it could show:** post-training emphasis *causes* trait stability — and that you
can deliberately train a trait to be persona-stable, a training-side mitigation no one
has demonstrated.

### P9 · Feature-level anatomy: SAE decomposition & residue ablation — `advanced`, priority 8/10 *(gap-critic addition)*

**One-liner:** Every other project treats trait vectors as opaque directions. Use the
IT-matched **Gemma-Scope-2 SAE on Gemma-3-27B-IT** (layer 31) to decompose
v_T(c) = shared-trait features + persona-identity features + noise — then causally test
whether the cross-steering "persona residue" disappears when persona-identity features
are ablated from the vector before injection.

**Why:** The repo is unusually ready: `e5`/`e7` scripts and the Gemma-Scope-2-aware
`sae_loader.py` exist, and [PROJECT_STATUS.md](../PROJECT_STATUS.md) explicitly
quarantines the old E7 results because no IT-matched SAE exists for Gemma-2 — Gemma-3
removes that blocker. Best case: "persona residue is removable feature mass" — a
mechanism *and* a mitigation in one result.

**Plan sketch:** (1) CPU on-ramp: rerun E5/E7 on Gemma-2 through the base SAE,
exploratory; (2) decompose Gemma-3 CAA cells (reuse P1's extraction; else ~6 personas ×
4 traits reduced battery) into shared/persona/idiosyncratic feature mass; test whether
persona-feature mass predicts σ_T(c); (3) purified-vector cross-steering + the existing
SBERT residue classifier: does residue accuracy drop to chance while trait lift
survives?; (4) writeup + validated replacement for the quarantined E7.

**Depends on:** P1's Gemma-3 activations (becomes mostly CPU if P1 lands first).
**Null is fine:** "residue is not linearly feature-separable" constrains mitigation
hopes — publishable.

---

## Suggested assignment

| If the mentee is… | Start with |
|---|---|
| Strong engineer, new to interp | **P1** (cross-model) or **P7** (realistic contexts) |
| Stats-inclined, careful | **P4** (probe hardening) or **P6** (health check) |
| Wants a buildable artifact | **P5** (ensembles) or **P6** (health check) |
| Interested in training dynamics | **P8** (training origin) |
| Interested in mech interp | **P9** (SAE anatomy, after P1 starts) |
| Comfortable with fine-tuning | **P2** (induction modes) |

Two mentees pair well on **P1 + P9** (shared Gemma-3 extraction) and on **P4 + P5/P6**
(shared probe ground truths).
