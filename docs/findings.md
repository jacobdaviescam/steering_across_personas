# Findings Log — Context-Dependent Trait Representations (v2 / causal-figures)

Running log of what experiments have revealed and the open methodological questions they've raised. Intended to stabilise framing as we build the paper.

---

## Top-line framing (current)

> **Trait directions are geometrically context-specific for all measured traits, but naive linear probes trained on single-context CAA data fail to recover the per-context structure for traits whose common-mode signal dominates. Either (a) residualizing against the common mode or (b) training on richer free-form data is needed to reveal context-specific detectability across all traits.**

This replaces the earlier "within-context probes outperform null probes" claim, which held for only 3 of 8 traits when tested naively on IV activations.

---

## Experimental evidence so far

### 1. Geometry: trait directions differ by context (consistent signal)
- Per-trait pairwise cosine between persona trait vectors: **mean 0.64–0.77**, min in some cases negative (deference, assertiveness, warmth).
- Clear clusters in honesty (identity-honest vs strategic-honest personas) and empathy (warm vs gruff personas).
- Visible in `figures_vectors/caa_vectors_cosine_*.pdf`.

### 2. Behavioural classifier: context-sensitivity varies by trait
- Per-trait SBERT classifier accuracy: empathy 0.34 (lowest), risk-taking 0.64 (highest). Chance = 0.083.
- Overall 0.54 — strongly above chance for all traits.
- Supports geometric finding: traits where expression varies more across personas (empathy) are classified better.

### 3. Test 1 (IV transfer of CAA probes): mixed per-trait result
Applied CAA-trained A (null-only) and within-context probes to free-form IV activations.

| Trait | A_mean | W_mean | Δ | cells W > A |
|---|---|---|---|---|
| empathy | 0.74 | 0.84 | **+0.09** | 7/12 |
| confidence | 0.66 | 0.74 | **+0.08** | 10/12 |
| impulsivity | 0.62 | 0.71 | **+0.09** | 10/12 |
| risk_taking | 0.94 | 0.92 | −0.03 | 4/12 |
| warmth | 0.94 | 0.91 | −0.04 | 4/12 |
| deference | 0.63 | 0.59 | −0.05 | 1/12 |
| assertiveness | 0.90 | 0.85 | −0.05 | 3/12 |
| honesty | 0.82 | 0.75 | −0.08 | 0/12 |

**Three traits support the hypothesis (empathy, confidence, impulsivity); five don't.**

### 4. X3c causal sweep: probe divergence fails broadly
Under context-direction steering (targeting null persona system prompt), both null and within probes degrade together across traits. Only 4 of 26 completed (trait, ctx) cells show within > null at α=4.

Pattern is likely driven by:
- **Distribution mismatch**: within probes trained on (persona system prompt + CAA A/B) tested on (null system prompt + eliciting prompts + steering) — three-way domain shift
- **Mode collapse at α≥4**: inspection shows generations lose pos/neg distinction, becoming generic context-flavoured loops regardless of prompt content

Behavioural signal (P(context) rise under steering) is robust: mean +0.52 from α=0 to α=4.

### 5. X2 probe regimes: saturation on both CAA and IV
- X2 on CAA: A/B/B-parity all hit ~1.0 AUROC → trivially separable at answer-token
- X2 on IV: A/B/B-parity all hit ~0.99 AUROC → trivially separable in free-form under instruction
- Probes trained on pos-instruction vs neg-instruction always find the universal valence axis

---

## Core methodological hypothesis

A linear probe trained on one persona's ~100 pos + ~100 neg activations learns a direction that best separates pos from neg in that small training set. **The direction with the strongest signal is the common-mode trait-valence axis shared across all personas.**

Persona-specific components exist in the trait vectors (visible geometrically — see §1) but are smaller in magnitude and noisier, so the linear probe procedure defaults to the common-mode direction for most traits.

**Consequence**: within probes trained on persona X's data learn roughly "universal trait axis + noise," not "persona X's specific trait axis." They perform similarly to (or slightly worse than) null probes that also learn the common-mode axis.

For **empathy, confidence, impulsivity**, the persona-specific component is large enough to survive this filtering. For **honesty, warmth, assertiveness, risk_taking, deference**, it isn't — the common-mode dominates and within probes lose to null.

---

## Experimental designs under consideration

1. **Residual probes (next experiment)**. Remove the common-mode direction from activations before probe training. Force within probes to use only the persona-specific residual. Test whether this recovers within > null for the "universal" traits (honesty, etc.).

2. **IV-trained probes with within-persona holdout**. Instead of CAA-trained probes, train probes on ~80% of a persona's IV activations and test on the held-out 20%. Free-form activations encode more persona-specific trait expression than answer-token CAA activations.

3. **Question-matched probes**. Generate the same neutral question under all 10 personas. Use LLM judge to score each response for trait expression. Train probes on judge scores as labels (continuous or thresholded). Cleaner separation of trait expression from prompt content.

4. **X3c with persona system prompt** (not null). Would give within probes matching training distribution, but changes the causal interpretation of steering.

---

## Open questions

- Does the residual probe recover context-specificity for "universal" traits like honesty?
- Is the pos/neg axis in CAA activations essentially 1-dimensional (single universal axis + tiny per-persona residual), or are there multiple orthogonal per-persona directions that probes could learn with better procedures?
- For the X3c causal sweep, is there a target-persona-system-prompt design that both isolates the causal effect of steering AND gives within probes a fair distribution match?

---

## Update — 5 weird personas added, geometric analysis on 15-persona set

Added pathological_liar, six_year_old, sociopath, contrarian_deceiver, actor_in_rehearsal. Ran per-trait cosine heatmaps, PCA, Isomap, null-projection, residual-space PCA.

**Headline**: mainstream personas already contain the strongest OOD cases at the activation level.

- **con_artist is in the top-3 outlier set for 6 of 8 traits** in Isomap space — more frequently than any engineered weird persona. It's the only persona with a NEGATIVE projection onto null's assertiveness direction (−0.05×||null||).
- **drill_sergeant is the top outlier on empathy, warmth, deference** in null-projection space (0.05–0.21 × ||null||, well below the mainstream cluster at 0.5–0.9).
- **contrarian_deceiver is a strong outlier on honesty/confidence/warmth/assertiveness** (appears in Isomap top-3 for 4 traits). Its literal speech-act inversion produces activation-level inversion.
- **six_year_old has one clear signature**: impulsivity (0.08 × ||null|| — lowest of all personas; a child's "positive impulsivity" direction is genuinely different from an adult's).
- **pathological_liar, sociopath, actor_in_rehearsal don't appear as Isomap outliers**. Their weirdness is entirely content-level; at the activation-vector level they sit inside the mainstream cluster.

**Implication**: content-level weirdness (what the persona says) does NOT automatically produce activation-level weirdness (how the model represents "being trait-positive"). Only structurally-inverted personas (contrarian) or genuinely distinct cognitive/developmental contexts (six-year-old on impulsivity) produce activation-level deviance.

**Residual-space PCA surprise**: for 7 of 8 traits, weird personas sit CLOSER to the centroid of residual space than mainstream personas do (ratio 0.38–0.87). Only honesty flips. Mainstream personas drive the residual variance; weird personas don't.

**Paper framing update**: the central claim can now rest on mainstream personas themselves. "Even among familiar, well-defined personas, the linear direction for a trait varies substantially from the null direction — con artist's assertiveness direction is anti-aligned with null's, drill sergeant's warmth direction is barely aligned." This is a stronger, more defensible claim than "weird personas break probes" because reviewers won't dismiss our chosen OOD cases as cherry-picked.

Weird personas become a robustness check in the appendix. Contrarian_deceiver and six_year_old (on impulsivity) remain useful as illustrations.

---

## Artefacts

- Cosine heatmaps: `outputs/gemma-2-27b-it/v2/figures_vectors/caa_vectors_cosine_*.pdf`
- Test 1 IV transfer: `outputs/gemma-2-27b-it/v2/caa_probes/iv_transfer_test.json`, figure `figures_vectors/iv_transfer_test.pdf`
- X2 (CAA) results: `outputs/gemma-2-27b-it/v2/caa_probes/metrics.json`
- X2 (IV) results: `outputs/gemma-2-27b-it/v2/iv_probes/metrics.json`
- X3c main run: in progress, `outputs/gemma-2-27b-it/v2/causal_main/`
- Paper figures: `outputs/gemma-2-27b-it/v2/figures_paper/fig1{a,b,c}_*`
