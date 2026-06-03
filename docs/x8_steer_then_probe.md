# X8 — Steer-then-probe

## Motivation

The project builds two kinds of directions in the residual stream:

- **Context directions** `u_C` — mean activation under persona C minus
  mean activation under neutral, built in `x3b`.
- **Trait directions** `v_T` — CAA-style mean activation under pro-trait
  instructions minus anti-trait instructions, one per trait × persona
  (including a `null` persona for the context-free version).

Both are used interchangeably as *steering vectors* (add α·v during a
forward pass) and as *probes* (cosine of h with the unit-norm direction).
X8 is a mechanistic sanity check that the two roles are consistent
— *does injecting α·v raise the probe cos(h, v̂)?* — and asks the
central project question: *does the persona-conditioned trait direction
`v_{T,C}` encode something different from the null trait direction
`v_{T,null}`?*

The experiment is not behavioural. No text is generated. It runs one
forward pass per prompt and inspects the residual stream at the position
where generation would begin.

## Method

### Inputs

For each `(context, trait)` pair we load three unit directions at the
steer layer L\* = 22:

| symbol | file | contrast |
|---|---|---|
| `û_C` | `directions/u_{context}.pt` | persona-in-context vs neutral |
| `v̂_{T,null}` | `caa_vectors/null_{trait}.pt` | pro-trait vs anti-trait under **no** persona |
| `v̂_{T,C}` | `caa_vectors/{context}_{trait}.pt` | pro-trait vs anti-trait under persona C |

Live vectors are kept at natural (un-normalised) scale for steering;
unit-norm versions are used only as probes.

### The forward pass

For each of N = 20 neutral user prompts:

1. Tokenise `[{role:"user", content:prompt}]` with
   `add_generation_prompt=True`. The last token of the input is the
   newline immediately after `<start_of_turn>model`.
2. Run the model once with an `ActivationSteering` hook at L\* adding
   `α·v_steer` to every token position of the residual stream.
3. Register extraction hooks on layers 15, 20, 22, 25, 30, 35, 40
   *inside* the steering context so they see the post-steer state.
   Each hook stores `h` at the last token position (position `-1`).
4. For each extraction layer L and each probe d̂ ∈ {û_C, v̂_{T,null},
   v̂_{T,C}} record `cos(h_L, d̂)` and per-prompt projections for
   AUROC.

No generation. The measurement is purely the geometry of the residual
stream at the moment the model would emit its first token.

### Experimental axes

Three *single-direction* steering conditions, each swept over
α ∈ {0, 0.25, 0.5, 1, 2, 4}:

- **ctx**: `v_steer = α · u_C`
- **trait**: `v_steer = α · v_{T,null}`
- **random**: `v_steer = α · ‖u_C‖ · r̂`

One *mix grid* over α_ctx × α_trait ∈ {0, 0.5, 1, 2}²:
`v_steer = α_ctx · u_C + α_trait · v_{T,null}`.

Every steering configuration is probed with all three directions at
every extraction layer.

## Direction geometry

Before any steering, the three unit directions have the following
pairwise cosines at L\*:

| pair | cos(u, v_null) | cos(u, v_C) | **cos(v_null, v_C)** |
|---|---:|---:|---:|
| therapist : empathy | +0.18 | +0.24 | **+0.90** |
| drill_sergeant : assertiveness | −0.14 | −0.19 | **+0.80** |
| con_artist : honesty | −0.18 | −0.09 | **+0.44** |

`cos(v_null, v_C)` is the key number. It quantifies how much the
persona-conditioned trait direction differs from the null trait
direction:

- therapist and drill sergeant have `v_C` almost parallel to `v_null`
  — the persona barely rotates the trait encoding.
- **con artist has `v_C` only 44 % aligned with `v_null`** — acting
  as a con artist meaningfully reshapes what "the honesty direction"
  means.

Baseline cos(h, probe) at α = 0 tells a complementary story — how
much probe energy is present in the unsteered residual stream on a
neutral prompt:

| pair | baseline cos_u | baseline cos_v_null | baseline cos_v_C |
|---|---:|---:|---:|
| therapist : empathy | +0.54 | +0.18 | **+0.37** |
| drill : assertiveness | −0.27 | +0.19 | **+0.38** |
| con_artist : honesty | −0.08 | +0.01 | **+0.23** |

In every pair the persona probe `v̂_{T,C}` fires more strongly on
neutral prompts than `v̂_{T,null}` does. For con artist this is
dramatic: **`v̂_{T,null}` sees essentially nothing (0.01) while
`v̂_{T,C}` sees 0.23**. The persona probe is picking up residual-stream
structure that the null probe is blind to.

## Results

### 1. Steering fires the matching probe, and quickly

At α = 1 (one natural norm of steering) the matching probe reaches
**AUROC = 1.00** at L\* for every pair (both probes saturate).
Matched-norm random steering shifts any probe's cosine by < 0.02 and
keeps AUROC near 0.5 up to α = 2. Probe responses are direction-
specific, not sensitive to generic perturbation energy.

### 2. The null vs persona probe behave differently under the same steering

**Under `u_C` (context) steering.** Both trait probes move in the
direction predicted by their cosine with `u_C` (see "off-axis leakage"
below). But the rate differs, and the difference tracks how aligned
`v_null` and `v_C` are.

For therapist and drill — where `cos(v_null, v_C) ≈ 0.8–0.9` — the
two probes' AUROC curves are near-identical; the persona probe is
marginally more responsive (e.g. at α = 1 for therapist,
AUROC_v_C = 0.94 vs AUROC_v_null = 0.86).

For con artist — where `cos(v_null, v_C) = 0.44` — the two probes
diverge meaningfully:

| α | AUROC_v_null | AUROC_v_C | cos_v_null | cos_v_C |
|---:|---:|---:|---:|---:|
| 0 | 0.50 | 0.50 | +0.01 | +0.23 |
| 1 | 0.10 | 0.29 | −0.00 | +0.22 |
| 4 | 0.00 | 0.03 | −0.04 | +0.19 |

Pushing toward the con-artist context direction drives `v̂_{T,null}`
strongly *below* baseline (AUROC → 0 means steered activations are
consistently less honest-looking than neutral, by the generic probe).
**`v̂_{T,C}` resists this push**: cos(h, v̂_{T,C}) drops from +0.23 to
+0.19 (a 17 % relative drop), compared with `v̂_{T,null}` going from
+0.01 to −0.04 (an absolute sign flip). The persona-conditioned
honesty direction is not the same object as the null honesty direction
even after a strong persona-context intervention — some of what
makes "honesty-under-con-artist" survives the context push, while
"generic honesty" does not.

**Under `v_{T,null}` (trait) steering.** At low α, the persona-probe
AUROC rises more slowly than the null-probe AUROC — because steering
with `v_null` only partially overlaps with `v_C`. At α = 0.25 for con
artist, AUROC_v_null = 0.98 but AUROC_v_C = 0.78. At higher α both
saturate because enough null-trait energy has been injected. So
**steering with `v_null` is a factor ~0.4–0.5 less efficient at
driving the `v̂_{T,C}` probe**, which is exactly what `cos(v_null, v_C)`
predicts.

### 3. Off-axis leakage still tracks cos(u, v)

Steering along `u_C` only, at α = 4:

| pair | cos(u, v_null) | Δcos_v_null | cos(u, v_C) | Δcos_v_C |
|---|---:|---:|---:|---:|
| therapist : empathy | +0.18 | +0.02 | +0.24 | +0.00 |
| drill : assertiveness | −0.14 | −0.04 | −0.19 | −0.05 |
| con_artist : honesty | −0.18 | −0.06 | −0.09 | −0.04 |

Sign and magnitude follow each probe's own cosine with `u_C`. The
con-artist case is again the most informative: `u_C` is *less*
anti-aligned with `v_C` (−0.09) than with `v_null` (−0.18), so `v_C`
is pushed down less by ctx steering.

### 4. The mix grid still decomposes near-additively at L\*

`cos(h, û_C)` depends almost entirely on α_ctx; `cos(h, v̂_{T,null})`
on α_trait. At L\* this is forced by linearity of the steering
operation, but the axes also remain approximately separable at L30–L40.

### 5. Effect propagates, `v_C` signal is more durable

Propagation for con_artist : honesty at α = 1, ctx steering:

| L | cos_v_null | cos_v_C | AUROC_v_null | AUROC_v_C |
|---:|---:|---:|---:|---:|
| 15 | +0.01 | +0.24 | 0.50 | 0.50 |
| 20 | +0.02 | +0.23 | 0.50 | 0.50 |
| 22 | −0.00 | +0.22 | 0.10 | 0.29 |
| 25 | −0.03 | +0.20 | 0.10 | 0.57 |
| 30 | −0.05 | +0.18 | 0.26 | 0.44 |
| 40 | −0.07 | +0.17 | 0.40 | 0.46 |

Two things stand out:

1. Pre-steer layers (15, 20) are unchanged from baseline for both
   probes — consistency check on hook ordering.
2. Post-steer, `cos_v_C` stays positive at all depths while `cos_v_null`
   is driven negative. The persona-conditioned probe sees sustained
   persona-honesty signal all the way through the stack; the null
   probe flips sign. At L25 `AUROC_v_C = 0.57` — slightly *above*
   baseline, meaning ctx-steered activations at L25 appear marginally
   *more* `v̂_{T,C}`-aligned than neutral prompts.

## What this shows

1. **`v_{T,C}` is genuinely distinct from `v_{T,null}`**, and how
   distinct depends on the persona–trait combination. Therapist–empathy
   and drill–assertiveness have `cos(v_null, v_C) ≈ 0.8–0.9` — almost
   the same direction. Con-artist–honesty has `cos(v_null, v_C) = 0.44`
   — genuinely different, consistent with the "inverted honesty"
   semantic story.

2. **The persona probe reads more natural structure from the residual
   stream.** On neutral prompts, baseline `cos(h, v̂_{T,C})` is 2× higher
   than `cos(h, v̂_{T,null})` for therapist and drill, and **20× higher
   for con artist** (0.23 vs 0.01). The null probe is near-blind to
   activation structure that the persona probe picks up.

3. **The persona probe resists context-direction steering.** Under
   strong `u_C` steering, the null probe is driven below baseline on
   the anti-aligned pairs, while the persona probe shrinks
   proportionally less. This supports a reading where `v_{T,C}` carves
   out some component of "how this persona does this trait" that isn't
   just "pushed persona ⊕ generic trait."

4. **Steering efficiency is predicted by cos(v_null, v_C).** Under
   `v_{T,null}` steering, AUROC on the persona probe lags AUROC on the
   null probe at low α in direct proportion to how far apart the two
   directions are. This is a cheap empirical way to quantify "how much
   a persona-conditioned probe differs from its null counterpart."

## What this does not show

- Anything behavioural. Whether steered generation reads as more
  honest / assertive / empathetic to a judge is a separate experiment
  (main pipeline step 1 + `evaluation.py`).
- Whether `v̂_{T,C}` is the *right* probe to use in downstream
  evaluation — just that it differs from `v̂_{T,null}` and differs in
  a direction consistent with persona semantics.

## Reproducing

```bash
python pipeline/x8_steer_then_probe.py \
    --model google/gemma-2-27b-it \
    --directions-dir outputs/gemma-2-27b-it/v2/causal_pilot/directions \
    --vectors-dir outputs/gemma-2-27b-it/v2/caa_vectors \
    --output-dir outputs/gemma-2-27b-it/v2/steer_probe \
    --pairs therapist:empathy drill_sergeant:assertiveness con_artist:honesty \
    --n-prompts 20

python pipeline/x8b_steer_then_probe_plot.py \
    --summary outputs/gemma-2-27b-it/v2/steer_probe/summary.json
```

Figures: `x8_single_sweep.png`, `x8_mix_grid.png`,
`x8_layer_propagation.png`, `x8_null_vs_persona.png`.
