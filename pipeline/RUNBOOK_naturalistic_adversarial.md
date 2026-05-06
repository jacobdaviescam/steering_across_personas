# Runbook — naturalistic + adversarial Fig 3 + persona-residue classifier

Three new experiment threads for the four-pager. Run in order shown; later
steps depend on earlier outputs.

Set this once for the whole session:

```bash
export OPENROUTER_API_KEY=sk-or-...
MODEL=google/gemma-2-27b-it
ROOT=outputs/gemma-2-27b-it/v2
```

---

## Thread 1 — Naturalistic-response Fig 3 v2

**Goal:** the version of Fig 3 that actually tests the deployment-monitoring
claim — null-trained probe applied to free-form responses, judged by Claude
for trait expression, scattered against distance-from-null per cell.

```bash
# 1. Generate ~50 responses per (persona, trait) under persona prompt only
python pipeline/n1_naturalistic_generate.py \
    --model $MODEL \
    --output-dir $ROOT/naturalistic/responses \
    --n-questions 10

# 2. Extract layer-22 activations from the generated responses
python pipeline/2_activations.py \
    --model $MODEL \
    --responses-dir $ROOT/naturalistic/responses \
    --output-dir    $ROOT/naturalistic/activations

# 3. Score each response with Claude via OpenRouter
#    ~4000 calls, ~$15-25 at Sonnet 4.5 rates
python pipeline/n3_naturalistic_judge.py \
    --responses-dir $ROOT/naturalistic/responses \
    --output-dir    $ROOT/naturalistic/judged \
    --max-workers 8

# 4. Build the new Fig 3 (per-cell |r| and AUROC vs distance from null)
python pipeline/n4_naturalistic_eval.py \
    --judged-dir       $ROOT/naturalistic/judged \
    --activations-dir  $ROOT/naturalistic/activations \
    --probes-dir       $ROOT/caa_probes/probes_pkl \
    --vectors-dir      $ROOT/caa_vectors \
    --output-dir       $ROOT/naturalistic/figures
```

Output figure: `$ROOT/naturalistic/figures/n4_naturalistic_fig3.pdf`. Two
panels: |r| (probe vs judge) and AUROC (probe vs binarised judge), both
against `1 - cos(v_T,c, v_T,null)`.

---

## Thread 2 — Adversarial cells

**Goal:** confirm the null probe degrades hardest in cells where the
persona's natural answer disagrees with the trait label.

```bash
# 1. Have Claude write 10 adversarial questions per (persona, trait)
#    ~80 calls, ~$2
python pipeline/a1_generate_adversarial_questions.py \
    --output-dir data/prompts/adversarial \
    --n 10

# 2. Generate model responses to those questions
python pipeline/a2_adversarial_generate.py \
    --model $MODEL \
    --questions-dir data/prompts/adversarial \
    --output-dir    $ROOT/adversarial/responses

# 3. Extract layer-22 activations
python pipeline/2_activations.py \
    --model $MODEL \
    --responses-dir $ROOT/adversarial/responses \
    --output-dir    $ROOT/adversarial/activations

# 4. Judge
python pipeline/n3_naturalistic_judge.py \
    --responses-dir $ROOT/adversarial/responses \
    --output-dir    $ROOT/adversarial/judged \
    --max-workers 8

# 5. Paired-AUROC scatter (naturalistic vs adversarial per cell)
python pipeline/a3_adversarial_analysis.py \
    --naturalistic-judged $ROOT/naturalistic/judged \
    --naturalistic-acts   $ROOT/naturalistic/activations \
    --adversarial-judged  $ROOT/adversarial/judged \
    --adversarial-acts    $ROOT/adversarial/activations \
    --probes-dir          $ROOT/caa_probes/probes_pkl \
    --vectors-dir         $ROOT/caa_vectors \
    --output-dir          $ROOT/adversarial/figures
```

Output figure: `$ROOT/adversarial/figures/a3_paired_auroc.pdf`. Points
below the diagonal = the cell where persona-natural answers contradict the
trait label is exactly where the null probe loses calibration.

---

## Thread 3 — Persona residue (classifier on existing steered responses)

**Goal:** when the model is system-prompted as Farmer but steered with
Politician's honesty vector, does the response classify as more
politician-like than self-steered Farmer? Reuses existing alpha=2 data —
nothing to regenerate.

```bash
python pipeline/p1_classifier_on_steered.py \
    --steered-dir   outputs/gemma-2-27b-it/steered_responses_alpha2 \
    --classifier-dir $ROOT/classifier \
    --output-dir     $ROOT/persona_residue
```

Outputs:
- `$ROOT/persona_residue/p1_residue_scatter.pdf` — per (src≠tgt) cell:
  Δ P(source | self) vs Δ P(source | cross). Self-steering should leave
  P(source) flat; cross-steering should bump it.
- `$ROOT/persona_residue/p1_residue_per_trait.pdf` — mean Δ P(source) under
  cross-steering, per trait. Should rank-correlate with Result-1 spread.
- `$ROOT/persona_residue/cell_summary.json` — full per-cell numbers.
- `$ROOT/persona_residue/residue_pairs.json` — per (src, tgt, trait) row.

This is the cheapest of the three — no GPU needed, just SBERT inference on
~88k responses. ~10 min on CPU, faster on GPU.

---

## Cost / time estimate

| Thread | GPU | Claude (OpenRouter) | Wall-clock |
|---|---|---|---|
| 1 (naturalistic) | ~1 h (vLLM gen + activation extract) | ~$15-25 | ~2 h |
| 2 (adversarial)  | ~30 min | ~$5 | ~1 h |
| 3 (persona residue) | ~10 min CPU (SBERT) | $0 | ~10 min |

Thread 3 produces a paper-ready figure today; Threads 1+2 need GPU access.
