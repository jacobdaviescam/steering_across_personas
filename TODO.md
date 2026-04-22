# ICML 2026 Sprint TODO — due 2026-04-15

Single-day push. Ruthless prioritization: everything here serves filling the `\todo{}` slots in `icml2026/main.tex` or the nearest defensible version of the story.

Branch: `experiments/paper-hardening`. Paper lives in `icml2026/main.tex` (not `example_paper.tex` — that's still the template).

---

## P0 — Must land today

### 1. Pull pod outputs to local
Colleague + my GPU-pod E-series runs live on the pod's network volume. Pull everything into `outputs/gemma-2-27b-it/experiments/` so figure-making and paper writing can happen locally.

- [ ] rsync pod `outputs/gemma-2-27b-it/experiments/` → local
- [ ] rsync pod `outputs/gemma-2-27b-it/council/` if any council runs finished there
- [ ] Confirm which of E2–E7 actually have outputs on pod (list them here):
  - E2 bootstrap CIs: ?
  - E3 IV–CAA decomposition: ?
  - E4 probe transfer: ?
  - E5 SAE features: ?
  - E6 basin geometry (steps 4/5/6): ?
  - E7 sparse codes: ?

### 2. Council scripts (C9 gate → C1 → C11; C2 pilot only)
Cheap, decisive, and they unlock the reframed story in `brainstorm.md`.

- [ ] `pipeline/c9_residualization.py --layer 22` — **gate**. Inspect `summary.json` and the bar chart. If `cos_resid_t > 0.95` for most traits, reframe the paper (see fallback below) — don't continue to C1/C11 under the current framing.
- [ ] `pipeline/c1_assistant_centroid.py --layer 22` — needs an Assistant-baseline vector. Check if `outputs/.../vectors/default_*.pt` already suffices (slug `default` is in PERSONA_SLUGS); if not, generate it via the pipeline on the pod before running C1.
- [ ] `pipeline/c11_persona_extrapolation.py` — 300 generations + ~900 Claude judge calls. Runs in ~1–2 hrs; start it on the pod in a tmux while you do paper writing.
- [ ] **C2 pilot only** (3 traits × 5 personas × 3 conditions × 20 gens). Full sweep is 4,800 generations + 9,600 judge calls — not going to happen in a day. If the pilot shows a clean (b) > (a) and (b) > (c) signal, report it as preliminary; full sweep becomes a post-submission extension.

### 3. Robustness figures into paper
Colleague's R1–R5 for both IV and CAA are done and live in W&B. They need to be pulled down and inserted.

- [ ] Download figures from the W&B runs linked in `RobustnessResultsIV.md` and `RobustnessResultsCAA.md` (8 runs × ~2–4 figures each).
- [ ] Save under `icml2026/figures/robustness/`.
- [ ] Decide which make it into the main paper vs. appendix — suggest: R1 bootstrap heatmap + R4 context-dependence bar chart in main; R2/R3/R5 to appendix.
- [ ] Write the new subsection: "Robustness of Extracted Vectors" (one paragraph each for R1–R5, ~0.75 page). This is a new subsection in §Geometric Structure or its own §.

### 4. Paper `\todo{}` hit-list
Every `\todo{}` in `main.tex` needs either a figure or a decision to cut. Current list (from grep):

- [ ] L113 — related work citations (Anthropic features, SAE lit, RepEng). 15 min with scholar.
- [ ] L168 — per-trait cosine similarity heatmaps (IV top, CAA bottom). Already exist in `outputs/gemma-2-27b-it/figures/per_trait_heatmaps.png` and `caa_figures/`. Just assemble and insert.
- [ ] L230 — layer_sweep.pdf — already at `outputs/gemma-2-27b-it/experiments/layer_sweep.pdf`. Copy into `icml2026/figures/` and insert.
- [ ] L285 — behavioral transfer heatmaps. Check whether these exist under `eval/` or `eval_alpha2/`; if not, skip and tighten prose.
- [ ] L314 — variance trajectory figure. OLMo trajectory outputs exist in `outputs/OLMo-2-1124-7B/figures/`. Pick the right file, insert.
- [ ] L400 / L405 — E4 probe transfer figures + results text. Depends on #1 pull.
- [ ] L429 — E5 SAE results (Jaccard, shared vs unique features). Depends on #1 pull.
- [ ] L452 — appendix: full context/persona system prompts. Dump from `data/personas/*.yaml`.
- [ ] L457 — appendix: full instruction pairs + CAA contrastive construction. Dump from `data/traits/*.json` + `data/caa/`.
- [ ] L462 — appendix: full IV–CAA comparison matrices. Depends on E3 output.
- [ ] L467 — appendix: LLM judge prompt, calibration. Pull from `persona_steering/evaluation.py`.

### 5. Final paper build + submit
- [ ] Abstract pass: align with the reframed story (RepEng results are implicitly Assistant-conditioned; persona-conditioning is measurable geometrically and behaviorally, survives the E9 baseline-drift control, shows up developmentally at SFT).
- [ ] Conclusion + safety-implications pass.
- [ ] `latexmk -pdf main.tex` clean build, no undefined refs, no overfull boxes that matter.
- [ ] Submit.

---

## P1 — Do only if P0 finishes with time to spare

- [ ] Full C2 sweep (kick off overnight on pod if P0 is green by evening).
- [ ] E10 prompt-content matching control (second reviewer defense, cheap in principle).
- [ ] Layer sweep for C9 (confirm residualization gate holds at layers 15 and 30, not just 22).

## P2 — Post-submission / rebuttal prep

- [ ] C2 full sweep analysis.
- [ ] E4, E5 extensions if pod results are incomplete.
- [ ] E12 cross-model replication (Llama / Qwen).
- [ ] E13 SAE feature variance under persona shift.
- [ ] Maniprobe direction (new experiments in `maniprobe/`) — scope separately after submission.

---

## Fallback if C9 gate fails

If `cos_resid_t > 0.95` for most traits, the current story is undermined. Reframe the paper as:

> "Existing persona-trait steering work is dominated by persona baseline drift in activation space. We show how to disentangle baseline shift from genuine trait-representation change, and re-evaluate the cross-persona transfer story under the corrected measurement."

This is still a contribution, still uses all the same data, and the robustness + trajectory results still land. But the narrative changes from "persona-conditioned representations" to "methodology for measuring them correctly."

Decide within the first hour of the day based on C9 output.

---

## Day-of sequencing (concrete)

| Block | Task |
|-------|------|
| Morning 1 (09–11) | Kick off pod rsync; run C9 locally on pulled-or-existing activations; inspect. |
| Morning 2 (11–13) | C1 + C11 kickoff on pod (tmux). Start pulling W&B robustness figures. |
| Afternoon 1 (13–16) | Paper: insert already-existing figures (layer sweep, per-trait heatmaps, trajectory, robustness). Kill easy `\todo{}`s. |
| Afternoon 2 (16–19) | Write robustness subsection + E4/E5 results text from pulled outputs. |
| Evening (19–22) | C2 pilot results in; insert; abstract + conclusion pass; clean build. |
| Late (22+) | Final proofread + submit. |
