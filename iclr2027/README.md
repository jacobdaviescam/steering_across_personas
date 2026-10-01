# ICLR 2027 results: conditionality of trait representations

This folder holds everything produced for the ICLR 2027 conference paper on top of the
ICML 2026 workshop pipeline. Code changes to the pipeline itself are in `pipeline/`
(see "Pipeline changes" below). Large activation files are not in git; see "Where the
raw outputs are".

## What was run

Every cell is a CAA direction for one (model, layer, context, trait), extracted with
`pipeline/2c_caa_activations.py` from answer-token activations over 500 questions, with
the context text in the user message on every model. Contexts are the 10 occupational
personas from the workshop paper plus seven evaluation contexts
(`data/personas/{auto_grader,human_evaluator,simulated_env,real_deployment,under_evaluation,unobserved,coding_harness}.yaml`,
five paraphrases each), a null context and gibberish prompts.

| Result | Models | Driver script | Analysis script | Output |
|---|---|---|---|---|
| Evaluation contexts vs floors (Result C) | Gemma-2-27B-IT L22, Llama-3.1-8B-Instruct L15/L20, Qwen3-32B L31/L42/L49, Qwen3-8B L18/L24 | `scripts/run_grid.sh`, `run_queue.sh`, `run_qwen8b.sh` | `scripts/analyze_evalctx.py` (floors: bootstrap, paraphrase, gibberish band), `scripts/paired_boot.py` (paired question-bootstrap, 200 redraws) | `analysis/<model>_evalctx_L<layer>/`, `analysis/evalctx_L22_*` |
| Assistant-axis decomposition | Gemma L22 | | `scripts/axis_projection.py` (Lu et al. axis) | `analysis/evalctx_L22_*/axis*` |
| OLMo-2 stages, raw format vs chat template | OLMo-2-1124-7B base/SFT/DPO/Instruct, L15 | `scripts/run_queue2.sh` | `scripts/olmo_stage_analysis.py` | `analysis/olmo_rawfmt_control_L15.json` |
| Qwen3 base vs instruct under raw format | Qwen3-8B, 14B pairs; Qwen3-32B raw; Qwen2.5-32B | `scripts/run_master.sh`, `run_extra.sh` | `scripts/analyze_evalctx.py` | on the volume (see below) |
| Reward-hacking RL trajectory | gpt-oss-120b, 5 checkpoints (uwuwuwuwuwuwu/gpt-oss-120b-reward-hacker-step-{0,216,496,752,952}) | `scripts/gptoss_trajectory.py` (uses `gptoss_lora.py`) | `scripts/trajectory_summary.py` | `trajectory/` |
| Forced-choice behavioural readout (experiment 2, prepared, not run) | | `scripts/run_exp2.sh`, `pod_bootstrap_exp2.sh` | `scripts/exp2_forced_choice.py` | |

Figures: `figs_v4/` (spread metric per trait and context, s1 to s8, built by `scripts/make_figs_v4.py`),
`figs_trajectory/` (t1 to t4, `scripts/make_figs_trajectory.py`), `figs_v3/` and the top-level
`fig*.png` are earlier versions. Decks: `iclr2027_results_2026-09-15_v3.pptx`, `iclr2027_results_2026-09-17_v4.pptx`.

Authoritative numbers, dates and decisions are in the results ledger (vault note, ask Jacob)
and in the status doc shared with co-authors.

## Pipeline changes (branch `iclr2027/results` vs `main`)

`pipeline/2c_caa_activations.py` gained:

- `--paraphrase-variants N --paraphrase-questions M`: write paraphrases 1..N-1 on the first M questions to `<stem>.variants.pt` (paraphrase floor).
- `--persona-placement user`: persona text in the user message rather than the system prompt (cross-model comparability; Gemma has no system role).
- `--raw-format`: plain Context/Question/Answer text with no chat template (base vs instruct comparisons; isolates the template effect).
- `--save-layers 9,12,...`: store per-question activations only at the listed layers (`layers.json` sidecar); all-layer per-cell means go to `<stem>.means.pt`. Cuts disk 5 to 8 times.
- `--save-logits` / `--logits-only`: store answer-token logits for the forced-choice readout.
- Qwen3: thinking disabled in the template.

`pipeline/t1_trajectory_activations.py` gained `--force-raw-format`, `--dir-suffix`, `--save-layers`.
The `scripts/patch_*.py` files are the patches that produced these changes.

## gpt-oss-120b trajectory

The reward-hacker checkpoints are Tinker LoRA exports (r=32, alpha=32) over attention, all 128
experts and the unembedding. PEFT cannot apply them (attention keys are `attn.*`, expert
factors are per-expert 3-D tensors on fused MXFP4 parameters). `scripts/gptoss_lora.py` applies
them functionally: the base is loaded dequantised to bf16 with `experts_implementation="eager"`,
and the expert forward is reimplemented with the gate/up/down corrections inside the
nonlinearity. Validation gates (in `trajectory/gate_report.json`): the step-0 adapter reproduces
the base exactly (cosine 1.0000 at every layer); later steps drift gradedly; forced-choice
log-odds on 60 School of Reward Hacks items (`data/sorh_forced_choice.jsonl`) move from
−3.06 to −0.50 between step 0 and step 952. Runs on one 288 GB card in about 4 hours.

## Where the raw outputs are

- Per-question activations (thinned to analysis layers), cell means, variants and logits:
  RunPod network volume `fgnwcvlmzi` (EU-RO-1) under `/workspace/iclr2026/repo/outputs/<model>/`.
  Readable over S3 (`aws s3 --endpoint-url https://s3api-eu-ro-1.runpod.io --region eu-ro-1`); ask Jacob for credentials.
- gpt-oss trajectory: raw activations were not preserved (Hub private-storage quota at upload
  time). Cross-checkpoint summaries at L12/L18/L24 and the gate report are in `trajectory/`
  and in `summaries.tar.gz` on the private HF dataset `jacobdavies/iclr2027-conditionality-results`.
- Workshop-paper Gemma grid: `girishgupta/persona-steering-activations` on the Hub
  (its README's "5 paraphrases x 100 questions" is wrong; the files are 1 paraphrase x 500 questions).

## Known caveats

- Trajectory run used one gibberish prompt per cell, not the five-prompt band.
- Gemma and Llama paired intervals were computed against the single gibberish prompt; the
  five-prompt bands exist for those models but the intervals were not recomputed against them.
- Behavioural experiments (probe transfer, logits readout, steering, free text + judge) were
  planned and not run.
