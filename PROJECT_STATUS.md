# Project Status — Persona-Conditional Steering Vectors

**Last updated**: 2026-04-10 (written to network volume, not tracked by git).

---

## TL;DR

Gemma 2 27B IT pipeline is **complete** for the paper-hardening experiments (E2–E7). Results are on disk at `/workspace/steering_across_personas/outputs/gemma-2-27b-it/` and in wandb project `persona-steering` under group `gemma-2-27b-it`. Two source-code commits landed locally (unpushed at time of writing); see "Git state" below. The next step — running the same pipeline on Gemma 3 27B IT for cross-model validation — is **blocked on a paper-scope decision**.

---

## What completed this session

### New Gemma 2 27B IT data generated

Step 0 (from EXPERIMENTS.md) ran end-to-end for the 17 basin personas on 3 traits (honesty, empathy, risk-taking), on top of the existing 10 archetypes × 8 traits that were already on disk:

- `outputs/gemma-2-27b-it/responses/` — 262 JSONL files
- `outputs/gemma-2-27b-it/activations/` — 262 `.pt` files
- `outputs/gemma-2-27b-it/vectors/` — 131 `.pt` files (80 pre-existing, 51 new)

The core-persona vectors were also silently recomputed (their files were rewritten bit-for-bit identically), because `pipeline/3_vectors.py` does not skip existing files by default. That's an optimization opportunity but caused no data loss.

### Experiments with results now on disk

| Experiment | Path | Notes |
|---|---|---|
| E2 bootstrap (IV+CAA) | `experiments/bootstrap_rho.json` | ρ_t with 500-iter 95% CIs, all 8 traits |
| E3 IV-CAA decomposition | `experiments/iv_caa_decomposition.{json,pdf,png}` | from prior session |
| E4 probe transfer (IV) | `experiments/probe_transfer.{json,pdf,png}` | self/cross/gap, all 8 traits |
| E4 probe transfer (CAA) | `experiments/e4_caa/probe_transfer.{json,pdf,png}` | CAA counterpart, gaps much smaller |
| E6 basin geometry | `analysis/basin/basin_results.json`, `cross_trait_control.json` | honesty, empathy, risk-taking gradients |
| E6b dynamic activations | `analysis/basin/dynamic/` | positional activations at tokens [50, 100, 200, 400, 800] |
| E6c basin figures | `analysis/basin/figures/` | PDF+PNG for all basin figure types |
| E7 sparse codes | `experiments/e7_sparse_codes/` | **ran unexpectedly — see caveat below** |

#### Summary numbers (E2 ρ_t, E4 transfer gaps)

| Trait | ρ_t [95% CI] | IV gap | CAA gap |
|---|---|---|---|
| honesty | 0.928 [0.918, 0.928] | +0.212 | +0.038 |
| assertiveness | 0.885 [0.875, 0.888] | +0.100 | +0.051 |
| confidence | 0.864 [0.847, 0.867] | +0.205 | +0.028 |
| warmth | 0.851 [0.840, 0.854] | +0.100 | +0.080 |
| deference | 0.839 [0.814, 0.840] | +0.249 | +0.057 |
| empathy | 0.838 [0.828, 0.842] | +0.091 | +0.051 |
| impulsivity | 0.752 [0.729, 0.758] | +0.280 | +0.060 |
| risk_taking | 0.729 [0.697, 0.735] | +0.177 | +0.048 |

Rough story: CAA probes generalize across contexts (tiny gaps); IV probes degrade substantially. Honesty has the **highest** shared variance *and* one of the **highest** IV probe gaps — a paper-worthy contradiction suggesting the contrastive direction is consistent but the surrounding activation geometry isn't.

#### Caveat on E7

E7 was supposed to be skipped this run (Gemma Scope only ships SAEs for base 27B, not 27B IT — see `~/.claude/projects/-workspace-steering-across-personas/memory/sae_availability.md`). But `experiments/e7_sparse_codes/` exists with outputs, which means either (a) the edit to the orchestration script that commented E7 out didn't take effect in the running bash process, or (b) E7 did run using the base-model SAE anyway. The results should be treated as **unvalidated — pt→it SAE transfer is an unverified assumption**. Don't cite them in the paper without the Gemma 3 replication (see below).

---

## Git state

### Local commits landed this session (unpushed)

On branch `experiments/paper-hardening`:

- `3ed6d1a` — Share JumpReLU SAE loader; support Gemma Scope 2 format
- `af3baab` — Add wandb to E2/E4; add tmux orchestration script

### Intentionally NOT committed

- `outputs/gemma-2-27b-it/vectors/*.pt` (25 files show as Modified) — tracked-but-should-be-ignored binaries from before the `outputs/` gitignore rule. Their file contents were rewritten bit-identically by the vectors rerun. Recommended action for a future session: `git rm --cached outputs/gemma-2-27b-it/vectors/*.pt` followed by a commit to stop tracking them.
- `outputs/gemma-2-27b-it/steered_responses/*.jsonl` (5 deletions) — unknown provenance; they were already deleted before this session started. Decide on intent before committing the deletion.
- `experiments_output.log`, `gpu_experiments.log`, `cpu_experiments.log` — tmux log captures; metrics already synced to wandb, logs are ephemeral.
- `run_cpu_experiments.sh`, `run_experiments.sh` — untracked; not inspected, possibly stale drafts from pre-session. Worth a read next session before committing or deleting.

---

## What is open / blocked

### Biggest open decision: paper scope

The current 27 personas × 8 traits were chosen as a **method-validation set**, not a narrative set. The user (Jacob) wants to think about which context pairs best serve the paper's story, potentially swapping in new personas that don't exist yet.

**Unresolved questions** (see the scope writeup sent in conversation on 2026-04-10):

1. Which of these is the paper's **core claim**? Different claims point to different ideal persona sets:
   - A — "Trait representations are persona-conditional, not universal" (E2/E3 geometric story)
   - B — "Safety probes trained in one context don't generalize" (E4 story)
   - C — "Trait geometry has basin structure" (E6 story)
   - D — "CAA is a more robust extraction method than IV" (methods story)
2. Which personas in the current 27 are **load-bearing** for the existing paper draft? (I don't have access to the draft.)
3. For each flagship trait, what's the **one-sentence story** that the reader should take away?

**Once answered**: draft a concrete persona-and-trait shortlist, then finalize `run_gemma3.sh` to run *that* shortlist (not the full 27×8 matrix by default).

### Blocked on the scope decision

- `run_gemma3.sh` — not drafted yet. Will be a parameterized version of `run_all_experiments.sh` pointed at `google/gemma-3-27b-it` at layer 31, using `google/gemma-scope-2-27b-it/resid_post/layer_31_width_65k_l0_medium` for SAEs (or `width_262k_l0_medium` if we want closer parity with Gemma Scope 1's expansion factor).
- Pre-flight test for Gemma 3 model loading — needs a free GPU. Worth doing before kicking off the real run: confirm `vllm.LLM("google/gemma-3-27b-it")` loads cleanly (it's a multimodal wrapper and may need explicit kwargs) and `ProbingModel("google/gemma-3-27b-it").get_layers()` returns `model.language_model.layers`.
- OpenRouter integration for `pipeline/evaluation.py` and `pipeline/6_behavioral_eval.py` — deferred. We only have an OpenRouter key, no Anthropic key, and those scripts call `anthropic.Anthropic()` directly. Not needed for E2–E7, but required if we ever want to run main-pipeline steps 6/7/9 on a new model.

---

## Key facts for a future session

### Model / SAE mapping
- Current analysis model: `google/gemma-2-27b-it`, layer 22 (mid-stack of 46). No IT-matched SAE exists.
- Proposed cross-model: `google/gemma-3-27b-it`, layer 31 (mid-stack of 62). SAE: `google/gemma-scope-2-27b-it`.
- Gemma 3 27B IT is gated — access was granted on 2026-04-10 to the account bound to `$HF_TOKEN` in `.env`.

### Gemma Scope 2 format is different from Gemma Scope 1
Covered in `persona_steering/sae_loader.py` (shared loader handles both):
- `.safetensors` instead of `.npz`
- Path layout `<site>/layer_<L>_width_<W>_l0_<size>/` (not `<layer>/<width>/<l0>/`)
- `w_enc` keys are lowercase, shape `(d_in, d_sae)` — **transposed vs Gemma Scope 1** (loader normalizes internally to `(d_sae, d_in)`)
- L0 values are encoded as `small`/`medium`/`big` labels; for `layer_31_width_65k` these are L0=20, 60, 150 respectively.

### Pod / environment reminders
- **Always** `export HF_HOME=/workspace/.cache` and `export PIP_CACHE_DIR=/workspace/.cache/pip`. The root overlay is only ~20GB and fills instantly when downloading 27B-scale weights. A previous pod was lost to this.
- `.env` contains working `HF_TOKEN`, `WANDB_API_KEY`, `OPENROUTER_API_KEY`. Source it with `set -a && source .env && set +a` before running pipeline steps.
- `/workspace` is the persistent network volume (MFS fuse mount, ~petabyte). Anything under here survives pod shutdown; anything on `/` does not.
- vLLM is installed (tested on 0.19.0) but an env upgrade during install can wipe the editable `persona_steering` and `assistant_axis` installs. If they disappear: `pip install -e .` and `pip install -e assistant-axis-ref/`.

### Network disk is slow
`pipeline/3_vectors.py` is I/O-bound on the MFS mount, taking ~4.5 min per vector file that the math alone should take seconds for. If a future run cares about speed, copy activations to `/tmp` first. Noted but not fixed.

---

## wandb references
- Project: `persona-steering`
- Entity: `jacob_e_davies-university-of-cambridge`
- URL: https://wandb.ai/jacob_e_davies-university-of-cambridge/persona-steering
- Runs from this session: tagged `model:gemma-2-27b-it`, `step:e{2,4,5,6,7}_*`
