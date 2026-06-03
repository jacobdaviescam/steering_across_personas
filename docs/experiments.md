# Experiments To Run

## Pod Setup

```bash
# Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# Authenticate (set API key for headless use)
export ANTHROPIC_API_KEY="your-key"

# Or interactive login (opens browser URL to paste back)
claude auth login

# Install project
pip install -e .
pip install -e assistant-axis-ref/
```

If your pod has persistent storage on `~/`, this only needs to happen once. If ephemeral, add the curl line to your startup script — it's idempotent.

---

All experiments assume the main pipeline (steps 0–9) has been run for `google/gemma-2-27b-it` and that `outputs/gemma-2-27b-it/{activations,vectors,caa_activations,caa_vectors}` exist.

`SHORT=gemma-2-27b-it` and `LAYER=22` throughout.

---

## Step 0: Generate data for new basin personas

E6 and E7 need 17 new personas (added in `data/personas/`) run through the pipeline on 3 traits. **This is the bottleneck — do this first on GPU, then everything else follows.**

Existing data: 10 personas x 8 traits = 80 combos (already done).
New data needed: 19 new persona x trait combos, ~3,800 responses.

The pipeline skips existing files, so it's safe to include existing personas in the command.

```bash
# 0a. Generate responses (GPU, vLLM, ~1-2 hrs)
python pipeline/1_generate.py \
    --model google/gemma-2-27b-it \
    --personas default librarian science_teacher journalist doctor diplomat \
               defence_lawyer poker_player spy counsellor nurse \
               hostage_negotiator interrogator accountant venture_capitalist \
               base_jumper war_correspondent \
               con_artist therapist kindergarten_teacher drill_sergeant \
               farmer surgeon tech_ceo street_hustler \
    --traits honesty empathy risk_taking

# 0b. Extract activations (GPU, ~1 hr)
python pipeline/2_activations.py \
    --model google/gemma-2-27b-it \
    --responses-dir outputs/gemma-2-27b-it/responses \
    --output-dir outputs/gemma-2-27b-it/activations

# 0c. Compute contrastive vectors (CPU, ~5 min)
python pipeline/3_vectors.py \
    --activations-dir outputs/gemma-2-27b-it/activations
```

After this, all experiments below can run.

---

## E2: Bootstrap CIs on shared variance

Resamples contrastive activation pairs with replacement, recomputes vectors per context, reports 95% CIs on shared variance ratio. Needed for the paper's Table 1 error bars.

**Requires:** CPU only (~30 min)

```bash
# IV activations
python pipeline/e2_bootstrap_rho.py \
    --activations-dir outputs/gemma-2-27b-it/activations \
    --layer 22

# IV + CAA together
python pipeline/e2_bootstrap_rho.py \
    --activations-dir outputs/gemma-2-27b-it/activations \
    --caa-activations-dir outputs/gemma-2-27b-it/caa_activations \
    --layer 22
```

**Outputs:** `outputs/gemma-2-27b-it/experiments/` — per-trait bootstrap distributions + 95% CIs

---

## E3: IV–CAA geometric decomposition

Projects CAA vectors onto the IV direction to decompose the CAA-specific component. Tests whether CAA captures a context-modulation signal beyond what IV finds.

**Requires:** CPU only (~5 min), needs both IV and CAA vectors

```bash
python pipeline/e3_iv_caa_decomposition.py \
    --iv-dir outputs/gemma-2-27b-it/vectors \
    --caa-dir outputs/gemma-2-27b-it/caa_vectors \
    --layer 22
```

**Outputs:** `outputs/gemma-2-27b-it/experiments/` — projection magnitudes, orthogonal residuals, per-trait decomposition

---

## E4: Probe transfer across contexts

Trains logistic regression probes per persona to detect trait presence, evaluates cross-persona transfer. Key safety experiment — measures how much a monitoring probe degrades outside its training context.

**Requires:** CPU only (~15 min)

```bash
# IV
python pipeline/e4_probe_transfer.py \
    --activations-dir outputs/gemma-2-27b-it/activations \
    --layer 22

# CAA
python pipeline/e4_probe_transfer.py \
    --activations-dir outputs/gemma-2-27b-it/caa_activations \
    --layer 22 \
    --output-dir outputs/gemma-2-27b-it/experiments/e4_caa
```

**Outputs:** `outputs/gemma-2-27b-it/experiments/` — self vs cross accuracy, transfer gap per trait, confusion matrices

---

## E5: SAE feature divergence

Encodes trait activations through a Gemma Scope SAE (131K features, layer 22) to identify which features correspond to each trait in each context. Tests whether different SAE features activate for the same trait in different persona contexts.

**Requires:** GPU optional but recommended (~20 min)

```bash
python pipeline/e5_sae_features.py \
    --activations-dir outputs/gemma-2-27b-it/activations \
    --sae-repo google/gemma-scope-27b-pt-res \
    --sae-folder layer_22/width_131k/average_l0_82 \
    --layer 22
```

**Outputs:** `outputs/gemma-2-27b-it/experiments/` — Jaccard similarity of top-K features across contexts, shared vs unique features per trait

---

## E6: Basin geometry

Tests whether trait representations form basins — cosine similarity to the default assistant decays monotonically along a conceptual gradient of personas. Three trait gradients (honesty: 11 personas, empathy: 8, risk-taking: 9) with a priori ring assignments.

**Prerequisite:** Step 0 (data generation for basin personas).

### E6 step 4: Basin analysis (CPU, ~5 min)

Core analysis: cosine similarity vs ring, Spearman rank correlation, 10K permutation test, cross-trait control.

```bash
python pipeline/e6_basin_geometry.py \
    --vectors-dir outputs/gemma-2-27b-it/vectors \
    --layer 22
```

**Outputs:** `outputs/gemma-2-27b-it/analysis/basin/` — `basin_results.json`, `cross_trait_control.json`

### E6 step 5: Dynamic positional activations (GPU)

Extracts activations at token positions [50, 100, 200, 400, 800] to test whether representations drift further from default as generation continues (basin attractor dynamics).

```bash
python pipeline/e6b_dynamic_activations.py \
    --model google/gemma-2-27b-it \
    --responses-dir outputs/gemma-2-27b-it/responses \
    --layer 22
```

**Outputs:** `outputs/gemma-2-27b-it/analysis/basin/dynamic/` — per-persona positional vector `.pt` files

### E6 step 6: Figures (CPU)

Generates 5 figure types: similarity-vs-ring curves, overlay plot, cross-trait control heatmap, dynamic drift curves, permutation null distributions.

```bash
python pipeline/e6c_basin_figures.py \
    --basin-dir outputs/gemma-2-27b-it/analysis/basin \
    --dynamic-dir outputs/gemma-2-27b-it/analysis/basin/dynamic \
    --vectors-dir outputs/gemma-2-27b-it/vectors \
    --layer 22
```

**Outputs:** `outputs/gemma-2-27b-it/analysis/basin/figures/` — PDF + PNG for all figure types

---

## E7: SAE sparse codes as persona fingerprints

Extends E5 with a sharper question. Instead of just comparing top-K feature sets (Jaccard), keeps the full contrastive sparse codes and uses them to build a mechanistic persona landscape.

Each persona x trait gets a 131K-dim sparse code: `SAE(mean_pos_activation) - SAE(mean_neg_activation)`. This is the persona's trait fingerprint in the SAE feature basis.

Three analyses:
1. **Sparse code similarity** — cosine similarity of sparse codes across personas. Should decay along E6 basin gradients (same pattern as steering vectors, but now in feature space).
2. **Feature classification** — universal features (active in >80% of personas for a trait) vs discriminative features (active in 1-2 only). The ratio should correlate with shared variance from the main decomposition.
3. **Basin gradient in feature space** — Spearman correlation between ring assignment and sparse code similarity to default. If this replicates E6's finding, the basin structure is not just a property of contrastive directions but of the underlying features.

**Requires:** GPU optional (~20 min), needs activations for basin personas (Step 0)

```bash
python pipeline/e7_sparse_codes.py \
    --activations-dir outputs/gemma-2-27b-it/activations \
    --sae-repo google/gemma-scope-27b-pt-res \
    --sae-folder layer_22/width_131k/average_l0_82 \
    --layer 22
```

**Outputs:** `outputs/gemma-2-27b-it/experiments/e7_sparse_codes/` — `sparse_code_analysis.json` (per-trait similarity matrices, feature classification), `sparse_code_basin.json` (basin gradient in feature space), `contrastive_sparse_codes.pt` (raw codes for downstream use)

---

## Run order

**Do Step 0 first** (data generation for new personas, GPU). Then E2–E5 can run on existing data in any order. E6 analysis and E7 need Step 0 outputs.

| Step | GPU needed | Approx time | Dependencies |
|------|-----------|-------------|--------------|
| **Step 0** | **Yes** | **~2-3 hrs** | **model weights, vLLM — do this first** |
| E2 | No | ~30 min | activations/, caa_activations/ |
| E3 | No | ~5 min | vectors/, caa_vectors/ |
| E4 | No | ~15 min | activations/, caa_activations/ |
| E5 | Optional | ~20 min | activations/ |
| E6 steps 4,6 | No | ~10 min | Step 0 outputs |
| E6 step 5 | Yes | ~30 min | Step 0 outputs, model weights |
| E7 | Optional | ~20 min | Step 0 outputs, SAE weights |
