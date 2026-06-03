#!/bin/bash
# Run all experiments from docs/experiments.md
# Uses tmux to persist sessions across disconnects
# Note: per-experiment failures are logged but do NOT abort the pipeline

set -uo pipefail

cd /workspace/steering_across_personas

# Helper: run a step, log status, but never abort the pipeline
run_step() {
    local name="$1"
    shift
    echo ""
    echo "[$(date)] >>> START: $name"
    if "$@"; then
        echo "[$(date)] <<< OK:    $name"
    else
        local rc=$?
        echo "[$(date)] <<< FAIL:  $name (exit=$rc)" >&2
    fi
}

# --- Environment setup ---
export HF_HOME=/workspace/.cache
export PIP_CACHE_DIR=/workspace/.cache/pip

# Load .env (without export lines, just key=value)
set -a
source .env
set +a

SHORT=gemma-2-27b-it
LAYER=22
ACT_DIR=outputs/$SHORT/activations
CAA_ACT_DIR=outputs/$SHORT/caa_activations
VEC_DIR=outputs/$SHORT/vectors

echo "=== Environment ==="
echo "HF_HOME=$HF_HOME"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_API_KEY set: $([ -n \"${WANDB_API_KEY:-}\" ] && echo yes || echo no)"
echo "HF_TOKEN set: $([ -n \"${HF_TOKEN:-}\" ] && echo yes || echo no)"
echo "==================="

# --- Phase 1: CPU experiments (existing data) ---
# E2 (IV+CAA bootstrap) and E4 (CAA probe) already completed in previous run.
# Only E5 needs to run from Phase 1.

echo ""
echo "=== Phase 1: CPU experiments on existing data ==="

# E5: SAE feature divergence
run_step "E5 SAE features" \
    python pipeline/e5_sae_features.py \
        --activations-dir $ACT_DIR \
        --sae-repo google/gemma-scope-27b-pt-res \
        --sae-folder layer_22/width_131k/average_l0_82 \
        --layer $LAYER

echo ""
echo "=== Phase 1 complete ==="

# --- Phase 2: Step 0 - Generate data for new basin personas (GPU) ---
echo ""
echo "=== Phase 2: Step 0 - Generate new persona data ==="

# 0a: Generate responses (vLLM, GPU)
run_step "Step 0a response generation" \
    python pipeline/1_generate.py \
        --model google/gemma-2-27b-it \
        --personas default librarian science_teacher journalist doctor diplomat \
                   defence_lawyer poker_player spy counsellor nurse \
                   hostage_negotiator interrogator accountant venture_capitalist \
                   base_jumper war_correspondent \
                   con_artist therapist kindergarten_teacher drill_sergeant \
                   farmer surgeon tech_ceo street_hustler \
        --traits honesty empathy risk_taking

# 0b: Extract activations
run_step "Step 0b activation extraction" \
    python pipeline/2_activations.py \
        --model google/gemma-2-27b-it \
        --responses-dir outputs/$SHORT/responses \
        --output-dir outputs/$SHORT/activations

# 0c: Compute vectors
run_step "Step 0c vector computation" \
    python pipeline/3_vectors.py \
        --activations-dir outputs/$SHORT/activations

echo ""
echo "=== Phase 2 complete ==="

# --- Phase 3: E6 + E7 (need Step 0 outputs) ---
echo ""
echo "=== Phase 3: E6 basin geometry + E7 sparse codes ==="

# E6 step 4: Basin analysis
run_step "E6 basin geometry" \
    python pipeline/e6_basin_geometry.py \
        --vectors-dir outputs/$SHORT/vectors \
        --layer $LAYER

# E6 step 5: Dynamic positional activations (GPU)
run_step "E6b dynamic activations" \
    python pipeline/e6b_dynamic_activations.py \
        --model google/gemma-2-27b-it \
        --responses-dir outputs/$SHORT/responses \
        --layer $LAYER

# E6 step 6: Figures
run_step "E6c basin figures" \
    python pipeline/e6c_basin_figures.py \
        --basin-dir outputs/$SHORT/analysis/basin \
        --dynamic-dir outputs/$SHORT/analysis/basin/dynamic \
        --vectors-dir outputs/$SHORT/vectors \
        --layer $LAYER

# E7: SAE sparse codes — SKIPPED
# Gemma Scope only ships SAEs for base 27B (pt) and IT 9B, not 27B IT.
# Decide later: rerun pipeline on Gemma 3 27B IT (matching SAE) vs trusting
# pt→it transfer. Until then, no SAE experiments.
echo ""
echo "[$(date)] SKIPPING E7 sparse codes — no matching SAE for gemma-2-27b-it"

echo ""
echo "=========================================="
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "=========================================="
echo "[$(date)] Finished."
