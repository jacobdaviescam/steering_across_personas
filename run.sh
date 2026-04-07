#!/usr/bin/env bash
# run.sh — Run the full persona-steering pipeline for a given model.
#
# Runs both extraction methods (IV and CAA) and all downstream analysis.
# Each step is idempotent: existing output files are skipped automatically.
#
# Usage:
#   ./run.sh google/gemma-2-27b-it          # full pipeline, both methods
#   ./run.sh google/gemma-2-27b-it --iv     # instruction-variant only
#   ./run.sh google/gemma-2-27b-it --caa    # CAA only
#   ./run.sh google/gemma-2-27b-it --from 3 # resume from step 3
#
# Prerequisites:
#   - pip install -e .
#   - .env with ANTHROPIC_API_KEY, HF_TOKEN (optional: WANDB_API_KEY)
#   - GPU access for steps 1, 2, 2c, 8
#   - assistant-axis-ref/ cloned

set -euo pipefail

MODEL="${1:?Usage: ./run.sh <model> [--iv|--caa] [--from N] [--layer L]}"
shift

# Defaults
RUN_IV=true
RUN_CAA=true
FROM_STEP=0
LAYER=22

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --iv)  RUN_IV=true; RUN_CAA=false; shift ;;
        --caa) RUN_IV=false; RUN_CAA=true; shift ;;
        --from) FROM_STEP="$2"; shift 2 ;;
        --layer) LAYER="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

SHORT="${MODEL##*/}"
OUTPUTS="outputs/${SHORT}"

step() {
    local n="$1"; shift
    if (( n < FROM_STEP )); then
        echo "--- Skipping step $n (--from $FROM_STEP) ---"
        return
    fi
    echo ""
    echo "=== Step $n: $1 ==="
    shift
    "$@"
}

# ─── Data generation (requires ANTHROPIC_API_KEY) ────────────────────────

if $RUN_IV; then
    step 0 "Generate IV trait datasets" \
        python pipeline/0_generate_data.py --traits
fi

if $RUN_CAA; then
    step 0 "Generate CAA datasets" \
        python pipeline/0c_generate_caa_data.py --traits
fi

# ─── IV branch: generate responses + extract activations (GPU) ───────────

if $RUN_IV; then
    step 1 "Generate IV responses via vLLM" \
        python pipeline/1_generate.py --model "$MODEL"

    step 2 "Extract IV activations" \
        python pipeline/2_activations.py --model "$MODEL"

    step 3 "Compute IV contrastive vectors" \
        python pipeline/3_vectors.py --activations-dir "${OUTPUTS}/activations"

    step 4 "IV analysis" \
        python pipeline/4_analysis.py \
            --vectors-dir "${OUTPUTS}/vectors" \
            --layer "$LAYER"

    step 5 "IV figures" \
        python pipeline/5_visualize.py \
            --analysis-dir "${OUTPUTS}/analysis" \
            --vectors-dir "${OUTPUTS}/vectors"

    step 5 "IV persona landscape" \
        python pipeline/5b_persona_landscape.py \
            --vectors-dir "${OUTPUTS}/vectors" \
            --analysis-dir "${OUTPUTS}/analysis"

    step 5 "IV activation landscape" \
        python pipeline/5_activation_landscape.py \
            --activations-dir "${OUTPUTS}/activations" \
            --vectors-dir "${OUTPUTS}/vectors" \
            --layer "$LAYER"
fi

# ─── CAA branch: extract activations (GPU, no generation needed) ─────────

if $RUN_CAA; then
    step 2 "Extract CAA activations" \
        python pipeline/2c_caa_activations.py --model "$MODEL"

    step 3 "Compute CAA contrastive vectors" \
        python pipeline/3_vectors.py \
            --activations-dir "${OUTPUTS}/caa_activations" \
            --output-dir "${OUTPUTS}/caa_vectors"

    step 4 "CAA analysis" \
        python pipeline/4_analysis.py \
            --vectors-dir "${OUTPUTS}/caa_vectors" \
            --output-dir "${OUTPUTS}/caa_analysis" \
            --layer "$LAYER"

    step 5 "CAA figures" \
        python pipeline/5_visualize.py \
            --analysis-dir "${OUTPUTS}/caa_analysis" \
            --vectors-dir "${OUTPUTS}/caa_vectors" \
            --output-dir "${OUTPUTS}/caa_figures"

    step 5 "CAA persona landscape" \
        python pipeline/5b_persona_landscape.py \
            --vectors-dir "${OUTPUTS}/caa_vectors" \
            --analysis-dir "${OUTPUTS}/caa_analysis" \
            --output-dir "${OUTPUTS}/caa_figures/persona_landscape"

    step 5 "CAA activation landscape" \
        python pipeline/5_activation_landscape.py \
            --activations-dir "${OUTPUTS}/caa_activations" \
            --vectors-dir "${OUTPUTS}/caa_vectors" \
            --layer "$LAYER"
fi

# ─── Behavioral evaluation (requires ANTHROPIC_API_KEY) ──────────────────

if $RUN_IV; then
    step 6 "Behavioral evaluation (IV responses)" \
        python pipeline/6_behavioral_eval.py \
            --responses-dir "${OUTPUTS}/responses"

    step 7 "Evaluation analysis" \
        python pipeline/7_eval_analysis.py \
            --eval-dir "${OUTPUTS}/eval"
fi

# ─── Steered generation + evaluation (GPU + ANTHROPIC_API_KEY) ───────────

if $RUN_IV; then
    VECTORS_DIR="${OUTPUTS}/vectors"
elif $RUN_CAA; then
    VECTORS_DIR="${OUTPUTS}/caa_vectors"
fi

step 8 "Steered generation" \
    python pipeline/8_steered_generation.py \
        --model "$MODEL" \
        --vectors-dir "$VECTORS_DIR" \
        --layer "$LAYER"

step 9 "Steering evaluation" \
    python pipeline/9_steering_eval.py \
        --steered-dir "${OUTPUTS}/steered_responses" \
        --output-dir "${OUTPUTS}/steering_eval"

echo ""
echo "=== Pipeline complete for ${MODEL} ==="
echo "Outputs: ${OUTPUTS}/"
