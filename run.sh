#!/usr/bin/env bash
# run.sh — Run the full persona-steering pipeline for a given model.
#
# Runs both extraction methods (IV and CAA) and all downstream analysis.
# Each step is idempotent: existing output files are skipped automatically.
#
# Usage:
#   ./run.sh google/gemma-2-27b-it              # full pipeline, both methods
#   ./run.sh google/gemma-2-27b-it --iv         # instruction-variant only
#   ./run.sh google/gemma-2-27b-it --caa        # CAA only
#   ./run.sh google/gemma-2-27b-it --from 3     # resume from step 3
#   ./run.sh google/gemma-2-27b-it --to 3       # stop after step 3
#   ./run.sh google/gemma-2-27b-it --from 10    # robustness only (needs steps 0-3 done)
#   ./run.sh --trajectory                       # OLMo training trajectory pipeline
#
# Prerequisites (auto-checked below):
#   - .env with ANTHROPIC_API_KEY, HF_TOKEN (optional: WANDB_API_KEY)
#   - GPU access for steps 1, 2, 2c, 8

set -euo pipefail

# ─── Prerequisites ──────────────────────────────────────────────────────
if [ ! -d "assistant-axis-ref" ]; then
    echo "--- Cloning assistant-axis-ref ---"
    git clone https://github.com/safety-research/assistant-axis.git assistant-axis-ref
fi

if ! python -c "import assistant_axis" 2>/dev/null; then
    echo "--- Installing assistant-axis ---"
    pip install -e assistant-axis-ref/
fi

if ! python -c "import persona_steering" 2>/dev/null; then
    echo "--- Installing persona_steering ---"
    pip install -e .
fi

# ─── Trajectory mode (OLMo training stages) ────────────────────────────
if [[ "${1:-}" == "--trajectory" ]]; then
    shift
    FROM_STEP=1
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --from) FROM_STEP="$2"; shift 2 ;;
            *) echo "Unknown flag: $1"; exit 1 ;;
        esac
    done

    step() {
        local n="$1"; shift
        if (( n < FROM_STEP )); then
            echo "--- Skipping t$n (--from $FROM_STEP) ---"
            return
        fi
        echo ""
        echo "=== t$n: $1 ==="
        shift
        "$@"
    }

    step 1 "Extract CAA activations across OLMo checkpoints" \
        python pipeline/t1_trajectory_activations.py

    step 2 "Compute trajectory contrastive vectors" \
        python pipeline/t2_trajectory_vectors.py

    step 3 "Cross-stage trajectory analysis" \
        python pipeline/t3_trajectory_analysis.py

    step 4 "Trajectory figures" \
        python pipeline/t4_trajectory_figures.py

    echo ""
    echo "=== Trajectory pipeline complete ==="
    echo "Outputs: outputs/OLMo-2-1124-7B/"
    exit 0
fi

# ─── Main pipeline ─────────────────────────────────────────────────────

MODEL="${1:?Usage: ./run.sh <model> [--iv|--caa] [--from N] [--to N] [--layer L] | ./run.sh --trajectory}"
shift

# Defaults
RUN_IV=true
RUN_CAA=true
FROM_STEP=0
TO_STEP=999
LAYER=22

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --iv)  RUN_IV=true; RUN_CAA=false; shift ;;
        --caa) RUN_IV=false; RUN_CAA=true; shift ;;
        --from) FROM_STEP="$2"; shift 2 ;;
        --to) TO_STEP="$2"; shift 2 ;;
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
    if (( n > TO_STEP )); then
        echo "--- Stopping at step $n (--to $TO_STEP) ---"
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

    step 5 "IV figures (5a: main)" \
        python pipeline/5_visualize.py \
            --analysis-dir "${OUTPUTS}/analysis" \
            --vectors-dir "${OUTPUTS}/vectors"

    step 5 "IV figures (5b: persona landscape)" \
        python pipeline/5b_persona_landscape.py \
            --vectors-dir "${OUTPUTS}/vectors" \
            --analysis-dir "${OUTPUTS}/analysis"

    step 5 "IV figures (5c: activation landscape)" \
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

    step 5 "CAA figures (5a: main)" \
        python pipeline/5_visualize.py \
            --analysis-dir "${OUTPUTS}/caa_analysis" \
            --vectors-dir "${OUTPUTS}/caa_vectors" \
            --output-dir "${OUTPUTS}/caa_figures"

    step 5 "CAA figures (5b: persona landscape)" \
        python pipeline/5b_persona_landscape.py \
            --vectors-dir "${OUTPUTS}/caa_vectors" \
            --analysis-dir "${OUTPUTS}/caa_analysis" \
            --output-dir "${OUTPUTS}/caa_figures/persona_landscape"

    step 5 "CAA figures (5c: activation landscape)" \
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

# ─── Robustness experiments (CPU only, no API) ────────────────────────────

if $RUN_IV; then
    ACT_DIR="${OUTPUTS}/activations"
    VEC_DIR="${OUTPUTS}/vectors"
else
    ACT_DIR="${OUTPUTS}/caa_activations"
    VEC_DIR="${OUTPUTS}/caa_vectors"
fi

step 10 "Bootstrap stability (r1)" \
    python pipeline/r1_bootstrap_vectors.py \
        --activations-dir "$ACT_DIR" \
        --vectors-dir "$VEC_DIR" \
        --layer "$LAYER"

step 10 "Convergence analysis (r2)" \
    python pipeline/r2_convergence.py \
        --activations-dir "$ACT_DIR" \
        --vectors-dir "$VEC_DIR" \
        --layer "$LAYER"

step 10 "Syntactic invariance (r3)" \
    python pipeline/r3_syntactic_invariance.py \
        --activations-dir "$ACT_DIR" \
        --layer "$LAYER"

step 10 "General vs contextual (r4)" \
    python pipeline/r4_general_vs_contextual.py \
        --vectors-dir "$VEC_DIR" \
        --layer "$LAYER"

step 10 "Context similarity (r6)" \
    python pipeline/r6_context_similarity.py \
        --vectors-dir "$VEC_DIR" \
        --layer "$LAYER"

if $RUN_IV; then
    step 10 "Variant convergence (r7)" \
        python pipeline/r7_variant_convergence.py \
            --activations-dir "$ACT_DIR" \
            --layer "$LAYER"
fi

step 10 "Cluster bias (r8)" \
    python pipeline/r8_cluster_bias.py \
        --vectors-dir "$VEC_DIR" \
        --layer "$LAYER"

step 10 "Safety context dependence (r9)" \
    python pipeline/r9_safety_context.py \
        --vectors-dir "$VEC_DIR" \
        --layer "$LAYER"

echo ""
echo "=== Pipeline complete for ${MODEL} ==="
echo "Outputs: ${OUTPUTS}/"
