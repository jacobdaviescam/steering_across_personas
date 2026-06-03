#!/usr/bin/env bash
# Re-run steered generation with v2 vectors + L2-normalisation rescaled to mean ||v||.
# Outputs to outputs/{model}/steered_responses_alpha2_v2_norm/.
# Then runs p1_classifier_on_steered.py against the new dir.
set -euo pipefail
cd /workspace/steering_across_personas
set -a; source .env; set +a
export PATH=/workspace/venv/bin:$PATH
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/.pipcache
export XDG_CACHE_HOME=/workspace/.cache
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

MODEL=google/gemma-2-27b-it
V2_VECTORS=outputs/gemma-2-27b-it/v2/vectors
OUT=outputs/gemma-2-27b-it/steered_responses_alpha2_v2_norm
RESIDUE_OUT=outputs/gemma-2-27b-it/v2/persona_residue_v2_norm

V2_PERSONAS="con_artist drill_sergeant farmer kindergarten_teacher politician professor street_hustler surgeon tech_ceo therapist"

echo "[steer-v2-norm] $(date -Iseconds) ===== 8: steered generation (v2 vectors, normalised, 10 v2 personas only) ====="
python pipeline/8_steered_generation.py \
    --model         $MODEL \
    --vectors-dir   $V2_VECTORS \
    --output-dir    $OUT \
    --source-personas $V2_PERSONAS \
    --target-personas $V2_PERSONAS \
    --layer 22 \
    --alpha 2.0 \
    --n-questions 5 \
    --normalize

echo "[steer-v2-norm] $(date -Iseconds) ===== p1: classifier residue on new outputs ====="
python pipeline/p1_classifier_on_steered.py \
    --steered-dir    $OUT \
    --classifier-dir outputs/gemma-2-27b-it/v2/classifier \
    --output-dir     $RESIDUE_OUT

echo "[steer-v2-norm] $(date -Iseconds) DONE"
