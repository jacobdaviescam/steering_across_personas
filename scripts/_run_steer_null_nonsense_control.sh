#!/usr/bin/env bash
# Control (e): steer the v2 personas with null and nonsense vectors.
# Predicts zero or near-zero ΔP(src|cross) — null/nonsense vectors should
# not carry any persona-specific content.
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
OUT=outputs/gemma-2-27b-it/steered_responses_alpha2_v2_norm_nullnonsense
RESIDUE_OUT=outputs/gemma-2-27b-it/v2/persona_residue_v2_norm_nullnonsense

V2_TARGETS="con_artist drill_sergeant farmer kindergarten_teacher politician professor street_hustler surgeon tech_ceo therapist"
NULL_NONSENSE="null nonsense"

echo "[steer-null-nonsense] $(date -Iseconds) ===== 8: steered generation (sources=null,nonsense; targets=v2 personas) ====="
python pipeline/8_steered_generation.py \
    --model         $MODEL \
    --vectors-dir   $V2_VECTORS \
    --output-dir    $OUT \
    --source-personas $NULL_NONSENSE \
    --target-personas $V2_TARGETS \
    --layer 22 \
    --alpha 2.0 \
    --n-questions 5 \
    --normalize

echo "[steer-null-nonsense] $(date -Iseconds) ===== p1: classifier residue on null/nonsense-steered outputs ====="
python pipeline/p1_classifier_on_steered.py \
    --steered-dir    $OUT \
    --classifier-dir outputs/gemma-2-27b-it/v2/classifier \
    --output-dir     $RESIDUE_OUT

echo "[steer-null-nonsense] $(date -Iseconds) DONE"
