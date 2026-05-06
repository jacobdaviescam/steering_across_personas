#!/usr/bin/env bash
set -euo pipefail
cd /workspace/steering_across_personas
set -a; source .env; set +a
export PATH=/workspace/venv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ROOT=outputs/gemma-2-27b-it/v2

echo "[t3] $(date -Iseconds) starting persona-residue classifier"
python pipeline/p1_classifier_on_steered.py \
    --steered-dir   outputs/gemma-2-27b-it/steered_responses_alpha2 \
    --classifier-dir $ROOT/classifier \
    --output-dir     $ROOT/persona_residue
echo "[t3] $(date -Iseconds) DONE"
