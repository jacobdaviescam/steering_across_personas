#!/usr/bin/env bash
set -euo pipefail
cd /workspace/steering_across_personas
set -a; source .env; set +a
export PATH=/workspace/venv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
MODEL=google/gemma-2-27b-it
ROOT=outputs/gemma-2-27b-it/v2

echo "[t1] $(date -Iseconds) ===== n1: naturalistic generation ====="
python pipeline/n1_naturalistic_generate.py \
    --model $MODEL \
    --output-dir $ROOT/naturalistic/responses \
    --n-questions 10

echo "[t1] $(date -Iseconds) ===== 2: activation extraction ====="
python pipeline/2_activations.py \
    --model $MODEL \
    --responses-dir $ROOT/naturalistic/responses \
    --output-dir    $ROOT/naturalistic/activations

echo "[t1] $(date -Iseconds) ===== n3: Claude judging via OpenRouter ====="
python pipeline/n3_naturalistic_judge.py \
    --responses-dir $ROOT/naturalistic/responses \
    --output-dir    $ROOT/naturalistic/judged \
    --max-workers 8

echo "[t1] $(date -Iseconds) ===== n4: build new Fig 3 ====="
python pipeline/n4_naturalistic_eval.py \
    --judged-dir       $ROOT/naturalistic/judged \
    --activations-dir  $ROOT/naturalistic/activations \
    --probes-dir       $ROOT/caa_probes/probes_pkl \
    --vectors-dir      $ROOT/caa_vectors \
    --output-dir       $ROOT/naturalistic/figures
echo "[t1] $(date -Iseconds) DONE"
