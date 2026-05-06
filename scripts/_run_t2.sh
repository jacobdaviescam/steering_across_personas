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

# Wait for t1 to finish naturalistic gen+activations+judge first so a3 has
# something to compare against. We poll the t1 log.
echo "[t2] $(date -Iseconds) waiting for t1 to complete naturalistic judging..."
until grep -q "^\[t1\].*DONE" logs/t1.log 2>/dev/null; do
    sleep 30
done
echo "[t2] $(date -Iseconds) t1 finished — proceeding"

echo "[t2] $(date -Iseconds) ===== a1: generate adversarial questions ====="
python pipeline/a1_generate_adversarial_questions.py \
    --output-dir data/prompts/adversarial \
    --n 10

echo "[t2] $(date -Iseconds) ===== a2: generate adversarial responses ====="
python pipeline/a2_adversarial_generate.py \
    --model $MODEL \
    --questions-dir data/prompts/adversarial \
    --output-dir    $ROOT/adversarial/responses

echo "[t2] $(date -Iseconds) ===== 2: extract adversarial activations ====="
python pipeline/2_activations.py \
    --model $MODEL \
    --responses-dir $ROOT/adversarial/responses \
    --output-dir    $ROOT/adversarial/activations

echo "[t2] $(date -Iseconds) ===== n3: judge adversarial responses ====="
python pipeline/n3_naturalistic_judge.py \
    --responses-dir $ROOT/adversarial/responses \
    --output-dir    $ROOT/adversarial/judged \
    --max-workers 8

echo "[t2] $(date -Iseconds) ===== a3: paired-AUROC scatter ====="
python pipeline/a3_adversarial_analysis.py \
    --naturalistic-judged $ROOT/naturalistic/judged \
    --naturalistic-acts   $ROOT/naturalistic/activations \
    --adversarial-judged  $ROOT/adversarial/judged \
    --adversarial-acts    $ROOT/adversarial/activations \
    --probes-dir          $ROOT/caa_probes/probes_pkl \
    --vectors-dir         $ROOT/caa_vectors \
    --output-dir          $ROOT/adversarial/figures
echo "[t2] $(date -Iseconds) DONE"
