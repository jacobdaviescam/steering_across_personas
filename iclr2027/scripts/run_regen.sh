#!/usr/bin/env bash
set -uo pipefail
cd /workspace/iclr2026/repo
export HF_HOME=/workspace/hf-cache WANDB_MODE=disabled
set -a; . ./.env; set +a
OUT=/workspace/iclr2026/outputs/gemma-2-27b-it
echo "[$(date)] regen therapist/risk_taking"
uv run --no-sync python pipeline/2c_caa_activations.py --model google/gemma-2-27b-it --output-dir $OUT/caa_activations_v3 --personas therapist --traits risk_taking --paraphrase-variants 4 --paraphrase-questions 100 --batch-size 32
echo "[$(date)] regen exit=$?"
uv run --no-sync python pipeline/3_vectors.py --activations-dir $OUT/caa_activations_v3 --output-dir $OUT/caa_vectors_v3
echo "[$(date)] vectors exit=$?"
A=/workspace/iclr2026/analysis/evalctx_L22_v3ref
CTX="auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness"
uv run --no-sync python /workspace/iclr2026/axis_projection.py --acts $OUT/caa_activations_evalctx $OUT/caa_activations_v3 --ref $OUT/caa_activations_v3 --axis /workspace/iclr2026/laptop_outputs_backup/gemma-2-27b-it/axis.pt --layer 22 --contexts $CTX nonsense farmer politician therapist drill_sergeant street_hustler professor tech_ceo kindergarten_teacher surgeon con_artist --out $A
echo "[$(date)] axis exit=$?"
echo "[$(date)] REGEN DONE"
