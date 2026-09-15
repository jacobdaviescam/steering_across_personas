#!/usr/bin/env bash
set -uo pipefail
cd /workspace/iclr2026/repo
export HF_HOME=/workspace/hf-cache WANDB_MODE=disabled
set -a; . ./.env; set +a
OUT=/workspace/iclr2026/outputs; M=Qwen/Qwen3-8B; S=Qwen3-8B
CTX="auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness"
X="uv run --no-sync python pipeline/2c_caa_activations.py --persona-placement user --paraphrase-variants 4 --paraphrase-questions 100 --batch-size 32"
echo "[$(date)] START Qwen3-8B 12-context"; $X --model $M --output-dir $OUT/$S/caa_activations_v3; echo "[$(date)] exit=$?"
echo "[$(date)] START Qwen3-8B eval-context"; $X --model $M --output-dir $OUT/$S/caa_activations_evalctx --personas $CTX; echo "[$(date)] exit=$?"
for L in 18 24; do
  uv run --no-sync python /workspace/iclr2026/analyze_evalctx.py --acts $OUT/$S/caa_activations_evalctx --ref $OUT/$S/caa_activations_v3 --layer $L --contexts $CTX --out /workspace/iclr2026/analysis/${S}_evalctx_L$L --n-boot 50
  uv run --no-sync python /workspace/iclr2026/paired_boot.py --acts $OUT/$S/caa_activations_evalctx --ref $OUT/$S/caa_activations_v3 --layer $L --contexts $CTX --out /workspace/iclr2026/analysis/${S}_evalctx_L$L --n-boot 200
done
echo "[$(date)] QWEN8B DONE"
