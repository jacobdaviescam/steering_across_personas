#!/usr/bin/env bash
# Extra queue, 15 Sep: Qwen3-14B base-vs-instruct raw pair; Qwen2.5-32B-Instruct and Gemma-3-27B-IT eval contexts + controls.
set -uo pipefail
cd /workspace/iclr2026/repo
export HF_HOME=/workspace/hf-cache WANDB_MODE=disabled
set -a; . ./.env; set +a
O=/workspace/iclr2026/outputs; A=/workspace/iclr2026/analysis
CTX="auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness"
run() { echo "[$(date)] START $1"; shift; "$@"; echo "[$(date)] exit=$?"; }
X="uv run --no-sync python pipeline/2c_caa_activations.py --persona-placement user --paraphrase-variants 4 --paraphrase-questions 100 --batch-size 32"
an() { # model-dir layer
  run "analysis $1 L$2" uv run --no-sync python /workspace/iclr2026/analyze_evalctx.py --acts $O/$1/caa_activations_evalctx --ref $O/$1/caa_activations_evalctx --layer $2 --contexts $CTX --out $A/$1_evalctx_L$2 --n-boot 50
  run "paired $1 L$2" uv run --no-sync python /workspace/iclr2026/paired_boot.py --acts $O/$1/caa_activations_evalctx --ref $O/$1/caa_activations_evalctx --layer $2 --contexts $CTX --out $A/$1_evalctx_L$2 --n-boot 200
}
# 1. Qwen3-14B pair (40 layers): keep 10,14,18,20,24,28,32 ; analysis at 20 (0.5)
for M in Qwen3-14B-Base Qwen3-14B; do
  run "$M raw eval+controls" $X --model Qwen/$M --raw-format --output-dir $O/$M-raw/caa_activations_evalctx --personas $CTX null nonsense --save-layers 10,14,18,20,24,28,32
  an $M-raw 20
done
# 2. Qwen2.5-32B-Instruct (64 layers): keep 16,24,31,38,42,49 ; analysis at 31
run "Qwen2.5-32B-Instruct eval+controls" $X --model Qwen/Qwen2.5-32B-Instruct --output-dir $O/Qwen2.5-32B-Instruct/caa_activations_evalctx --personas $CTX null nonsense --save-layers 16,24,31,38,42,49
an Qwen2.5-32B-Instruct 31; an Qwen2.5-32B-Instruct 42
# 3. Gemma-3-27B-IT (62 layers): keep 15,22,27,31,34,40 ; analysis at 31 (the spring run's layer)
run "Gemma-3-27B-IT eval+controls" $X --model google/gemma-3-27b-it --output-dir $O/gemma-3-27b-it/caa_activations_evalctx --personas $CTX null nonsense --save-layers 15,22,27,31,34,40
an gemma-3-27b-it 31
echo "[$(date)] EXTRA DONE"
