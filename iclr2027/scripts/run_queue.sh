#!/usr/bin/env bash
# Runs after the Gemma grid: Qwen + Llama standardised grids (user placement, paraphrase companions),
# their evaluation-context grids, and the OLMo raw-format control. Idempotent.
set -uo pipefail
cd /workspace/iclr2026/repo
export HF_HOME=/workspace/hf-cache WANDB_MODE=disabled
set -a; . ./.env; set +a
OUT=/workspace/iclr2026/outputs
CTX="auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness"
while tmux has-session -t grid 2>/dev/null; do sleep 60; done
echo "[$(date)] grid session ended; queue starting"
QOK=$(grep -c "QSMOKE_EXIT=0" /workspace/iclr2026/qsmoke.log 2>/dev/null || true)
run() { echo "[$(date)] START $1"; shift; "$@"; echo "[$(date)] exit=$?"; }
X="uv run --no-sync python pipeline/2c_caa_activations.py --persona-placement user --paraphrase-variants 4 --paraphrase-questions 100 --batch-size 32"
if [ "$QOK" = "1" ]; then
  run "Qwen 12-context grid" $X --model Qwen/Qwen3-32B --output-dir $OUT/Qwen3-32B/caa_activations_v3
else
  echo "[$(date)] SKIP Qwen: smoke test did not pass"
fi
run "Llama 12-context grid" $X --model meta-llama/Llama-3.1-8B-Instruct --output-dir $OUT/Llama-3.1-8B-Instruct/caa_activations_v3
run "Llama eval-context grid" $X --model meta-llama/Llama-3.1-8B-Instruct --output-dir $OUT/Llama-3.1-8B-Instruct/caa_activations_evalctx --personas $CTX
if [ "$QOK" = "1" ]; then
  run "Qwen eval-context grid" $X --model Qwen/Qwen3-32B --output-dir $OUT/Qwen3-32B/caa_activations_evalctx --personas $CTX
fi
run "OLMo raw-format control (sft dpo instruct)" uv run --no-sync python pipeline/t1_trajectory_activations.py --stages sft dpo instruct --force-raw-format --dir-suffix _rawfmt --batch-size 32
for d in Qwen3-32B/caa_activations_v3 Qwen3-32B/caa_activations_evalctx Llama-3.1-8B-Instruct/caa_activations_v3 Llama-3.1-8B-Instruct/caa_activations_evalctx; do
  [ -d $OUT/$d ] && run "vectors $d" uv run --no-sync python pipeline/3_vectors.py --activations-dir $OUT/$d --output-dir $OUT/${d/caa_activations/caa_vectors}
done
echo "[$(date)] QUEUE DONE"
