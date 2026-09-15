#!/usr/bin/env bash
# After queue 1: evaluation-context grid across OLMo-2 training stages.
# Primary: raw prompt format on every stage (format held fixed). Secondary: chat template on post-trained stages.
set -uo pipefail
cd /workspace/iclr2026/repo
export HF_HOME=/workspace/hf-cache WANDB_MODE=disabled
set -a; . ./.env; set +a
CTX="auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness null nonsense"
# (wait removed: queue 1 already complete)
echo "[$(date)] queue 1 ended; queue 2 starting"
run() { echo "[$(date)] START $1"; shift; "$@"; echo "[$(date)] exit=$?"; }
T="uv run --no-sync python pipeline/t1_trajectory_activations.py --batch-size 32 --personas $CTX"
run "OLMo eval-context, raw format, base sft dpo instruct" $T --stages base sft dpo instruct --force-raw-format --dir-suffix _evalctx_rawfmt
run "OLMo eval-context, chat template, sft dpo instruct" $T --stages sft dpo instruct --dir-suffix _evalctx
echo "[$(date)] QUEUE2 DONE"
echo "[$(date)] all queues finished; stopping pod aoyrzpwid2llcq"
sleep 30
runpodctl stop pod aoyrzpwid2llcq
