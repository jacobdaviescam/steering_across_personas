#!/usr/bin/env bash
set -uo pipefail
cd /workspace/iclr2026/repo
O=/workspace/iclr2026/outputs; A=/workspace/iclr2026/analysis
CTX="auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness"
for spec in "Qwen3-32B 31" "Qwen3-32B 42" "Qwen3-32B 49" "Llama-3.1-8B-Instruct 15" "Llama-3.1-8B-Instruct 20"; do
  set -- $spec; M=$1; L=$2
  echo "[$(date)] $M layer $L"
  uv run --no-sync python /workspace/iclr2026/analyze_evalctx.py --acts $O/$M/caa_activations_evalctx --ref $O/$M/caa_activations_v3 --layer $L --contexts $CTX --out $A/${M}_evalctx_L$L --n-boot 50
  uv run --no-sync python /workspace/iclr2026/paired_boot.py --acts $O/$M/caa_activations_evalctx --ref $O/$M/caa_activations_v3 --layer $L --contexts $CTX --out $A/${M}_evalctx_L$L --n-boot 200
done
echo "[$(date)] ANALYSES2 DONE"
