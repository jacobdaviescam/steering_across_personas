#!/usr/bin/env bash
set -uo pipefail
cd /workspace/iclr2026/repo
O=/workspace/iclr2026/outputs; T="uv run --no-sync python /workspace/iclr2026/thin_activations.py"
$T --dir $O/gemma-2-27b-it/caa_activations_evalctx --keep 5,10,15,20,22,25,30,35,40
$T --dir $O/gemma-2-27b-it/caa_activations_v3      --keep 5,10,15,20,22,25,30,35,40
$T --dir $O/Llama-3.1-8B-Instruct/caa_activations_evalctx --keep 8,12,15,20,24,28
$T --dir $O/Llama-3.1-8B-Instruct/caa_activations_v3      --keep 8,12,15,20,24,28
$T --dir $O/Qwen3-32B/caa_activations_evalctx --keep 16,24,31,38,42,49,56
$T --dir $O/Qwen3-32B/caa_activations_v3      --keep 16,24,31,38,42,49,56
echo "[$(date)] THIN DONE"; du -sh $O/* 2>/dev/null
