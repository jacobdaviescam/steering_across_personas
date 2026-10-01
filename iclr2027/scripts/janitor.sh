#!/usr/bin/env bash
# Remove each extra-queue model cache once its extraction stage has printed exit=; never touch OLMo/Qwen3-8B while master runs.
declare -A C=( ["Qwen3-14B-Base raw"]="models--Qwen--Qwen3-14B-Base" ["Qwen3-14B raw"]="models--Qwen--Qwen3-14B" ["Qwen2.5-32B-Instruct eval"]="models--Qwen--Qwen2.5-32B-Instruct" ["Gemma-3-27B-IT eval"]="models--google--gemma-3-27b-it" )
while true; do
  for k in "${!C[@]}"; do
    dir=/workspace/hf-cache/hub/${C[$k]}
    if [ -d "$dir" ] && grep -aA1 "START $k" /workspace/iclr2026/extra.log | grep -aq "exit="; then rm -rf "$dir" && echo "[$(date)] removed ${C[$k]}"; fi
  done
  grep -aq "EXTRA DONE" /workspace/iclr2026/extra.log && break; sleep 120
done
