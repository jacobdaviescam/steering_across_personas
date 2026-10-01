#!/usr/bin/env bash
# Master queue, 15 Sep: Qwen3-8B grids; Qwen3-8B base-vs-instruct raw pair; OLMo eval contexts x stages; analyses.
# Layer-subset storage. Idempotent. No pod stop here: the watchdog session does that.
set -uo pipefail
cd /workspace/iclr2026/repo
export HF_HOME=/workspace/hf-cache WANDB_MODE=disabled
set -a; . ./.env; set +a
O=/workspace/iclr2026/outputs; A=/workspace/iclr2026/analysis
CTX="auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness"
run() { echo "[$(date)] START $1"; shift; "$@"; echo "[$(date)] exit=$?"; }
X="uv run --no-sync python pipeline/2c_caa_activations.py --persona-placement user --paraphrase-variants 4 --paraphrase-questions 100 --batch-size 32"
Q8L="9,12,15,18,21,24,27,30"
# 1. Qwen3-8B instruct, chat template
run "Qwen3-8B 12-context"  $X --model Qwen/Qwen3-8B --output-dir $O/Qwen3-8B/caa_activations_v3 --save-layers $Q8L
run "Qwen3-8B eval-context" $X --model Qwen/Qwen3-8B --output-dir $O/Qwen3-8B/caa_activations_evalctx --personas $CTX --save-layers $Q8L
for L in 18 24; do
  run "analysis Qwen3-8B L$L" uv run --no-sync python /workspace/iclr2026/analyze_evalctx.py --acts $O/Qwen3-8B/caa_activations_evalctx --ref $O/Qwen3-8B/caa_activations_v3 --layer $L --contexts $CTX --out $A/Qwen3-8B_evalctx_L$L --n-boot 50
  run "paired Qwen3-8B L$L" uv run --no-sync python /workspace/iclr2026/paired_boot.py --acts $O/Qwen3-8B/caa_activations_evalctx --ref $O/Qwen3-8B/caa_activations_v3 --layer $L --contexts $CTX --out $A/Qwen3-8B_evalctx_L$L --n-boot 200
done
# 2. Base vs instruct, raw format held fixed, eval contexts + null + nonsense
for M in Qwen3-8B-Base Qwen3-8B; do
  run "$M raw eval+controls" $X --model Qwen/$M --raw-format --output-dir $O/$M-raw/caa_activations_evalctx --personas $CTX null nonsense --save-layers $Q8L
  run "analysis $M raw L18" uv run --no-sync python /workspace/iclr2026/analyze_evalctx.py --acts $O/$M-raw/caa_activations_evalctx --ref $O/$M-raw/caa_activations_evalctx --layer 18 --contexts $CTX --out $A/${M}-raw_evalctx_L18 --n-boot 50
  run "paired $M raw L18" uv run --no-sync python /workspace/iclr2026/paired_boot.py --acts $O/$M-raw/caa_activations_evalctx --ref $O/$M-raw/caa_activations_evalctx --layer 18 --contexts $CTX --out $A/${M}-raw_evalctx_L18 --n-boot 200
done
# 3. OLMo evaluation contexts across stages
T="uv run --no-sync python pipeline/t1_trajectory_activations.py --batch-size 32 --save-layers 8,12,15,20 --personas $CTX null nonsense"
run "OLMo eval-ctx raw, base sft dpo instruct" $T --stages base sft dpo instruct --force-raw-format --dir-suffix _evalctx_rawfmt
run "OLMo eval-ctx template, sft dpo instruct"  $T --stages sft dpo instruct --dir-suffix _evalctx
R=/workspace/iclr2026/repo/outputs/OLMo-2-1124-7B
for st in base_evalctx_rawfmt sft_evalctx_rawfmt dpo_evalctx_rawfmt instruct_evalctx_rawfmt sft_evalctx dpo_evalctx instruct_evalctx; do
  [ -d $R/$st/caa_activations ] && run "analysis OLMo $st L15" uv run --no-sync python /workspace/iclr2026/analyze_evalctx.py --acts $R/$st/caa_activations --ref $R/$st/caa_activations --layer 15 --contexts $CTX --out $A/OLMo_${st}_L15 --n-boot 50
done
echo "[$(date)] MASTER DONE"
