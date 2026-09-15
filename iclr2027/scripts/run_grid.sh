#!/usr/bin/env bash
# ICLR 2027 Result C + Gemma-2 standardisation rerun. Idempotent: existing files are skipped.
set -uo pipefail
cd /workspace/iclr2026/repo
export HF_HOME=/workspace/hf-cache WANDB_MODE=disabled
set -a; . ./.env; set +a
M=google/gemma-2-27b-it
OUT=/workspace/iclr2026/outputs/gemma-2-27b-it
echo "[$(date)] START eval-context grid"
uv run --no-sync python pipeline/2c_caa_activations.py --model $M \
  --output-dir $OUT/caa_activations_evalctx \
  --personas auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness \
  --paraphrase-variants 4 --paraphrase-questions 100 --batch-size 32
echo "[$(date)] eval-context grid exit=$?"
echo "[$(date)] START original 12-context rerun (current code)"
uv run --no-sync python pipeline/2c_caa_activations.py --model $M \
  --output-dir $OUT/caa_activations_v3 \
  --paraphrase-variants 4 --paraphrase-questions 100 --batch-size 32
echo "[$(date)] original rerun exit=$?"
echo "[$(date)] vectors"
uv run --no-sync python pipeline/3_vectors.py --activations-dir $OUT/caa_activations_evalctx --output-dir $OUT/caa_vectors_evalctx
uv run --no-sync python pipeline/3_vectors.py --activations-dir $OUT/caa_activations_v3 --output-dir $OUT/caa_vectors_v3
echo "[$(date)] ALL DONE"
echo "[$(date)] Result C analyses against same-run null/nonsense"
A=/workspace/iclr2026/analysis/evalctx_L22_v3ref; E=$OUT/caa_activations_evalctx; R=$OUT/caa_activations_v3
CTX="auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness"
uv run --no-sync python /workspace/iclr2026/analyze_evalctx.py --acts $E --ref $R --layer 22 --out $A --contexts $CTX --n-boot 50
uv run --no-sync python /workspace/iclr2026/axis_projection.py --acts $E $R --ref $R --axis /workspace/iclr2026/laptop_outputs_backup/gemma-2-27b-it/axis.pt --layer 22 --contexts $CTX nonsense farmer politician therapist drill_sergeant street_hustler professor tech_ceo kindergarten_teacher surgeon con_artist --out $A
uv run --no-sync python /workspace/iclr2026/paired_boot.py --acts $E --ref $R --layer 22 --contexts $CTX --out $A --n-boot 200
echo "[$(date)] ANALYSES DONE"
