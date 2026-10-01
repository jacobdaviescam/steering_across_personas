#!/usr/bin/env bash
# Experiment 2 queue (17 Sep): forced-choice log-odds under evaluation contexts, three models.
# Logits-only extraction (no activations), persona in the user message on every model, 500 questions
# per trait, both answer directions (the pos/neg pair shares its prefix, so agreement is a free check).
# Small models first so results land early; analysis re-run after each model. Idempotent (2c skips done
# cells). No pod stop here, ever: completion is signalled by the marker file $W/EXP2_DONE.
set -uo pipefail
W=${W:-/root/work}
cd $W/repo
export PATH=$HOME/.local/bin:$PATH HF_HOME=$W/hf-cache WANDB_MODE=disabled
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
set -a; . $W/.env; set +a
CTX="auto_grader human_evaluator simulated_env real_deployment under_evaluation unobserved coding_harness null nonsense"
PY=.venv/bin/python
run() { echo "[$(date -u +%FT%TZ)] START $1"; shift; "$@"; echo "[$(date -u +%FT%TZ)] exit=$?"; }
X="$PY pipeline/2c_caa_activations.py --logits-only --persona-placement user --batch-size ${BS:-32}"
MODELS=${MODELS:-"meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B google/gemma-2-27b-it"}

# Preflight: the answer-letter token ids used for the log-odds must be real single tokens on each tokenizer.
run "preflight A/B token ids" $PY - $MODELS <<'PYX'
import sys
from transformers import AutoTokenizer
for m in sys.argv[1:]:
    t = AutoTokenizer.from_pretrained(m)
    ia, ib = t.convert_tokens_to_ids("A"), t.convert_tokens_to_ids("B")
    print(m, "A", ia, repr(t.decode([ia])), "B", ib, repr(t.decode([ib])), "unk", t.unk_token_id)
    assert ia not in (None, t.unk_token_id) and ib not in (None, t.unk_token_id), m
PYX

for M in $MODELS; do
  S=${M##*/}
  run "$S exp2 logits" $X --model $M --personas $CTX --output-dir $W/outputs/$S/exp2_logits
  run "analysis after $S" $PY iclr2027/scripts/exp2_forced_choice.py --root $W/outputs --out $W/analysis/exp2
done
echo "[$(date -u +%FT%TZ)] EXP2 DONE"
touch $W/EXP2_DONE
