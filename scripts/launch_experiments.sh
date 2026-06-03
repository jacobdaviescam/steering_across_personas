#!/usr/bin/env bash
# Launch the three experiment threads in a tmux session.
#
# Sets up:
#   - HF cache on /workspace/.cache
#   - pip / tmp on /workspace
#   - .env loaded (so OPENROUTER_API_KEY etc are available)
#   - venv python on PATH
#
# Windows in the tmux session "psteer":
#   t3   — persona-residue classifier (CPU, fast, runs first)
#   t1   — naturalistic Fig 3 (GPU + OpenRouter)
#   t2   — adversarial cells (CPU then GPU, runs last)
#   mon  — tail of the three log files
#
# Logs land in /workspace/steering_across_personas/logs/{t1,t2,t3}.log
# so the monitor can keep tailing them between Claude turns.

set -euo pipefail

cd /workspace/steering_across_personas

mkdir -p logs

# --- env ---------------------------------------------------------------
set -a
source .env
set +a

export PATH=/workspace/venv/bin:$PATH
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/.pipcache
export XDG_CACHE_HOME=/workspace/.cache
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "OPENROUTER_API_KEY not set after sourcing .env — abort"
    exit 1
fi

MODEL=google/gemma-2-27b-it
ROOT=outputs/gemma-2-27b-it/v2

# --- per-thread runner scripts -----------------------------------------
cat > scripts/_run_t3.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/steering_across_personas
set -a; source .env; set +a
export PATH=/workspace/venv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ROOT=outputs/gemma-2-27b-it/v2

echo "[t3] $(date -Iseconds) starting persona-residue classifier"
python pipeline/p1_classifier_on_steered.py \
    --steered-dir   outputs/gemma-2-27b-it/steered_responses_alpha2 \
    --classifier-dir $ROOT/classifier \
    --output-dir     $ROOT/persona_residue
echo "[t3] $(date -Iseconds) DONE"
EOF
chmod +x scripts/_run_t3.sh

cat > scripts/_run_t1.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/steering_across_personas
set -a; source .env; set +a
export PATH=/workspace/venv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
MODEL=google/gemma-2-27b-it
ROOT=outputs/gemma-2-27b-it/v2

echo "[t1] $(date -Iseconds) ===== n1: naturalistic generation ====="
python pipeline/n1_naturalistic_generate.py \
    --model $MODEL \
    --output-dir $ROOT/naturalistic/responses \
    --n-questions 10

echo "[t1] $(date -Iseconds) ===== 2: activation extraction ====="
python pipeline/2_activations.py \
    --model $MODEL \
    --responses-dir $ROOT/naturalistic/responses \
    --output-dir    $ROOT/naturalistic/activations

echo "[t1] $(date -Iseconds) ===== n3: Claude judging via OpenRouter ====="
python pipeline/n3_naturalistic_judge.py \
    --responses-dir $ROOT/naturalistic/responses \
    --output-dir    $ROOT/naturalistic/judged \
    --max-workers 8

echo "[t1] $(date -Iseconds) ===== n4: build new Fig 3 ====="
python pipeline/n4_naturalistic_eval.py \
    --judged-dir       $ROOT/naturalistic/judged \
    --activations-dir  $ROOT/naturalistic/activations \
    --probes-dir       $ROOT/caa_probes/probes_pkl \
    --vectors-dir      $ROOT/caa_vectors \
    --output-dir       $ROOT/naturalistic/figures
echo "[t1] $(date -Iseconds) DONE"
EOF
chmod +x scripts/_run_t1.sh

cat > scripts/_run_t2.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/steering_across_personas
set -a; source .env; set +a
export PATH=/workspace/venv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
MODEL=google/gemma-2-27b-it
ROOT=outputs/gemma-2-27b-it/v2

# Wait for t1 to finish naturalistic gen+activations+judge first so a3 has
# something to compare against. We poll the t1 log.
echo "[t2] $(date -Iseconds) waiting for t1 to complete naturalistic judging..."
until grep -q "^\[t1\].*DONE" logs/t1.log 2>/dev/null; do
    sleep 30
done
echo "[t2] $(date -Iseconds) t1 finished — proceeding"

echo "[t2] $(date -Iseconds) ===== a1: generate adversarial questions ====="
python pipeline/a1_generate_adversarial_questions.py \
    --output-dir data/prompts/adversarial \
    --n 10

echo "[t2] $(date -Iseconds) ===== a2: generate adversarial responses ====="
python pipeline/a2_adversarial_generate.py \
    --model $MODEL \
    --questions-dir data/prompts/adversarial \
    --output-dir    $ROOT/adversarial/responses

echo "[t2] $(date -Iseconds) ===== 2: extract adversarial activations ====="
python pipeline/2_activations.py \
    --model $MODEL \
    --responses-dir $ROOT/adversarial/responses \
    --output-dir    $ROOT/adversarial/activations

echo "[t2] $(date -Iseconds) ===== n3: judge adversarial responses ====="
python pipeline/n3_naturalistic_judge.py \
    --responses-dir $ROOT/adversarial/responses \
    --output-dir    $ROOT/adversarial/judged \
    --max-workers 8

echo "[t2] $(date -Iseconds) ===== a3: paired-AUROC scatter ====="
python pipeline/a3_adversarial_analysis.py \
    --naturalistic-judged $ROOT/naturalistic/judged \
    --naturalistic-acts   $ROOT/naturalistic/activations \
    --adversarial-judged  $ROOT/adversarial/judged \
    --adversarial-acts    $ROOT/adversarial/activations \
    --probes-dir          $ROOT/caa_probes/probes_pkl \
    --vectors-dir         $ROOT/caa_vectors \
    --output-dir          $ROOT/adversarial/figures
echo "[t2] $(date -Iseconds) DONE"
EOF
chmod +x scripts/_run_t2.sh

# --- tmux session ------------------------------------------------------
SESSION=psteer

if tmux has-session -t $SESSION 2>/dev/null; then
    echo "tmux session $SESSION already exists — kill and restart"
    tmux kill-session -t $SESSION
fi

tmux new-session -d -s $SESSION -n t3 \
    "scripts/_run_t3.sh 2>&1 | tee logs/t3.log; bash"
tmux new-window -t $SESSION -n t1 \
    "scripts/_run_t1.sh 2>&1 | tee logs/t1.log; bash"
tmux new-window -t $SESSION -n t2 \
    "scripts/_run_t2.sh 2>&1 | tee logs/t2.log; bash"
tmux new-window -t $SESSION -n mon \
    "tail -F logs/t3.log logs/t1.log logs/t2.log"

echo "Started tmux session '$SESSION'."
echo "  attach:    tmux attach -t $SESSION"
echo "  list:      tmux ls"
echo "  logs:      tail -F logs/t{1,2,3}.log"
