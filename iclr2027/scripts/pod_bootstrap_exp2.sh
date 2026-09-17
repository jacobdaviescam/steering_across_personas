#!/usr/bin/env bash
# One-shot environment build for a fresh RunPod container (17 Sep layout: everything under /root/work).
# Before running: scp the laptop's .env (HF_TOKEN) and uv.lock to /root/work/. Idempotent.
set -euo pipefail
W=${W:-/root/work}
mkdir -p $W/hf-cache $W/outputs $W/analysis
cd $W
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)
[ -d repo/.git ] || git clone -q -b iclr2027/results https://github.com/jacobdaviescam/steering_across_personas.git repo
cd repo
git pull -q --ff-only
# The pipeline imports the Lu et al. assistant-axis package by path (sys.path), pinned to the laptop's checkout.
if [ ! -d assistant-axis-ref/.git ]; then
  git clone -q https://github.com/safety-research/assistant-axis.git assistant-axis-ref
  git -C assistant-axis-ref checkout -q a98961956072224eaf244eb289d6c01700b63795
fi
[ -f $W/uv.lock ] && cp $W/uv.lock uv.lock
if [ -f uv.lock ]; then uv sync --frozen --no-dev; else uv sync --no-dev; fi
.venv/bin/python - <<'PY'
import torch, transformers, accelerate
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)
print("gpus", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h $W | tail -1
echo "BOOTSTRAP OK"
