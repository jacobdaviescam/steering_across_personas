#!/usr/bin/env bash
set -euo pipefail
cd /workspace/steering_across_personas

export PATH=/workspace/venv/bin:$PATH
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/.pipcache
export XDG_CACHE_HOME=/workspace/.cache
export HF_HOME=/workspace/.cache/huggingface
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"

echo "[reinstall] $(date -Iseconds) starting"
python -V
which pip

echo "[reinstall] uninstalling current torch/torchvision/vllm and stale cu13 nvidia libs"
pip uninstall -y vllm torch torchvision torchaudio torch-c-dlpack-ext flashinfer-python flashinfer-cubin || true
rm -rf /workspace/venv/lib/python3.10/site-packages/nvidia/cu13 || true

echo "[reinstall] ensuring uv is available"
pip install -U uv

echo "[reinstall] installing vllm==0.12.0 (cu128 default) via uv"
uv pip install --python /workspace/venv/bin/python --force-reinstall \
    "vllm==0.12.0" --torch-backend=cu128

echo "[reinstall] verifying torch/cuda + vllm._C"
python - <<'PY'
import torch, vllm
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
import vllm._C  # noqa: F401
print("vllm._C loaded OK")
PY

touch /workspace/steering_across_personas/.vllm_installed
echo "[reinstall] $(date -Iseconds) DONE"
