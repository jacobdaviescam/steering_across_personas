#!/usr/bin/env bash
# Upload the trajectory outputs (layer-subset activations, means, analyses, gate report) to the private HF dataset.
set -uo pipefail
cd /root/work/repo; set -a; . /root/work/.env; set +a; export HF_HOME=/root/work/hf-cache
REPO=jacobdavies/iclr2027-conditionality-results
.venv/bin/python - <<PY
from huggingface_hub import HfApi
HfApi().create_repo("$REPO", repo_type="dataset", private=True, exist_ok=True); print("repo ok")
PY
.venv/bin/hf upload-large-folder "$REPO" /root/work/outputs/gpt-oss-120b --repo-type dataset --num-workers 8 2>&1 | tail -3
echo "[$(date)] UPLOAD exit=$?"
