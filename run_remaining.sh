#!/usr/bin/env bash
# Run remaining pipeline: IV from step 4 + full CAA
set -euo pipefail

export HF_TOKEN=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
source .env 2>/dev/null || true

echo "=== IV: steps 4 onwards (analysis + robustness) ==="
./run.sh google/gemma-2-27b-it --iv --from 4

echo ""
echo "=== CAA: full pipeline ==="
./run.sh google/gemma-2-27b-it --caa

echo ""
echo "=== ALL DONE ==="
