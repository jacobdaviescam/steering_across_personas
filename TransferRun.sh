#!/usr/bin/env bash
# Run x5 + x6 for both CAA and IV probes.
# Assumes venv is already active and IV vectors exist at v2/vectors/.
# If IV vectors are missing, run:
#   python pipeline/3_vectors.py --activations-dir outputs/gemma-2-27b-it/v2/activations --output-dir outputs/gemma-2-27b-it/v2/vectors

set -euo pipefail

MODEL_DIR=outputs/gemma-2-27b-it/v2
LAYER=22

echo "==> CAA probe -> IV activations (x5)"
python pipeline/x5_probe_cross_transfer.py \
    --activations-dir "$MODEL_DIR/activations" \
    --probes-dir "$MODEL_DIR/caa_probes/probes_pkl" \
    --output-dir "$MODEL_DIR/caa_probes" \
    --layer "$LAYER"

echo "==> CAA probe correlation (x6)"
python pipeline/x6_correlation.py \
    --matrix-dir "$MODEL_DIR/caa_probes" \
    --vectors-dir "$MODEL_DIR/caa_vectors" \
    --output-dir "$MODEL_DIR/x6_correlation_caa" \
    --layer "$LAYER"

echo "==> IV probe -> IV activations (x5)"
python pipeline/x5_probe_cross_transfer.py \
    --activations-dir "$MODEL_DIR/activations" \
    --probes-dir "$MODEL_DIR/iv_probes/probes_pkl" \
    --output-dir "$MODEL_DIR/iv_probes" \
    --layer "$LAYER"

echo "==> IV probe correlation (x6)"
python pipeline/x6_correlation.py \
    --matrix-dir "$MODEL_DIR/iv_probes" \
    --vectors-dir "$MODEL_DIR/vectors" \
    --output-dir "$MODEL_DIR/x6_correlation_iv" \
    --layer "$LAYER"

echo ""
echo "==> Summaries"
echo "--- CAA x5 ---"
cat "$MODEL_DIR/caa_probes/cross_transfer_summary.json"
echo ""
echo "--- CAA x6 ---"
cat "$MODEL_DIR/x6_correlation_caa/summary.json"
echo ""
echo "--- IV x5 ---"
cat "$MODEL_DIR/iv_probes/cross_transfer_summary.json"
echo ""
echo "--- IV x6 ---"
cat "$MODEL_DIR/x6_correlation_iv/summary.json"
echo ""
