#!/usr/bin/env bash
# Stops the pod only when no job session remains (exact-name match). Sessions considered jobs: master dl probe.
sleep 120
while tmux list-sessions -F '#S' 2>/dev/null | grep -xqE 'master|dl|probe|extra|thin'; do sleep 120; done
echo "[$(date)] no job sessions left; stopping pod ds5ng3tf34seg9"
cd /workspace/iclr2026/repo && set -a && . ./.env && set +a && runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1
runpodctl stop pod ds5ng3tf34seg9
