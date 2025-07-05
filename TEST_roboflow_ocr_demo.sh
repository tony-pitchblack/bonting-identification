#!/usr/bin/env bash
set -euo pipefail

MAMBA="$HOME/.local/bin/micromamba"
# Initialize micromamba for this shell session and activate the desired environment
if [ -x "$MAMBA" ]; then
  eval "$($MAMBA shell hook --shell bash)"
  micromamba activate bonting-id
else
  echo "micromamba executable not found at $MAMBA" >&2
  exit 1
fi

if ! inference server status 2>&1 | grep -q 'Status: running'; then
  inference server start
fi
python roboflow_ocr_demo.py "$@" 