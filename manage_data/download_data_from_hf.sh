#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# load optional .env without clobbering existing vars
ENV_FILE="${PROJECT_ROOT}/.env"
source "$ENV_FILE"

echo "HF_TOKEN: $HF_TOKEN"
echo "HF_REPO_ID: $HF_REPO_ID"

# mandatory vars
: "${HF_TOKEN?Missing HF_TOKEN}"
: "${HF_REPO_ID?Missing HF_REPO_ID}"

DATA_DIR="${PROJECT_ROOT}/data/HF_dataset"
mkdir -p "$DATA_DIR"

echo "Downloading data from $HF_REPO_ID …"
huggingface-cli download "$HF_REPO_ID" \
  --repo-type dataset \
  --local-dir "$DATA_DIR" \
  --token "$HF_TOKEN" \
  --force-download
echo "✔ Data saved to $DATA_DIR"
