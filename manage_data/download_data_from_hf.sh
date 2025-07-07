#!/usr/bin/env bash
set -euo pipefail
echo "HF token length: ${#HF_TOKEN}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# load optional .env without clobbering existing vars
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r k v; do
    [[ $k =~ ^\s*# ]] && continue
    [[ -z $k ]]          && continue
    [[ -z ${!k:-} ]] && export "$k=$v"
  done < "$ENV_FILE"
fi

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
