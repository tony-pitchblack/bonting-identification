#!/usr/bin/env bash
set -euo pipefail

# Determine script and project directories
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Load optional .env without clobbering existing vars
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

# Ensure HF_TOKEN is provided
: "${HF_TOKEN?Missing HF_TOKEN}"

CONFIG_FILE="${PROJECT_ROOT}/config.yml"

# Read repo IDs from config.yml (requires yq or python)
if command -v yq >/dev/null 2>&1; then
  mapfile -t HF_REPO_IDS < <(yq e '.hf_repo_ids[]' "$CONFIG_FILE")
else
  mapfile -t HF_REPO_IDS < <(
    python - "$CONFIG_FILE" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
for repo in data.get('hf_repo_ids', []):
    print(repo)
PY
  )
fi

DATA_BASE_DIR="${PROJECT_ROOT}/data"
mkdir -p "$DATA_BASE_DIR"

for REPO_ID in "${HF_REPO_IDS[@]}"; do
  LOCAL_DIR="${DATA_BASE_DIR}/$(basename "$REPO_ID")"
  echo "Downloading data from $REPO_ID to $LOCAL_DIR …"
  huggingface-cli download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$LOCAL_DIR" \
    --token "$HF_TOKEN" \
    --force-download
  echo "✔ Data saved to $LOCAL_DIR"
done
