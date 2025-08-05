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

for REPO_ID in "${HF_REPO_IDS[@]}"; do
  LOCAL_DIR="${DATA_BASE_DIR}/$(basename "$REPO_ID")"
  if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "Warning: data directory not found for $REPO_ID at $LOCAL_DIR, skipping."
    continue
  fi

  echo "Uploading $LOCAL_DIR to $REPO_ID …"
  huggingface-cli upload "$REPO_ID" \
    "${LOCAL_DIR}/" \
    --token "$HF_TOKEN" \
    --repo-type dataset \
    --delete "*"
  echo "✔ Uploaded $LOCAL_DIR to $REPO_ID"
done
