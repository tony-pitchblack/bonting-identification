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

# Build download job list: each line "repo_id|subpath" (subpath may be empty)
mapfile -t DOWNLOAD_JOBS < <(
python - "$CONFIG_FILE" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
for item in cfg.get('hf_repo_ids', []):
    if isinstance(item, str):
        print(f"{item}|")
    elif isinstance(item, dict):
        for repo, subpaths in item.items():
            if not subpaths:
                print(f"{repo}|")
            else:
                for sp in subpaths:
                    print(f"{repo}|{sp}")
PY
)

DATA_BASE_DIR="${PROJECT_ROOT}/data"
mkdir -p "$DATA_BASE_DIR"

for JOB in "${DOWNLOAD_JOBS[@]}"; do
  IFS='|' read -r REPO_ID SUBPATH <<< "$JOB"
  LOCAL_DIR="${DATA_BASE_DIR}/$(basename "$REPO_ID")"
  mkdir -p "$LOCAL_DIR"

  if [[ -n "$SUBPATH" ]]; then
    echo "Downloading $REPO_ID/$SUBPATH to $LOCAL_DIR …"
    huggingface-cli download "$REPO_ID" \
      --repo-type dataset \
      --local-dir "$LOCAL_DIR" \
      --token "$HF_TOKEN" \
      --include "$SUBPATH/*" \
      --force-download
  else
    echo "Downloading entire dataset $REPO_ID to $LOCAL_DIR …"
    huggingface-cli download "$REPO_ID" \
      --repo-type dataset \
      --local-dir "$LOCAL_DIR" \
      --token "$HF_TOKEN" \
      --force-download
  fi
  echo "✔ Finished $REPO_ID ${SUBPATH:+($SUBPATH)}"
done
