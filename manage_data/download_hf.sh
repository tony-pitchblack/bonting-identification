#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Optional .env loading (contains HF_TOKEN or SSH settings)
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

GITMODULES_FILE="${PROJECT_ROOT}/.gitmodules"
if [[ ! -f "$GITMODULES_FILE" ]]; then
  echo "No .gitmodules found. Aborting." >&2
  exit 1
fi

# Collect submodules pointing to Hugging Face dataset repositories
mapfile -t HF_MODULES < <(
  git config -f "$GITMODULES_FILE" --get-regexp 'submodule\..*\.url' | \
  while read -r key url; do
    if [[ "$url" == *"huggingface.co"* || "$url" == *"@hf.co:"* ]]; then
      mod="${key#submodule.}"
      mod="${mod%.url}"
      echo "$mod|$url"
    fi
  done
)

if [[ ${#HF_MODULES[@]} -eq 0 ]]; then
  echo "No Hugging Face dataset submodules found in .gitmodules."
  exit 0
fi

echo "Updating Hugging Face dataset repositories…"
for entry in "${HF_MODULES[@]}"; do
  IFS='|' read -r mod url <<< "$entry"
  path=$(git config -f "$GITMODULES_FILE" --get "submodule.${mod}.path")
  local_dir="${PROJECT_ROOT}/${path}"

  if [[ -d "${local_dir}/.git" ]]; then
    echo "Pulling latest changes for $path"
    git -C "$local_dir" pull --ff-only
  else
    echo "Cloning $url into $path"
    git clone "$url" "$local_dir"
  fi
done

echo "Datasets are up to date."