#!/bin/bash

# Determine script directory to reliably locate the project root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Load environment variables from .env
set -a
source "${PROJECT_ROOT}/.env"
set +a

# Check if required environment variables are set
if [ -z "$HF_TOKEN" ] || [ -z "$HF_REPO_ID" ]; then
    echo "Error: HF_TOKEN and HF_REPO_ID must be set in .env file"
    exit 1
fi

DATA_DIR="${PROJECT_ROOT}/data/HF_dataset"
# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: data directory not found at ${DATA_DIR}"
    exit 1
fi

# Upload data directory
echo "Uploading data directory to $HF_REPO_ID..."
huggingface-cli \
    upload \
    "$HF_REPO_ID" \
    "${DATA_DIR}/" \
    --token "$HF_TOKEN" \
    --repo-type dataset \
    --delete "*"

echo "Data uploaded successfully to $HF_REPO_ID"