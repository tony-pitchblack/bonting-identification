#!/bin/bash

# Determine script directory to reliably locate the project root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Load environment variables from .env
set -a
source "${PROJECT_ROOT}/.env"
set +a

# Create data directory if it doesn't exist
DATA_DIR="${PROJECT_ROOT}/data/HF_dataset"
mkdir -p "${DATA_DIR}"

# Download data directory
echo "Downloading data from $HF_REPO_ID..."
huggingface-cli download "$HF_REPO_ID" \
    --repo-type dataset \
    --local-dir "${DATA_DIR}" \
    --token "$HF_TOKEN" \
    --force-download
echo "Data downloaded successfully to ${DATA_DIR}" 
