#!/bin/bash

# Check if micromamba environment is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Error: No micromamba environment activated. Please run: micromamba activate bonting-id"
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV}" != "bonting-id" ]]; then
    echo "Error: Wrong environment activated. Please run: micromamba activate bonting-id"
    exit 1
fi

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

# Check if data directory exists
if [ ! -d "../data" ]; then
    echo "Error: data directory not found"
    exit 1
fi

# Upload data directory
echo "Uploading data directory to $HF_REPO_ID..."
huggingface-cli upload "$HF_REPO_ID" "../data/" --token "$HF_TOKEN" --repo-type dataset

echo "Data uploaded successfully to $HF_REPO_ID"