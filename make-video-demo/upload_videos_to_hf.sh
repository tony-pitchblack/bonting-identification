#!/bin/bash

# Check if micromamba environment is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Error: No micromamba environment activated. Please run: micromamba activate bonting-exp"
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV}" != "bonting-exp" ]]; then
    echo "Error: Wrong environment activated. Please run: micromamba activate bonting-exp"
    exit 1
fi

# Load environment variables from .env
set -a
source ../.env
set +a

# Check if required environment variables are set
if [ -z "$HF_TOKEN" ] || [ -z "$HF_REPO_ID" ]; then
    echo "Error: HF_TOKEN and HF_REPO_ID must be set in .env file"
    exit 1
fi

# Check if videos directory exists
if [ ! -d "../videos" ]; then
    echo "Error: videos directory not found"
    exit 1
fi

# Upload videos directory
huggingface-cli upload "$HF_REPO_ID" "../videos/" --token "$HF_TOKEN"