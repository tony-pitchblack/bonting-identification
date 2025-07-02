#!/bin/bash

# Check if micromamba environment is activated
# Ensure the required micromamba environment is active (activate automatically if not)
if [[ "${CONDA_DEFAULT_ENV}" != "bonting-id" ]]; then
    if ! command -v micromamba &> /dev/null; then
        echo "Error: micromamba command not found"
        exit 1
    fi
    eval "$(micromamba shell hook -s bash)"
    micromamba activate bonting-id
fi

# Determine script directory & project root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

# Create directories if they don't exist
YT_DIR="${PROJECT_ROOT}/data/HF_dataset/source_videos/youtube_full"
mkdir -p "${YT_DIR}"

yt-dlp --no-cache-dir --cookies "${PROJECT_ROOT}/cookies.txt" -P "${YT_DIR}" https://www.youtube.com/watch?v=9sWtw_EtHKI