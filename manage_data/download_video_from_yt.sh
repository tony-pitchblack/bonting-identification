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

# Create directories if they don't exist
mkdir -p "../data/HF_dataset/source_videos/youtube_full"

yt-dlp --no-cache-dir --cookies ../cookies.txt -P ../data/HF_dataset/source_videos/youtube_full/ https://www.youtube.com/watch?v=9sWtw_EtHKI