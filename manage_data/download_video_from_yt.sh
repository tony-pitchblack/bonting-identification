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

# Create directories if they don't exist
mkdir -p "../data/source_videos/youtube_full"

yt-dlp --cookies ../cookies.txt -P ../data/source_videos/youtube_full/ https://www.youtube.com/watch?v=9sWtw_EtHKI