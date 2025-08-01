#!/bin/bash

# Determine script directory & project root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

# Create directories if they don't exist
YT_DIR="${PROJECT_ROOT}/data/bonting-identification/source_videos/youtube_full"
mkdir -p "${YT_DIR}"

yt-dlp --no-cache-dir --cookies "${PROJECT_ROOT}/cookies.txt" -P "${YT_DIR}" https://www.youtube.com/watch?v=9sWtw_EtHKI