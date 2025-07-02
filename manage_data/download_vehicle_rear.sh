#!/bin/bash

# Determine script directory & project root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
TMP_DIR="${PROJECT_ROOT}/tmp/vehicle_rear"
DATA_DIR="${PROJECT_ROOT}/data/vehicle_rear"

# Default settings
INCLUDE_DATA=false
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --include-data)
            INCLUDE_DATA=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--include-data] [--force]"
            exit 1
            ;;
    esac
done

mkdir -p "${TMP_DIR}"
mkdir -p "${DATA_DIR}"

# Function to check if directory has content
has_content() {
    [ -n "$(ls -A $1 2>/dev/null)" ]
}

# Function to download and extract
download_and_extract() {
    local name=$1
    local url=$2
    local archive="${TMP_DIR}/$name.tgz"
    local extract_dir="${DATA_DIR}"
    
    if [ "$FORCE" = true ] || [ ! -f "$archive" ]; then
        echo "Downloading vehicle rear dataset $name..."
        curl -L -o "$archive" "$url"
    else
        echo "Archive $archive already exists, skipping download. Use --force to override."
    fi

    if [ "$FORCE" = true ] || ! has_content "$extract_dir/$name"; then
        echo "Extracting $name..."
        tar -xzf "$archive" -C "$extract_dir/"
    else
        echo "Content already exists in $extract_dir/$name, skipping extraction. Use --force to override."
    fi
}

# Always download videos
download_and_extract "videos" "https://www.inf.ufpr.br/vri/databases/vehicle-reid/videos.tgz"

# Download data only if --include-data is specified
if [ "$INCLUDE_DATA" = true ]; then
    download_and_extract "data" "https://www.inf.ufpr.br/vri/databases/vehicle-reid/data.tgz"
fi

echo "Vehicle rear dataset download and extraction complete!" 