#!/bin/bash
# Determine script directory & project root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
TMP_DIR="${PROJECT_ROOT}/tmp"
DATA_DIR="${PROJECT_ROOT}/data"

# Ensure directories
mkdir -p "${TMP_DIR}"
mkdir -p "${DATA_DIR}"

# Download archive
curl -L -o "${TMP_DIR}/cow-eartag-recognition-dataset.zip" \
  https://www.kaggle.com/api/v1/datasets/download/fandaoerji/cow-eartag-recognition-dataset

# Extract and move
unzip "${TMP_DIR}/cow-eartag-recognition-dataset.zip" -d "${DATA_DIR}/"
unzipped_dir=$(unzip -Z -1 "${TMP_DIR}/cow-eartag-recognition-dataset.zip" | head -1 | cut -d/ -f1)
mv "${DATA_DIR}/${unzipped_dir}" "${DATA_DIR}/CEGD-R"
