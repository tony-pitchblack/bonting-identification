#!/usr/bin/env bash
# Fail on error and unset variables
set -euo pipefail

# Initialize and update all submodules (including nested ones)
git submodule update --init

# Install submodules in editable mode inside the activated environment
pip install mindspore # required by mindocr
pip install -e mindocr
pip install -e dl-ocr-bench