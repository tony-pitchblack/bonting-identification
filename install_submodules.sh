#!/usr/bin/env bash
# Fail on error and unset variables
set -euo pipefail

# Initialize and update all submodules (including nested ones)
git submodule update --init