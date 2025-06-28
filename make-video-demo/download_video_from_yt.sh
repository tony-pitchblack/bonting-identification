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

yt-dlp --cookies ../cookies.txt -P ../videos/source/ https://www.youtube.com/watch?v=9sWtw_EtHKI