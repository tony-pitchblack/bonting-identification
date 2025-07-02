#!/bin/bash
mkdir -p tmp
mkdir -p data
curl -L -o tmp/cow-eartag-detection-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/fandaoerji/cow-eartag-detection-dataset

unzip tmp/cow-eartag-detection-dataset.zip -d data/
unzipped_dir=$(unzip -Z -1 tmp/cow-eartag-detection-dataset.zip | head -1 | cut -d/ -f1)
mv "data/$unzipped_dir" data/CEID-D
