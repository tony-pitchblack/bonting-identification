#!/bin/bash
mkdir -p tmp
mkdir -p data
curl -L -o tmp/cow-eartag-recognition-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/fandaoerji/cow-eartag-recognition-dataset

unzip tmp/cow-eartag-recognition-dataset.zip -d data/
unzipped_dir=$(unzip -Z -1 tmp/cow-eartag-recognition-dataset.zip | head -1 | cut -d/ -f1)
mv "data/$unzipped_dir" data/CEID-R
