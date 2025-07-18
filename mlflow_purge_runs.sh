#!/bin/bash
# WARNING: does not purge corresponding models, needs debugging
source .env
mlflow gc \
  --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
  --artifacts-destination $MLFLOW_ARTIFACT_ROOT \
  --older-than 0d