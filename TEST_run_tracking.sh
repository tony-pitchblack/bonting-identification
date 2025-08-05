#!/bin/bash

# Default class names based on mode
DEFAULT_ROBOFLOW_CLASS="ear-tag"
DEFAULT_ULTRALYTICS_CLASS="cow"

# Parse arguments to check for --roboflow and --class-name
ROBOFLOW_MODE=false
CLASS_NAME=""
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --roboflow)
      ROBOFLOW_MODE=true
      shift
      ;;
    --class-name)
      CLASS_NAME="$2"
      shift 2
      ;;
    *)
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

# Set default class name if not specified
if [[ -z "$CLASS_NAME" ]]; then
  if [[ "$ROBOFLOW_MODE" == true ]]; then
    CLASS_NAME="$DEFAULT_ROBOFLOW_CLASS"
  else
    CLASS_NAME="$DEFAULT_ULTRALYTICS_CLASS"
  fi
fi

# Build the command
if [[ "$ROBOFLOW_MODE" == true ]]; then
  python run_tracking.py \
    'data/bonting-identification/source_videos/youtube_segments/Milking R Dairy - Nedap CowControl with the SmartTag Ear merged 4 cuts.webm' \
    --roboflow-ckpt 'cow-test-yeo0m/cows-gyup1/2' \
    --class-name "$CLASS_NAME" \
    "${OTHER_ARGS[@]}"
else
  python run_tracking.py \
    'data/bonting-identification/source_videos/youtube_segments/Milking R Dairy - Nedap CowControl with the SmartTag Ear merged 4 cuts.webm' \
    --roboflow-ckpt 'cow-test-yeo0m/cows-gyup1/2' \
    --class-name "$CLASS_NAME" \
    "${OTHER_ARGS[@]}"
fi