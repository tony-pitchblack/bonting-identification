if [[ "$1" == "--roboflow" ]]; then
  shift
  python run_tracking.py \
    'data/HF_dataset/source_videos/youtube_segments/Milking R Dairy - Nedap CowControl with the SmartTag Ear merged 4 cuts.webm' \
    --roboflow-ckpt 'cow-test-yeo0m/cows-gyup1/2' \
    --class-name cow "$@"
else
  python run_tracking.py \
    'data/HF_dataset/source_videos/youtube_segments/Milking R Dairy - Nedap CowControl with the SmartTag Ear merged 4 cuts.webm' \
    --roboflow-ckpt 'cow-test-yeo0m/cows-gyup1/2' \
    --class-name ear-tag "$@"
fi