# Video Demo & Tracking Management System

A streamlined workflow for downloading source videos, running object-tracking, and syncing all data with your Hugging Face dataset repository.

## Project Structure

```text
.
├── README.md              # This file – general project info
├── .env                   # HF_TOKEN and HF_REPO_ID credentials
├── cookies.txt            # YouTube cookies (optional)
├── ckpt/                  # Cached YOLO weights (created automatically)
├── data/HF_dataset/                  # All project data
│   ├── source_videos/     # Input videos grouped by themed folders
│   └── tracking_videos/   # Tracking results (auto-generated)
├── track_animals.py       # Runs YOLO + tracker and writes results into data/HF_dataset/tracking_videos/
└── manage_data/           # Helper scripts for HF sync
    ├── README.md
    ├── download_data_from_hf.sh
    └── upload_data_to_hf.sh
```

## Installation

1. Create and activate the environment

```bash
micromamba create -f environment.yml
```

> **Important:** Pip wheels for OpenCV do not ship with the H.264 codec required to read/write `.mp4` files.  
> This project therefore pins the conda package `opencv==4.10.0.84`, which includes H.264 support and is fully compatible with Roboflow's `supervision` and `inference` libraries.

3. Configure credentials

Create `.env` in the project root:

```text
HF_TOKEN=your_huggingface_token
HF_REPO_ID=your_username/your_repo_name
```

If you need YouTube downloads, place a valid `cookies.txt` in the project root.

## Features

* Download / upload the entire `data/HF_dataset/` folder with `manage_data/{download,upload}_data_from_hf.sh`.
* Run multi-object tracking on any video (YOLOv8 + ByteTrack or BoTSORT) with `track_animals.py`.
* Results are written directly to `data/HF_dataset/tracking_videos/…` and include:
  * `processed_video.mp4` – annotated clip
  * `tracking_timestamps.csv` – per-ID visibility intervals
* YOLO weights are cached under `ckpt/` to avoid re-downloads.
* Temporary Ultralytics `runs/` folders are cleaned up automatically after each run.

## Basic Usage

1. Sync data from Hugging Face (optional):

```bash
cd manage_data
./download_data_from_hf.sh
```

2. Track animals in a folder of videos:

```bash
python track_animals.py \
    --input data/HF_dataset/source_videos/youtube_segments/ \
    --mode detect \
    --tracker botsort
```

3. Upload updated data back to Hugging Face:

```bash
cd manage_data
./upload_data_to_hf.sh
```

For script-specific details see `manage_data/README.md`.

## Interactive Video Demo

After tracking videos are present under `data/HF_dataset/tracking_videos/`, launch the Streamlit viewer:

```bash
micromamba activate bonting-id  # ensure the env is active
streamlit run video_demo.py
```

Select a run from the dropdown to watch the annotated clip and jump to visibility segments. 