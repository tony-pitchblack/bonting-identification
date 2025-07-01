# Data Management Scripts

This directory contains scripts for managing the project's data using Hugging Face datasets.

## Data Structure

The project uses the following folder structure:

```
data/
  source_videos/           # Source videos used for tracking
    youtube_segments/      # Videos downloaded from YouTube
    folder_n/              # Other themed folders
  tracking_videos/          # Tracking results
    youtube_segments/      # Results grouped by source folder
      video_name/          # Results for specific video
        timestamp_model_tracker/
          tracking_timestamps.csv
          processed_video.mp4
```

## Prerequisites

Ensure you have completed the installation steps in the root [README.md](../README.md):
- Activated micromamba environment (`bonting-id`)
- Set up `.env` file with Hugging Face credentials

## Available Scripts

- `download_data_from_hf.sh`: Downloads the entire `data/` directory from Hugging Face
- `upload_data_to_hf.sh`: Uploads the entire `data/` directory to your Hugging Face repository

## Usage Instructions

1. Activate the environment (if not already activated):
   ```bash
   micromamba activate bonting-id
   ```

2. Download data from Hugging Face:
   ```bash
   cd manage_data
   ./download_data_from_hf.sh
   ```
   This downloads the entire `data/` directory structure, including source videos and tracking results.

3. Upload data to Hugging Face:
   ```bash
   cd manage_data
   ./upload_data_to_hf.sh
   ```
   This uploads the entire `data/` directory to your specified Hugging Face repository.

## Troubleshooting

Common issues and solutions:

1. **Script fails with environment error**
   - Ensure you've activated the correct environment: `micromamba activate bonting-id`
   
2. **Download/Upload fails**
   - Verify your Hugging Face credentials in `.env`
   - Check your internet connection
   - Ensure the target repository exists and you have write permissions

## Data Organization

- Source videos should be placed in appropriate subdirectories under `data/source_videos/`
- Tracking results are automatically organized in `data/tracking_videos/` by the tracking script
- The directory structure is preserved when uploading/downloading from Hugging Face 