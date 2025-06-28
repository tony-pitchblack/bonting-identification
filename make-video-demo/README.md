# Creating Video Demos

This guide explains how to create and manage video demos using the provided scripts.

## Prerequisites

Ensure you have completed the installation steps in the root [README.md](../README.md):
- Activated micromamba environment (`bonting-exp`)
- Set up `.env` file with Hugging Face credentials
- Added `cookies.txt` for YouTube access (if needed)

## Available Scripts

- `download_video_from_yt.sh`: Downloads videos from YouTube to `videos/source/`
- `upload_videos_to_hf.sh`: Uploads videos to your Hugging Face repository

## Usage Instructions

1. Activate the environment (if not already activated):
   ```bash
   micromamba activate bonting-exp
   ```

2. Download a video from YouTube:
   ```bash
   cd make-bonting-exp
   ./download_video_from_yt.sh
   ```
   The video will be saved to `videos/source/` directory.

3. Upload videos to Hugging Face:
   ```bash
   cd make-bonting-exp
   ./upload_videos_to_hf.sh
   ```
   This uploads the entire `videos` directory to your specified Hugging Face repository.

## Troubleshooting

Common issues and solutions:

1. **Script fails with environment error**
   - Ensure you've activated the correct environment: `micromamba activate bonting-exp`
   
2. **YouTube download fails**
   - Check that `cookies.txt` is present in the project root
   - Verify the YouTube URL in `download_video_from_yt.sh`

3. **Upload fails**
   - Verify your Hugging Face credentials in `.env`
   - Check your internet connection
   - Ensure the target repository exists and you have write permissions

## Customization

To download a different video, edit the URL in `download_video_from_yt.sh`:
```bash
# Open the script in your editor
nano download_video_from_yt.sh

# Change the URL at the end of the file
yt-dlp --cookies ../cookies.txt -P ../videos/source/ YOUR_VIDEO_URL
``` 