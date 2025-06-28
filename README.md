# Video Demo Management System

A system for managing, downloading, and uploading video demos to Hugging Face repositories.

## Project Structure

```
.
├── README.md              # This file - general project info
├── .env                   # Environment variables (HF credentials)
├── cookies.txt           # YouTube cookies for video downloads
├── videos/               # Video storage directory
│   └── source/          # Downloaded source videos
└── make-bonting-exp/     # Scripts and instructions for video demo creation
    ├── README.md        # Detailed instructions for video demo creation
    ├── download_video_from_yt.sh
    └── upload_videos_to_hf.sh
```

## Installation

1. Set up micromamba environment:
   ```bash
   # Create environment
   micromamba create -n bonting-exp python=3.10
   
   # Activate environment
   micromamba activate bonting-exp
   
   # Install required packages
   micromamba install -c conda-forge yt-dlp
   pip install huggingface-hub
   ```

2. Configure environment:
   Create `.env` file in the project root with your Hugging Face credentials:
   ```
   HF_TOKEN=your_huggingface_token
   HF_REPO_ID=your_username/your_repo_name
   ```

3. Set up YouTube access (if needed):
   Place your `cookies.txt` file in the project root for YouTube video downloads.

## Features

- Download videos from YouTube with proper cookie handling
- Upload videos to Hugging Face repositories
- Organized video storage structure
- Environment-managed dependencies

## Usage

For detailed instructions on creating and managing video demos, see [make-bonting-exp/README.md](make-bonting-exp/README.md). 