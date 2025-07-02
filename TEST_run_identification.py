#!/usr/bin/env python
"""Process a real video for 0.1 s and extract the first frame via ffmpeg."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def run_identification(video_path: Path, extra_args: list[str] | None = None) -> None:
    """Run identification pipeline for the first 0.1 s of *video_path*."""

    cmd = [
        "python",
        str(THIS_DIR / "run_identification.py"),
        str(video_path),
        "--duration",
        "0.1",
        "--yolo-ckpt",
        "yolov8n.pt",
        "--trocr",
        "microsoft/trocr-small-printed",
        "--det-obj-name",
        "cow",
    ]

    if extra_args:
        cmd.extend(extra_args)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def latest_processed_video(video_stem: str) -> Path:
    """Return the most recently modified processed_video.mp4 for *video_stem*."""

    root = THIS_DIR / "data" / "HF_dataset" / "processed_videos" / "identification"
    candidates = [
        p
        for p in root.rglob("processed_video.mp4")
        if p.parent.parent.name == video_stem
    ]
    if not candidates:
        # Fallback: take the most recent processed video overall
        candidates = list(root.rglob("processed_video.mp4"))
        if not candidates:
            raise RuntimeError("No processed video found after identification run.")
        print("Warning: falling back to the most recent processed video irrespective of stem match.")
    
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def extract_first_frame(src_video: Path, dst_png: Path) -> None:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",  # overwrite if exists
        "-i",
        str(src_video),
        "-vframes",
        "1",
        str(dst_png),
    ]
    print("Extracting first frame:", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python TEST_identification.py <video_path> [extra-run-identification-args...]")

    video_path = Path(sys.argv[1]).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    extra_args = sys.argv[2:]

    run_identification(video_path, extra_args)

    processed_video = latest_processed_video(video_path.stem)

    extract_first_frame(
        processed_video,
        Path("tmp/debug_trocr/first_frame.png"),
    )


if __name__ == "__main__":
    main()
