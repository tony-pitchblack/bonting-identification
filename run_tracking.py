#!/usr/bin/env python
"""
Identify timestamps where each uniquely tracking cow is visible.

Examples
--------
# single video, pixel-accurate masks, ByteTrack IDs
python track_animals.py --input videos/pen1.mp4 --mode segment --tracker bytetrack

# process every video in a folder, bbox-only, BoTSORT IDs
python track_animals.py --input videos/ --mode detect --tracker botsort
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _slice_segments(id_frames: Dict[int, List[int]], fps: float) -> pd.DataFrame:
    """Convert {id: [frame_indices]} → DataFrame(id, start_ts, end_ts)."""
    rows: List[Tuple[int, float, float]] = []

    for cid, frames in tqdm(id_frames.items(), desc="Building tracking segments"):
        frames.sort()
        start = frames[0]

        for i in range(1, len(frames)):
            # non-contiguous ⇒ previous slice ends
            if frames[i] != frames[i - 1] + 1:
                rows.append((cid, start / fps, frames[i - 1] / fps))
                start = frames[i]

        # last run for this ID
        rows.append((cid, start / fps, frames[-1] / fps))

    df = pd.DataFrame(rows, columns=["id", "start_ts", "end_ts"]).sort_values(
        ["id", "start_ts"]
    )
    return df


def _collect_videos(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXT:
        return [path]
    if path.is_dir():
        videos = [p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXT]
        if not videos:
            raise FileNotFoundError(f"No video files found in directory: {path}")
        return videos
    raise FileNotFoundError(f"{path} is neither a video nor a folder with videos.")


# --------------------------------------------------------------------------- #
# main routine                                                                #
# --------------------------------------------------------------------------- #
def process_video(
    video_path: Path,
    mode: str,
    tracker_name: str,
    device: str = "cpu",
    duration_s: float | None = None,  # process only first N seconds if provided
    out_root: Path = Path("data/HF_dataset/processed_videos/tracking"),
) -> None:
    # ------------------------------------------------------------------ model
    # We store model checkpoints next to this script so they are reused
    ckpt_dir = Path("ckpt")  # relative to where the script is launched
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_file = "yolo11m.pt" if mode == "detect" else "yolo11m-seg.pt"
    local_weights = ckpt_dir / model_file

    if local_weights.exists():
        model = YOLO(str(local_weights))
    else:
        model = YOLO(model_file)
        # Get the downloaded weights path and move to our ckpt dir
        src_path = Path(model.ckpt_path)
        if src_path.exists():
            shutil.copy2(src_path, local_weights)
            # Delete the source file to save space
            src_path.unlink()
            # Reload model with the moved weights
            model = YOLO(str(local_weights))

    # ----------------------------------------------------------- video meta
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    # ---------------------------------------------------------- run tracker
    id_frames: Dict[int, List[int]] = {}
    frame_i = -1

    results = model.track(
        source=str(video_path),
        tracker=f"{tracker_name}.yaml",
        save=True,
        stream=True,  # yields results one frame at a time
        imgsz=640,
        device=device,
        verbose=False,  # Disable YOLO's built-in progress bar
    )

    # Determine how many frames to process if duration limit is set
    max_frames = min(frame_count, int(duration_s * fps)) if duration_s is not None and fps > 0 else frame_count

    for r in tqdm(results, desc=f"Processing {video_path.stem}", total=max_frames):
        frame_i += 1
        ids = (
            []
            if r.boxes.id is None
            else r.boxes.id.cpu().numpy().astype(int).tolist()
        )
        for cid in ids:
            id_frames.setdefault(cid, []).append(frame_i)

        if duration_s is not None and frame_i + 1 >= max_frames:
            break

    # ---------------------------------------------------------- build timestamps
    timestamps_df = _slice_segments(id_frames, fps)

    # ------------------------------------------------------------- outputs
    time_tag = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Determine the folder structure based on the input video path
    # If video is at data/HF_dataset/source_videos/youtube_segments/video.webm
    # Output will be at data/HF_dataset/processed_videos/tracking/youtube_segments/video/{timestamp}/
    video_abs_path = video_path.resolve()
    
    # Try to determine the category folder from the path
    category_folder = "unknown"
    if "source_videos" in video_abs_path.parts:
        # Find the part after source_videos
        source_idx = video_abs_path.parts.index("source_videos")
        if source_idx + 1 < len(video_abs_path.parts):
            category_folder = video_abs_path.parts[source_idx + 1]
    
    out_dir = (
        out_root
        / category_folder
        / video_path.stem
        / f"{time_tag}_model={mode}_tracker={tracker_name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Move/convert annotated video to MP4 in the output directory
    track_base = Path("runs/detect" if mode == "detect" else "runs/segment")
    track_dirs = [d for d in track_base.glob("track*") if d.is_dir()]

    if track_dirs:
        runs_track = max(track_dirs, key=os.path.getctime)
        video_files = list(runs_track.glob("*.mp4")) + list(runs_track.glob("*.avi")) + list(runs_track.glob("*.webm"))
        if video_files:
            ann_video = video_files[0]  # take first
            dest_video = out_dir / "processed_video.mp4"
            if ann_video.suffix.lower() == ".mp4":
                shutil.move(str(ann_video), dest_video)
            else:
                # Convert to mp4 using ffmpeg if available
                try:
                    import subprocess, shlex
                    subprocess.run([
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(ann_video),
                        str(dest_video),
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.remove(ann_video)
                except Exception as e:
                    shutil.move(str(ann_video), dest_video.with_suffix(ann_video.suffix))

    # Clean up temporary Ultralytics output directory to save space
    shutil.rmtree("runs", ignore_errors=True)

    # — timestamps CSV
    timestamps_df.to_csv(out_dir / "tracking_timestamps.csv", index=False)


def cli() -> None:
    p = argparse.ArgumentParser(
        description="Track cattle and output visibility timestamps."
    )
    p.add_argument("input", help="Video file or folder of videos")
    p.add_argument(
        "--mode",
        choices=["detect", "segment"],
        default="detect",
        help="detect=YOLO boxes; segment=YOLO masks",
    )
    p.add_argument(
        "--tracker",
        choices=["bytetrack", "botsort"],
        default="bytetrack",
        help="Multi-object tracker to use",
    )
    p.add_argument(
        "--duration",
        type=float,
        help="Process only first N seconds of each video for debugging",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for YOLO processing (cpu or cuda)",
    )
    args = p.parse_args()

    videos = _collect_videos(Path(args.input))
    if not videos:
        raise SystemExit("❌ No video files found.")

    for i, vid in enumerate(videos, 1):
        if len(videos) > 1:
            print(f"\nProcessing video {i}/{len(videos)}: {vid.name}")
        process_video(vid, args.mode, args.tracker, args.device, args.duration)


if __name__ == "__main__":
    cli()
