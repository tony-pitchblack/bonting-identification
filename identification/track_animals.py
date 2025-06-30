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
    print(f"Building timestamps for {len(id_frames)} tracking IDs...")
    rows: List[Tuple[int, float, float]] = []

    for cid, frames in tqdm(id_frames.items(), desc="Processing tracking IDs"):
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
    print(f"Generated {len(df)} timestamps")
    return df


def _collect_videos(path: Path) -> List[Path]:
    print(f"Searching for videos in: {path}")
    if path.is_file() and path.suffix.lower() in VIDEO_EXT:
        print(f"Found single video file: {path.name}")
        return [path]
    if path.is_dir():
        videos = [p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXT]
        print(f"Found {len(videos)} video files in directory")
        for vid in videos:
            print(f"  - {vid.name}")
        return videos
    raise FileNotFoundError(f"{path} is neither a video nor a folder with videos.")


# --------------------------------------------------------------------------- #
# main routine                                                                #
# --------------------------------------------------------------------------- #
def process_video(
    video_path: Path,
    mode: str,
    tracker_name: str,
    duration_s: float | None = None,  # process only first N seconds if provided
    out_root: Path = Path("data/tracking_videos"),
) -> None:
    print(f"\nProcessing video: {video_path.name}")
    print(f"Mode: {mode}, Tracker: {tracker_name}")

    # ------------------------------------------------------------------ model
    # We store model checkpoints next to this script so they are reused
    script_dir = Path(__file__).resolve().parent
    ckpt_dir = script_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_file = "yolo11m.pt" if mode == "detect" else "yolo11m-seg.pt"
    local_weights = ckpt_dir / model_file

    if local_weights.exists():
        print(f"Loading YOLO model from local weights: {local_weights} ...")
        model = YOLO(str(local_weights))
    else:
        print(f"Downloading {model_file} to {ckpt_dir} ...")
        # Download to a temp location first
        model = YOLO(model_file)
        # Get the downloaded weights path and move to our ckpt dir
        src_path = Path(model.ckpt_path)
        if src_path.exists():
            shutil.copy2(src_path, local_weights)
            print(f"Moved weights to {local_weights}")
            # Delete the source file to save space
            src_path.unlink()
            print(f"Deleted source weights at {src_path}")
            # Reload model with the moved weights
            model = YOLO(str(local_weights))
        else:
            print(f"Warning: Could not find downloaded weights at {src_path}")

    print("Model loaded.")

    # ----------------------------------------------------------- video meta
    print("Reading video metadata ...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    print(f"Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s")

    # ---------------------------------------------------------- run tracker
    print(f"Running tracker ({tracker_name}) ...")
    id_frames: Dict[int, List[int]] = {}
    frame_i = -1

    results = model.track(
        source=str(video_path),
        tracker=f"{tracker_name}.yaml",
        save=True,
        stream=True,  # yields results one frame at a time
        imgsz=640,
        verbose=True,
    )

    # Determine how many frames to process if duration limit is set
    if duration_s is not None and fps > 0:
        max_frames = min(frame_count, int(duration_s * fps))
        print(f"Processing first {duration_s} seconds (~{max_frames} frames) ...")
    else:
        max_frames = frame_count
        print("Processing frames and tracking objects ...")

    for r in tqdm(results, desc=f"Tracking {video_path.stem}", total=max_frames):
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

    print(f"Tracking complete. Processed {frame_i + 1} frames")
    print(f"Found {len(id_frames)} unique tracking IDs")

    # ---------------------------------------------------------- build timestamps
    timestamps_df = _slice_segments(id_frames, fps)

    # ------------------------------------------------------------- outputs
    print("Preparing output files ...")
    time_tag = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Determine the folder structure based on the input video path
    # If video is at data/source_videos/youtube_segments/video.webm
    # Output should be at data/tracking_videos/youtube_segments/video/{timestamp}/
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
    print(f"Output directory: {out_dir}")

    # Move/convert annotated video to MP4 in the output directory
    print("Collecting annotated video ...")
    track_base = Path("runs/detect" if mode == "detect" else "runs/segment")
    track_dirs = [d for d in track_base.glob("track*") if d.is_dir()]

    if track_dirs:
        runs_track = max(track_dirs, key=os.path.getctime)
        video_files = list(runs_track.glob("*.mp4")) + list(runs_track.glob("*.avi")) + list(runs_track.glob("*.webm"))
        if video_files:
            ann_video = video_files[0]  # take first
            dest_video = out_dir / "tracking_video.mp4"
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
                    print(f"Warning: failed to convert video to mp4 ({e}). Saving original format.")
                    shutil.move(str(ann_video), dest_video.with_suffix(ann_video.suffix))
            print("Annotated video saved.")
        else:
            print("No annotated video file found.")
    else:
        print("No tracking output directory found.")

    # Clean up temporary Ultralytics output directory to save space
    shutil.rmtree("runs", ignore_errors=True)

    # — timestamps CSV
    print("Saving tracking timestamps CSV ...")
    timestamps_df.to_csv(out_dir / "tracking_timestamps.csv", index=False)
    print("Processing complete.\n")


def cli() -> None:
    print("Starting animal tracking ...")
    p = argparse.ArgumentParser(
        description="Track cattle and output visibility timestamps."
    )
    p.add_argument("--input", required=True, help="Video file or folder of videos")
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
    args = p.parse_args()

    print("Configuration:")
    print(f"  Input: {args.input}")
    print(f"  Mode: {args.mode}")
    print(f"  Tracker: {args.tracker}")
    print(f"  Duration: {args.duration} seconds")

    videos = _collect_videos(Path(args.input))
    if not videos:
        raise SystemExit("❌ No video files found.")

    print(f"\nProcessing {len(videos)} video(s) ...")
    for i, vid in enumerate(videos, 1):
        print(f"\n--- Video {i}/{len(videos)} ---")
        process_video(vid, args.mode, args.tracker, args.duration)

    print("All videos processed successfully.")


if __name__ == "__main__":
    cli()
