#!/usr/bin/env python3
"""
roboflow_head_demo.py

Minimal script demonstrating how to run a pretrained Roboflow model for cow-head
(cattle-face) tracking on a video.

Differences to roboflow_ocr_demo.py:
- Uses model `cattle-face-detection-v2/5`.
- Performs only object tracking and draws bounding boxes.
- No OCR post-processing.

Usage (example):
    python roboflow_head_demo.py --input-video input.mp4 --output-dir results/
"""
from pathlib import Path
import argparse
import os
from typing import Optional

import cv2
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# -----------------------------------------------------------------------------
# Core inference routine
# -----------------------------------------------------------------------------

def infer_video_with_roboflow_tracking(
    model_id: str,
    api_key: str,
    input_video: str,
    output_dir: str,
    api_url: str = "http://127.0.0.1:9001",
    conf: float = 0.1,
    iou_threshold: float = 0.3,
    font_size: float = 0.5,
    duration: Optional[str] = None,
):
    """Run cow-head tracking on a video and save an annotated copy."""
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    client.configure(InferenceConfiguration(confidence_threshold=conf, iou_threshold=iou_threshold))

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if duration is None:
        frames_to_process = 1
    elif duration == "full":
        frames_to_process = total_frames
    else:
        frames_to_process = min(int(float(duration) * fps), total_frames)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_video = out_path / "processed_video.mp4"

    fourcc = cv2.VideoWriter.fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {output_video}")

    processed = 0
    while processed < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        result = client.infer(frame, model_id=model_id)
        for pred in result.get("predictions", []):
            x1 = int(pred["x"] - pred["width"] / 2)
            y1 = int(pred["y"] - pred["height"] / 2)
            x2 = x1 + int(pred["width"])
            y2 = y1 + int(pred["height"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = pred.get("class", "")
            if label:
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 255, 0),
                    2,
                )

        writer.write(frame)
        processed += 1

    cap.release()
    writer.release()

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cow head tracking with Roboflow")
    parser.add_argument("--config", type=str, default="config_roboflow_head_demo.yml", help="Path to YAML configuration file")
    parser.add_argument("--input-video", type=str, help="Input video path (overrides config)")
    parser.add_argument("--duration", type=str, default=None, help="Duration: None (1 frame), 'full', or seconds")
    parser.add_argument("--font-size", type=float, default=None, help="Annotation font scale (overrides config)")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (overrides config)")
    parser.add_argument("--iou-threshold", type=float, default=None, help="IoU threshold for NMS (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (optional)")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Configuration handling (reuse logic from roboflow_ocr_demo)
    # ---------------------------------------------------------------------
    from dotenv import load_dotenv
    from roboflow_ocr_demo import load_config, create_test_video  # reuse existing helpers
    import datetime as dt

    CONFIG = load_config(args.config)

    # Determine configuration values (CLI overrides config)
    duration = args.duration if args.duration is not None else CONFIG['defaults']['duration']
    font_size = args.font_size if args.font_size is not None else CONFIG['defaults']['font_size']
    conf_threshold = args.conf if args.conf is not None else CONFIG['defaults']['confidence_threshold']
    iou_threshold = getattr(args, 'iou_threshold') if getattr(args, 'iou_threshold') is not None else CONFIG['defaults']['iou_threshold']
    
    if duration is not None and duration != 'full':
        try:
            duration = float(duration)
        except ValueError:
            raise ValueError(f"Invalid duration: {duration}. Must be None, 'full', or a float.")

    # ---------------------------------------------------------------------
    # Video input handling
    # ---------------------------------------------------------------------
    if args.input_video:
        input_video_path = Path(args.input_video)
    elif CONFIG['video']['input_path']:
        input_video_path = Path(CONFIG['video']['input_path'])
    else:
        input_video_path = None

    test_video_name = CONFIG['video']['test_video_name']

    if input_video_path is None or not input_video_path.exists():
        if input_video_path is not None:
            print(f"Warning: Input video {input_video_path} not found.")
        else:
            print("Warning: No input video specified in config or CLI.")

        if not Path(test_video_name).exists():
            create_test_video(test_video_name, duration=CONFIG['video']['test_video_duration'], font_size=font_size)
        input_video_path = Path(test_video_name)
        print(f"Using test video: {input_video_path}")

    # ---------------------------------------------------------------------
    # Output directory handling (mirrors roboflow_ocr_demo structure)
    # ---------------------------------------------------------------------
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        time_tag = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_abs = input_video_path.resolve()
        
        # Use remove_input_prefix from config to construct output path
        remove_prefix = CONFIG['output'].get('remove_input_prefix', '')
        video_path_str = str(video_abs)

        if remove_prefix and remove_prefix in video_path_str:
            # Strip everything up to (and including) the first occurrence of the prefix
            relative_path = video_path_str.split(remove_prefix, 1)[1]
        else:
            # Fallback – just use the filename
            relative_path = Path(video_path_str).name

        # Ensure we treat it as a relative path (no leading slash)
        relative_path = relative_path.lstrip('/')
        
        # Convert path to Path object and remove the file extension to make it a folder
        relative_path_obj = Path(relative_path)
        folder_path = relative_path_obj.with_suffix('')  # Remove extension
        
        out_root = Path(CONFIG['output']['base_dir'])
        model_name = CONFIG['model']['model_id'].replace('/', '_')
        out_dir = out_root / folder_path / f"{time_tag}_model={model_name}"

    # Ensure env variable
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY environment variable not set")

    # ---------------------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------------------
    model_id = CONFIG['model']['model_id']
    api_url = CONFIG['model']['api_url']

    print(f"[+] Input video: {input_video_path}")
    print(f"[+] Output directory: {out_dir}")
    print(f"[+] Using model: {model_id}")

    infer_video_with_roboflow_tracking(
        model_id=model_id,
        api_key=api_key,
        input_video=str(input_video_path),
        output_dir=str(out_dir),
        api_url=api_url,
        conf=conf_threshold,
        iou_threshold=iou_threshold,
        font_size=font_size,
        duration=duration,
    )
