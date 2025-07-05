#!/usr/bin/env python3
"""
roboflow_demo.py

1) Use Roboflow inference API to load model directly.
2) Run inference on a video and save an annotated output.
3) Write directly in .mp4 format using avc1 codec.
"""

import os
import cv2
import datetime as dt
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Silence inference warnings
os.environ['QWEN_2_5_ENABLED'] = 'False'
os.environ['CORE_MODEL_SAM_ENABLED'] = 'False'
os.environ['CORE_MODEL_SAM2_ENABLED'] = 'False'
os.environ['CORE_MODEL_CLIP_ENABLED'] = 'False'
os.environ['CORE_MODEL_GAZE_ENABLED'] = 'False'
os.environ['SMOLVLM2_ENABLED'] = 'False'
os.environ['CORE_MODEL_GROUNDINGDINO_ENABLED'] = 'False'
os.environ['CORE_MODEL_YOLO_WORLD_ENABLED'] = 'False'
os.environ['CORE_MODEL_PE_ENABLED'] = 'False'

def infer_video_with_roboflow(model_id: str,
                              api_key: str,
                              input_video: str,
                              output_dir: str,
                              conf: float = 0.4,
                              font_size: float = 1.0,
                              duration = None):
    """
    Runs inference on each frame of input_video using Roboflow inference API,
    annotates boxes+labels, and writes the result to output_dir.
    
    Args:
        model_id: Roboflow model ID
        api_key: Roboflow API key
        input_video: Path to input video
        output_dir: Directory to save output
        conf: Confidence threshold
        font_size: Font size scale for annotation text (default: 1.0)
        duration: Duration limit - None (1 frame), 'full' (all frames), or float (seconds)
    """
    # Initialize HTTP client pointing to local inference server
    client = InferenceHTTPClient(api_url="http://127.0.0.1:9001", api_key=api_key)
    # Configure confidence threshold once for all requests
    client.configure(InferenceConfiguration(confidence_threshold=conf))

    print(f"[+] Initialized InferenceHTTPClient for model: {model_id}")

    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {input_video!r}")

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frames to process based on duration parameter
    if duration is None:
        frames_to_process = 1
        print(f"[+] Processing exactly 1 frame")
    elif duration == 'full':
        frames_to_process = total_frames
        print(f"[+] Processing full video ({frames_to_process} frames)")
    else:
        frames_to_process = min(int(float(duration) * fps), total_frames)
        print(f"[+] Processing first {duration} seconds ({frames_to_process} frames)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Output video path
    output_video_mp4 = output_path / "processed_video.mp4"

    # Create MP4 writer with avc1 codec
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(output_video_mp4), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for MP4 output: {output_video_mp4}")

    print(f"[+] Starting inference on {input_video!r}")
    print(f"[+] Writing MP4 directly to {output_video_mp4}")
    
    # Process frames with progress bar
    frame_count = 0
    with tqdm(total=frames_to_process, desc="Processing frames") as pbar:
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference via HTTP client
            result = client.infer(frame, model_id=model_id)
            predictions = result.get("predictions", [])
            
            # Manual annotation using OpenCV
            annotated = frame.copy()
            
            for pred in predictions:
                # Extract prediction data (dictionary access)
                x = int(pred["x"] - pred["width"] / 2)
                y = int(pred["y"] - pred["height"] / 2)
                w = int(pred["width"])
                h = int(pred["height"])
                class_name = pred.get("class", "")
                confidence = pred.get("confidence", 0.0)
                    
                # Draw rectangle
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)

            out.write(annotated)
            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"[+] MP4 video saved to {output_video_mp4}")

def create_test_video(output_path: str, duration: int = 5, font_size: float = 1.0):
    """Create a simple test video for demonstration."""
    import numpy as np
    
    # Video properties
    fps = 30
    width, height = 640, 480
    
    # Create MP4 writer with avc1 codec
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Could not create test video: {output_path}")
    
    print(f"[+] Creating test video (MP4): {output_path}")
    
    for frame_num in range(fps * duration):
        # Create a simple animated frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some moving shapes
        x = int(50 + 100 * np.sin(frame_num * 0.1))
        y = int(100 + 50 * np.cos(frame_num * 0.1))
        
        cv2.rectangle(frame, (x, y), (x + 100, y + 80), (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"[+] Test MP4 video created: {output_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Roboflow inference on video")
    parser.add_argument("--duration", type=str, default=None, help="Duration limit: None (1 frame), 'full' (all frames), or float (seconds)")
    parser.add_argument("--font-size", type=float, default=1.0, help="Font size scale for annotation text (default: 1.0)")
    args = parser.parse_args()
    
    # Parse duration argument
    duration = args.duration
    if duration is not None and duration != 'full':
        try:
            duration = float(duration)
        except ValueError:
            raise ValueError(f"Invalid duration: {args.duration}. Must be None, 'full', or a float.")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get API key
    API_KEY = os.getenv("ROBOFLOW_API_KEY")
    if not API_KEY:
        raise ValueError("ROBOFLOW_API_KEY environment variable not set")
    
    # Configuration
    WORKSPACE = "cow-test-yeo0m"
    MODEL_NAME = "cows-gyup1"
    VERSION = 2
    MODEL_ID = f"{MODEL_NAME}/{VERSION}"
    
    INPUT_VIDEO = "data/HF_dataset/source_videos/youtube_segments/Milking R Dairy - Nedap CowControl with the SmartTag Ear merged 4 cuts.webm"
    TEST_VIDEO = "test_video.mp4"

    # Check if input video exists, if not create a test video
    if not os.path.exists(INPUT_VIDEO):
        print(f"Warning: Input video {INPUT_VIDEO} not found.")
        if not os.path.exists(TEST_VIDEO):
            create_test_video(TEST_VIDEO)
        INPUT_VIDEO = TEST_VIDEO
        print(f"Using test video: {INPUT_VIDEO}")
    
    # Create output directory structure similar to run_tracking.py
    time_tag = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_path = Path(INPUT_VIDEO)
    video_abs_path = video_path.resolve()
    
    # Determine category folder based on path structure
    category_folder = "unknown"
    if "source_videos" in video_abs_path.parts:
        source_idx = video_abs_path.parts.index("source_videos")
        if source_idx + 1 < len(video_abs_path.parts):
            category_folder = video_abs_path.parts[source_idx + 1]
    
    # Create output directory structure
    out_root = Path("data/HF_dataset/processed_videos/tracking")
    out_dir = (
        out_root
        / category_folder
        / video_path.stem
        / f"{time_tag}_model=roboflow_tracker=inference"
    )
    
    print(f"[+] Using model: {MODEL_ID}")
    print(f"[+] Input video: {INPUT_VIDEO}")
    print(f"[+] Output directory: {out_dir}")
    
    infer_video_with_roboflow(
        model_id=MODEL_ID,
        api_key=API_KEY,
        input_video=INPUT_VIDEO,
        output_dir=str(out_dir),
        conf=0.4,
        font_size=args.font_size,
        duration=duration
    )
