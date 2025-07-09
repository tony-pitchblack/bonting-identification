#!/usr/bin/env python3
"""
roboflow_ocr_demo.py

1) Use Roboflow inference API to load YOLO model for object detection.
2) For each detected bbox, run either TROCR, EasyOCR, or MindOCR on the cropped region.
3) Draw both detection boxes and OCR text results.
4) Save annotated output video.
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

def infer_video_with_roboflow_ocr(model_id: str,
                                  api_key: str,
                                  input_video: str,
                                  output_dir: str,
                                  conf: float = 0.4,
                                  font_size: float = 1.0,
                                  duration = None,
                                  use_trocr: bool = False,
                                  use_easyocr: bool = True,
                                  use_mmocr: bool = False,
                                  trocr_ckpt: str = "trocr-small-printed",
                                  easyocr_ckpt: str = "en",
                                  mmocr_det_algorithm: str = "dbnetpp",
                                  mmocr_rec_algorithm: str = "svtr-small"):
    """
    Runs inference on each frame of input_video using Roboflow inference API,
    detects objects with YOLO, runs OCR on detected regions, and annotates
    both boxes and OCR text.
    
    Args:
        model_id: Roboflow model ID
        api_key: Roboflow API key
        input_video: Path to input video
        output_dir: Directory to save output
        conf: Confidence threshold
        font_size: Font size scale for annotation text (default: 1.0)
        duration: Duration limit - None (1 frame), 'full' (all frames), or float (seconds)
        use_trocr: Whether to use TROCR model
        use_easyocr: Whether to use EasyOCR model
        use_mmocr: Whether to use MMOCR model
        trocr_ckpt: TROCR checkpoint name
        easyocr_ckpt: EasyOCR language codes
        mmocr_det_algorithm: MMOCR detection algorithm
        mmocr_rec_algorithm: MMOCR recognition algorithm
    """
    # Initialize HTTP client pointing to local inference server
    client = InferenceHTTPClient(api_url="http://127.0.0.1:9001", api_key=api_key)
    # Configure confidence threshold once for all requests
    client.configure(InferenceConfiguration(confidence_threshold=conf))

    print(f"[+] Initialized InferenceHTTPClient for model: {model_id}")

    # Initialize OCR models
    easyocr_reader = None
    mmocr_reader = None
    
    if use_easyocr:
        try:
            import easyocr
            # Parse language codes from easyocr_ckpt (comma-separated)
            languages = [lang.strip() for lang in easyocr_ckpt.split(',')]
            easyocr_reader = easyocr.Reader(languages)
            print(f"[+] Initialized EasyOCR with languages: {languages}")
        except ImportError:
            print("[-] EasyOCR not installed. Install with: pip install easyocr")
            raise
        except Exception as e:
            print(f"[-] Error initializing EasyOCR: {e}")
            raise
    
    if use_mmocr:
        try:
            from mmocr.apis import MMOCRInferencer
            mmocr_reader = MMOCRInferencer(det=mmocr_det_algorithm,
                                           rec=mmocr_rec_algorithm,
                                           device=None)
            print(f"[+] Initialized MMOCR with det_algorithm: {mmocr_det_algorithm}, rec_algorithm: {mmocr_rec_algorithm}")
        except ImportError:
            print("[-] MMOCR not installed. Install with: pip install mmocr")
            raise
        except Exception as e:
            print(f"[-] Error initializing MMOCR: {e}")
            raise

    if use_trocr:
        print(f"[+] Will use TROCR model: {trocr_ckpt}")

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
                
            # Run YOLO inference via HTTP client
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
                
                # Ensure coordinates are within frame bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))
                
                # Draw rectangle
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw class label with shadow
                label = f"{class_name}: {confidence:.2f}"
                # Draw shadow (black text slightly offset)
                cv2.putText(annotated, label, (x + 2, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 2)
                # Draw main text (green)
                cv2.putText(annotated, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)
                
                # Extract ROI for OCR
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:  # Make sure ROI is not empty
                    ocr_text = ""
                    try:
                        if use_easyocr and easyocr_reader:
                            # Run EasyOCR on the detected region
                            results = easyocr_reader.readtext(roi, detail=0)
                            ocr_text = " ".join(results) if results else ""
                        elif use_mmocr and mmocr_reader:
                            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            mmocr_out = mmocr_reader(roi_rgb, batch_size=1, return_vis=False, save_vis=False)
                            preds = mmocr_out.get("predictions", [])
                            if preds:
                                texts = preds[0].get("rec_texts", [])
                                if texts:
                                    ocr_text = " | ".join(texts)
                        elif use_trocr:
                            # Run TROCR via Roboflow API
                            ocr_result = client.ocr_image(inference_input=roi, model="trocr")
                            ocr_text = ocr_result.get("result", "")
                        
                        # Draw OCR text above the bbox
                        if ocr_text:
                            # Position text above the detection box with spacing
                            text_y = y - 50 if y > 50 else y + h + 30
                            ocr_label = f"OCR: {ocr_text}"
                            
                            # Draw shadow (black text slightly offset)
                            cv2.putText(annotated, ocr_label, (x + 2, text_y + 2),
                                       cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 2)
                            # Draw main text (yellow)
                            cv2.putText(annotated, ocr_label, (x, text_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 0), 2)
                    except Exception as e:
                        # If OCR fails, continue with next detection
                        print(f"OCR failed for detection: {e}")
                        continue

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
        
        # Add some text that could be detected
        cv2.putText(frame, "TEST123", (x + 10, y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"[+] Test MP4 video created: {output_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Roboflow inference with OCR on video")
    parser.add_argument("--duration", type=str, default=None, help="Duration limit: None (1 frame), 'full' (all frames), or float (seconds)")
    parser.add_argument("--font-size", type=float, default=0.5, help="Font size scale for annotation text (default: 1.0)")
    parser.add_argument("--use-trocr", action="store_true", help="Use TROCR model (default: False)")
    parser.add_argument("--use-easyocr", action="store_true", default=True, help="Use EasyOCR model (default: True)")
    parser.add_argument("--use-mmocr", action="store_true", help="Use MMOCR model (default: False)")
    parser.add_argument("--trocr-ckpt", type=str, default="trocr-small-printed", 
                       choices=["trocr-small-printed", "trocr-base-printed", "trocr-large-printed"],
                       help="TROCR checkpoint (default: trocr-small-printed)")
    parser.add_argument("--easyocr-ckpt", type=str, default="en", 
                       help="EasyOCR language codes, comma-separated (default: en)")
    parser.add_argument("--mmocr-det-algorithm", type=str, default="DBNet", help="MMOCR detection algorithm name (default: DBNet)")
    parser.add_argument("--mmocr-rec-algorithm", type=str, default="CRNN", help="MMOCR recognition algorithm name (default: CRNN)")
    args = parser.parse_args()
    
    # Parse duration argument
    duration = args.duration
    if duration is not None and duration != 'full':
        try:
            duration = float(duration)
        except ValueError:
            raise ValueError(f"Invalid duration: {args.duration}. Must be None, 'full', or a float.")
    
    # Handle mutual exclusivity - ensure only one OCR mode is active
    ocr_modes = [args.use_trocr, args.use_easyocr, args.use_mmocr]
    active_modes = sum(ocr_modes)
    
    if active_modes > 1:
        # If multiple modes are set, use the priority: MMOCR > TROCR > EasyOCR
        if args.use_mmocr:
            args.use_trocr = False
            args.use_easyocr = False
        elif args.use_trocr:
            args.use_easyocr = False
    elif active_modes == 0:
        # If no mode is set, default to EasyOCR
        args.use_easyocr = True
    
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
    
    INPUT_VIDEO = Path(os.getenv("DEMO_VIDEO_PATH", ""))
    TEST_VIDEO = "test_video.mp4"

    # Check if input video exists, if not create a test video
    if not os.path.exists(INPUT_VIDEO):
        print(f"Warning: Input video {INPUT_VIDEO} not found.")
        if not os.path.exists(TEST_VIDEO):
            create_test_video(TEST_VIDEO, duration=5, font_size=args.font_size)
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
    
    # Create model descriptor for output filename
    if args.use_mmocr:
        model_descriptor = f"mmocr_{args.mmocr_det_algorithm}_{args.mmocr_rec_algorithm}"
    elif args.use_trocr:
        model_descriptor = f"trocr_{args.trocr_ckpt}"
    else:
        model_descriptor = f"easyocr_{args.easyocr_ckpt.replace(',', '-')}"
    
    # Create output directory structure
    out_root = Path("data/HF_dataset/processed_videos/tracking")
    out_dir = (
        out_root
        / category_folder
        / video_path.stem
        / f"{time_tag}_model=roboflow_ocr={model_descriptor}"
    )
    
    print(f"[+] Using model: {MODEL_ID}")
    if args.use_mmocr:
        print(f"[+] OCR model: MMOCR")
        print(f"[+] OCR algorithms: det={args.mmocr_det_algorithm}, rec={args.mmocr_rec_algorithm}")
    elif args.use_trocr:
        print(f"[+] OCR model: TROCR")
        print(f"[+] OCR checkpoint: {args.trocr_ckpt}")
    else:
        print(f"[+] OCR model: EasyOCR")
        print(f"[+] OCR checkpoint: {args.easyocr_ckpt}")
    print(f"[+] Input video: {INPUT_VIDEO}")
    print(f"[+] Output directory: {out_dir}")
    
    infer_video_with_roboflow_ocr(
        model_id=MODEL_ID,
        api_key=API_KEY,
        input_video=str(INPUT_VIDEO),
        output_dir=str(out_dir),
        conf=0.1,
        font_size=args.font_size,
        duration=duration,
        use_trocr=args.use_trocr,
        use_easyocr=args.use_easyocr,
        use_mmocr=args.use_mmocr,
        trocr_ckpt=args.trocr_ckpt,
        easyocr_ckpt=args.easyocr_ckpt,
        mmocr_det_algorithm=args.mmocr_det_algorithm,
        mmocr_rec_algorithm=args.mmocr_rec_algorithm
    ) 