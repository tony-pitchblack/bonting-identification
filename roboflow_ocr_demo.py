#!/usr/bin/env python3
"""
roboflow_ocr_demo.py

1) Use Roboflow inference API to load YOLO model for object detection.
2) For each detected bbox, run either TROCR, EasyOCR, or MindOCR on the cropped region.
3) Draw both detection boxes and OCR text results.
4) Save annotated output video.

Configuration:
- Script reads settings from config_roboflow_ocr_demo.yml by default
- Use --config to specify a different config file
- Use --input-video to override the video path from config
- Config includes video paths, model settings, OCR algorithms, and default parameters

Dependencies:
- PyYAML: pip install pyyaml
"""

import os
import cv2
import datetime as dt
import argparse
import yaml
import mlflow
from pathlib import Path
from glob import glob
from typing import Optional, Dict, Any
from tqdm import tqdm
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from mlflow.tracking import MlflowClient

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

def load_model_ckpt(name, version, file_regex="*"):
    """Load model checkpoint from MLflow registry."""
    ckpt_root = Path("ckpt/mmocr/")
    model_path = ckpt_root / name / version
    model_path.mkdir(parents=True, exist_ok=True)
    
    model_uri = f"models:/{name}/{version}"
    _ = mlflow.pytorch.load_model(model_uri, dst_path=model_path)
    
    ckpt_paths = glob(f'{model_path}/extra_files/{file_regex}')
    assert len(ckpt_paths) == 1, f"Expected 1 checkpoint file, got {len(ckpt_paths)}"
    
    return ckpt_paths[0]

def download_config(model_name, model_version):
    """Download model config from MLflow registry."""
    client = MlflowClient()
    mv = client.get_model_version(name=model_name, version=model_version)
    run_id = mv.run_id
    
    config_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="config.py", 
        dst_path=f"./ckpt/mmocr/{model_name}/{model_version}"
    )
    return config_path

def load_config(config_path: str = "config_roboflow_ocr_demo.yml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default values.")
        return {
            'video': {
                'input_path': '',
                'test_video_duration': 5,
                'test_video_name': 'test_video.mp4'
            },
            'mmocr': {
                'detection': {'model_name': 'DBNet', 'model_version': '1'},
                'recognition': {'model_name': 'SATRN', 'model_version': '1'}
            },
            'trocr': {'default': 'trocr-small-printed'},
            'easyocr': {'default_languages': 'en'},
            'defaults': {
                'font_size': 0.5,
                'confidence_threshold': 0.1,
                'use_mmocr': True,
                'use_trocr': False,
                'use_easyocr': False,
                'duration': None
            },
            'model': {
                'workspace': 'cow-test-yeo0m',
                'model_name': 'cows-gyup1',
                'version': 2
            }
        }
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        raise

# Load default configuration (will be reloaded with proper config path in main)
CONFIG = load_config()
MMOCR_DET_MODEL_NAME = CONFIG['mmocr']['detection']['model_name']
MMOCR_RECOG_MODEL_NAME = CONFIG['mmocr']['recognition']['model_name']
MMOCR_DET_MODEL_VERSION = CONFIG['mmocr']['detection']['model_version']
MMOCR_RECOG_MODEL_VERSION = CONFIG['mmocr']['recognition']['model_version']

def infer_video_with_roboflow_ocr(model_id: str,
                                  api_key: str,
                                  input_video: str,
                                  output_dir: str,
                                  conf: float,
                                  font_size: float,
                                  duration,
                                  use_trocr: bool,
                                  use_easyocr: bool,
                                  use_mmocr: bool,
                                  trocr_ckpt: str,
                                  easyocr_ckpt: str,
                                  mmocr_det_model_name: str,
                                  mmocr_recog_model_name: str,
                                  mmocr_det_model_version: str,
                                  mmocr_recog_model_version: str):
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
        mmocr_det_model_name: MMOCR detection model name
        mmocr_recog_model_name: MMOCR recognition model name
        mmocr_det_model_version: MMOCR detection model version
        mmocr_recog_model_version: MMOCR recognition model version
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
            
            # Load model checkpoints and configs using MLflow
            det_ckpt_path = load_model_ckpt(mmocr_det_model_name, mmocr_det_model_version)
            recog_ckpt_path = load_model_ckpt(mmocr_recog_model_name, mmocr_recog_model_version)
            det_config_path = download_config(mmocr_det_model_name, mmocr_det_model_version)
            recog_config_path = download_config(mmocr_recog_model_name, mmocr_recog_model_version)
            
            mmocr_reader = MMOCRInferencer(det=det_config_path,
                                           det_weights=det_ckpt_path,
                                           rec=recog_config_path,
                                           rec_weights=recog_ckpt_path,
                                           device=None)
            print(f"[+] Initialized MMOCR with det_model: {mmocr_det_model_name} v{mmocr_det_model_version}, rec_model: {mmocr_recog_model_name} v{mmocr_recog_model_version}")
            print(f"[+] MMOCR det_weights: {det_ckpt_path}")
            print(f"[+] MMOCR rec_weights: {recog_ckpt_path}")
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
    fourcc = cv2.VideoWriter.fourcc(*"avc1")
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
            
            # Calculate text height for proper spacing (approximate)
            text_height = int(font_size * 20)  # Approximate text height based on font_size
            text_margin = max(5, int(text_height * 0.3))  # Margin between bbox and text

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
                
                # Draw rectangle only (no class label)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract ROI for OCR on all predictions
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:  # Make sure ROI is not empty
                    ocr_text = ""
                    avg_confidence = 0.0
                    try:
                        if use_easyocr and easyocr_reader:
                            # Run EasyOCR on the detected region
                            results = easyocr_reader.readtext(roi, detail=1)  # detail=1 to get confidence
                            if results:
                                # EasyOCR returns [(bbox, text, confidence), ...]
                                texts = [result[1] for result in results]
                                confidences = [result[2] for result in results]
                                ocr_text = " ".join(texts)
                                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                        elif use_mmocr and mmocr_reader:
                            # Run MMOCR on the detected region
                            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            mmocr_result = mmocr_reader(roi_rgb, return_vis=False, save_vis=False)
                            
                            # Extract text and confidence from MMOCR result
                            if mmocr_result and 'predictions' in mmocr_result:
                                predictions_list = mmocr_result['predictions']
                                if predictions_list:
                                    pred_dict = predictions_list[0]  # First (and only) image
                                    if 'rec_texts' in pred_dict:
                                        texts = pred_dict['rec_texts']
                                        rec_scores = pred_dict.get('rec_scores', [])
                                        det_scores = pred_dict.get('det_scores', [])

                                        if texts:
                                            # Select the single text with highest rec_score (or first if no scores)
                                            if rec_scores:
                                                best_idx = int(max(range(len(texts)), key=lambda i: rec_scores[i]))
                                                chosen_text = texts[best_idx]
                                                chosen_rec_conf = rec_scores[best_idx]
                                                chosen_det_conf = det_scores[best_idx] if det_scores else chosen_rec_conf
                                            else:
                                                chosen_text = texts[0]
                                                chosen_rec_conf = 0.0
                                                chosen_det_conf = 0.0

                                            ocr_text = chosen_text

                                            # Average det+rec confidence when available, else use rec confidence
                                            if chosen_det_conf and chosen_rec_conf:
                                                avg_confidence = (chosen_det_conf + chosen_rec_conf) / 2
                                            else:
                                                avg_confidence = chosen_rec_conf or chosen_det_conf
                        elif use_trocr:
                            # Run TROCR via Roboflow API
                            ocr_result = client.ocr_image(inference_input=roi, model="trocr")
                            ocr_text = ocr_result.get("result", "")
                            # TROCR doesn't provide confidence scores, use a default high value
                            avg_confidence = 0.9 if ocr_text else 0.0
                        
                        # Draw OCR text on the annotated frame
                        if ocr_text:
                            # Position text above or below the detection box with proper spacing
                            if y > text_height + text_margin:
                                text_y = y - text_margin
                            else:
                                text_y = y + h + text_height + text_margin
                            
                            # Draw shadow (black text slightly offset)
                            cv2.putText(annotated, ocr_text, (x + 2, text_y + 2),
                                       cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 2)
                            # Draw main text (yellow)
                            cv2.putText(annotated, ocr_text, (x, text_y),
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
    fourcc = cv2.VideoWriter.fourcc(*"avc1")
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
    parser.add_argument("--config", type=str, default="config_roboflow_ocr_demo.yml", help="Path to YAML configuration file")
    parser.add_argument("--input-video", type=str, help="Input video path (overrides config file)")
    parser.add_argument("--duration", type=str, default=None, help="Duration limit: None (1 frame), 'full' (all frames), or float (seconds)")
    parser.add_argument("--font-size", type=float, default=CONFIG['defaults']['font_size'], help="Font size scale for annotation text (default: from config)")
    parser.add_argument("--use-trocr", action="store_true", default=CONFIG['defaults']['use_trocr'], help="Use TROCR model (default: from config)")
    parser.add_argument("--use-easyocr", action="store_true", default=CONFIG['defaults']['use_easyocr'], help="Use EasyOCR model (default: from config)")
    parser.add_argument("--use-mmocr", action="store_true", default=CONFIG['defaults']['use_mmocr'], help="Use MMOCR model (default: from config)")
    parser.add_argument("--trocr-ckpt", type=str, default=CONFIG['trocr']['default'], 
                       choices=CONFIG['trocr'].get('checkpoints', ["trocr-small-printed", "trocr-base-printed", "trocr-large-printed"]),
                       help="TROCR checkpoint (default: from config)")
    parser.add_argument("--easyocr-ckpt", type=str, default=CONFIG['easyocr']['default_languages'], 
                       help="EasyOCR language codes, comma-separated (default: from config)")
    parser.add_argument("--mmocr-det-model-name", type=str, default=MMOCR_DET_MODEL_NAME, help="MMOCR detection model name (default: from config)")
    parser.add_argument("--mmocr-recog-model-name", type=str, default=MMOCR_RECOG_MODEL_NAME, help="MMOCR recognition model name (default: from config)")
    parser.add_argument("--mmocr-det-model-version", type=str, default=MMOCR_DET_MODEL_VERSION, help="MMOCR detection model version (default: from config)")
    parser.add_argument("--mmocr-recog-model-version", type=str, default=MMOCR_RECOG_MODEL_VERSION, help="MMOCR recognition model version (default: from config)")
    args = parser.parse_args()
    
    # Reload configuration with user-specified config file
    CONFIG = load_config(args.config)
    
    # Parse duration argument (priority: command line > config file)
    duration = args.duration if args.duration is not None else CONFIG['defaults']['duration']
    if duration is not None and duration != 'full':
        try:
            duration = float(duration)
        except ValueError:
            raise ValueError(f"Invalid duration: {duration}. Must be None, 'full', or a float.")
    
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
    
    # Configuration from config file
    WORKSPACE = CONFIG['model']['workspace']
    MODEL_NAME = CONFIG['model']['model_name']
    VERSION = CONFIG['model']['version']
    MODEL_ID = f"{MODEL_NAME}/{VERSION}"
    
    # Determine input video path (priority: command line > config file)
    if args.input_video:
        INPUT_VIDEO = Path(args.input_video)
    elif CONFIG['video']['input_path']:
        INPUT_VIDEO = Path(CONFIG['video']['input_path'])
    else:
        INPUT_VIDEO = None
    
    TEST_VIDEO = CONFIG['video']['test_video_name']

    # Check if input video exists, if not create a test video
    if INPUT_VIDEO is None or not os.path.exists(INPUT_VIDEO):
        if INPUT_VIDEO is not None:
            print(f"Warning: Input video {INPUT_VIDEO} not found.")
        else:
            print("Warning: No input video specified in config or command line.")
        
        if not os.path.exists(TEST_VIDEO):
            create_test_video(TEST_VIDEO, duration=CONFIG['video']['test_video_duration'], font_size=args.font_size)
        INPUT_VIDEO = Path(TEST_VIDEO)
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
        det_name = args.mmocr_det_model_name
        rec_name = args.mmocr_recog_model_name
        det_version = args.mmocr_det_model_version
        rec_version = args.mmocr_recog_model_version
        model_descriptor = f"mmocr_{det_name}_v{det_version}_{rec_name}_v{rec_version}"
    elif args.use_trocr:
        model_descriptor = f"trocr_{args.trocr_ckpt}"
    else:
        model_descriptor = f"easyocr_{args.easyocr_ckpt.replace(',', '-')}"
    
    # Create output directory structure
    out_root = Path("data/bonting-identification/processed_videos/tracking")
    out_dir = (
        out_root
        / category_folder
        / video_path.stem
        / f"{time_tag}_model=roboflow_ocr={model_descriptor}"
    )
    
    print(f"[+] Using config file: {args.config}")
    print(f"[+] Using model: {MODEL_ID}")
    if args.use_mmocr:
        print(f"[+] OCR model: MMOCR")
        print(f"[+] OCR models: det={args.mmocr_det_model_name} v{args.mmocr_det_model_version}, recog={args.mmocr_recog_model_name} v{args.mmocr_recog_model_version}")
    elif args.use_trocr:
        print(f"[+] OCR model: TROCR")
        print(f"[+] OCR checkpoint: {args.trocr_ckpt}")
    else:
        print(f"[+] OCR model: EasyOCR")
        print(f"[+] OCR checkpoint: {args.easyocr_ckpt}")
    print(f"[+] Input video: {INPUT_VIDEO}")
    print(f"[+] Output directory: {out_dir}")
    
    # Prepare all parameters from config and command line arguments
    inference_params = {
        'model_id': MODEL_ID,
        'api_key': API_KEY,
        'input_video': str(INPUT_VIDEO),
        'output_dir': str(out_dir),
        'conf': CONFIG['defaults']['confidence_threshold'],
        'font_size': args.font_size,
        'duration': duration,
        'use_trocr': args.use_trocr,
        'use_easyocr': args.use_easyocr,
        'use_mmocr': args.use_mmocr,
        'trocr_ckpt': args.trocr_ckpt,
        'easyocr_ckpt': args.easyocr_ckpt,
        'mmocr_det_model_name': args.mmocr_det_model_name,
        'mmocr_recog_model_name': args.mmocr_recog_model_name,
        'mmocr_det_model_version': args.mmocr_det_model_version,
        'mmocr_recog_model_version': args.mmocr_recog_model_version
    }
    
    infer_video_with_roboflow_ocr(**inference_params)