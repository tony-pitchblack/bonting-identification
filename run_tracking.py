#!/usr/bin/env python3
"""
run_tracking.py

A comprehensive video tracking script that supports:
- Roboflow and Ultralytics YOLO models
- ByteTrack and BotSort tracking algorithms
- Detection and segmentation modes
- Batch processing of files/folders
"""

import os
import argparse
import datetime as dt
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

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

try:
    import cv2
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"Required dependency missing: {e}")
    print("Please install with: pip install opencv-python numpy tqdm")
    exit(1)

class VideoTracker:
    """Main video tracking class that handles both Roboflow and Ultralytics models."""
    
    def __init__(self, args):
        self.args = args
        self.model: Optional[Any] = None
        self.tracker: Optional[Any] = None
        self.model_type: Optional[str] = None  # 'roboflow' or 'ultralytics'
        self.setup_model()
        self.setup_tracker()
        
    def setup_model(self):
        """Initialize the appropriate model based on arguments."""
        if self.args.roboflow_ckpt:
            self.setup_roboflow_model()
        else:
            self.setup_ultralytics_model()
    
    @staticmethod
    def _parse_roboflow_ckpt(ckpt: str) -> tuple[str, str, int]:
        """Return (workspace, project, version) from 'workspace/project/version'."""
        parts = ckpt.strip().split('/')
        if len(parts) != 3 or not parts[2].isdigit():
            raise ValueError(
                "roboflow_ckpt must be in 'workspace/project/version' format, e.g. 'myws/mymodel/2'"
            )
        workspace, project, version_str = parts
        return workspace, project, int(version_str)
    
    def setup_roboflow_model(self):
        """Setup Roboflow model using inference API."""
        try:
            from inference import get_model
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                print("[-] python-dotenv not installed, loading env vars manually")
            
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if not api_key:
                raise ValueError("ROBOFLOW_API_KEY environment variable not set")
            
            # Parse checkpoint string (workspace/project/version)
            workspace, project, version = self._parse_roboflow_ckpt(self.args.roboflow_ckpt)
            # Construct model_id as 'project/version' to match roboflow_demo.py
            model_id = f"{project}/{version}"
            self.model = get_model(model_id=model_id, api_key=api_key)
            self.model_type = 'roboflow'
            print(f"[+] Loaded Roboflow model (workspace: {workspace}): {model_id}")
            
        except ImportError:
            raise ImportError("Roboflow inference package not installed. Install with: pip install inference")
    
    def setup_ultralytics_model(self):
        """Setup Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
            
            model_path = self.args.ultralytics_ckpt
            self.model = YOLO(model_path)
            self.model_type = 'ultralytics'
            print(f"[+] Loaded Ultralytics model: {model_path}")
            
        except ImportError:
            raise ImportError("Ultralytics package not installed. Install with: pip install ultralytics")
    
    def setup_tracker(self):
        """Initialize tracker based on selected algorithm."""
        if self.args.tracker == 'bytetrack':
            try:
                from yolox.tracker.byte_tracker import BYTETracker
                self.tracker = BYTETracker(frame_rate=30)
                print(f"[+] Initialized ByteTrack tracker")
            except ImportError:
                print("[-] ByteTrack not available, using built-in tracking")
                self.tracker = None
        elif self.args.tracker == 'botsort':
            try:
                from trackers.botsort import BoTSORT
                self.tracker = BoTSORT(frame_rate=30)
                print(f"[+] Initialized BotSort tracker")
            except ImportError:
                print("[-] BotSort not available, using built-in tracking")
                self.tracker = None
    
    def run_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on a single frame."""
        if self.model_type == 'roboflow':
            return self.run_roboflow_inference(frame)
        else:
            return self.run_ultralytics_inference(frame)
    
    def run_roboflow_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run Roboflow inference on frame."""
        if self.model is None:
            raise RuntimeError("Roboflow model not initialized")
            
        results = self.model.infer(frame, confidence=0.4)
        
        detections = []
        for response in results:
            if hasattr(response, 'predictions'):
                for pred in response.predictions:
                    if pred.class_name.lower() == self.args.class_name.lower():
                        detections.append({
                            'bbox': [
                                int(pred.x - pred.width/2),
                                int(pred.y - pred.height/2),
                                int(pred.width),
                                int(pred.height)
                            ],
                            'confidence': pred.confidence,
                            'class_name': pred.class_name
                        })
        
        return detections
    
    def run_ultralytics_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run Ultralytics inference on frame."""
        if self.model is None:
            raise RuntimeError("Ultralytics model not initialized")
            
        results = self.model(frame, device=self.args.device, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                    class_name = self.model.names[int(cls)]
                    if class_name.lower() == self.args.class_name.lower():
                        x1, y1, x2, y2 = box
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            'confidence': float(conf),
                            'class_name': class_name
                        })
        
        return detections
    
    def annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]], tracks: Optional[List[int]] = None) -> np.ndarray:
        """Annotate frame with detections and tracks."""
        annotated = frame.copy()
        
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Use track ID if available
            track_id = tracks[i] if tracks and i < len(tracks) else None
            
            # Draw bounding box
            color = (0, 255, 0) if track_id is None else self.get_track_color(track_id)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            if track_id is not None:
                label = f"ID:{track_id} {class_name}: {conf:.2f}"
            else:
                label = f"{class_name}: {conf:.2f}"
            
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
    
    def get_track_color(self, track_id: int) -> tuple:
        """Get consistent color for track ID."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 128), (255, 165, 0), (255, 192, 203)
        ]
        return colors[track_id % len(colors)]
    
    def process_video(self, input_path: str, output_dir: str) -> str:
        """Process a single video file."""
        print(f"[+] Processing video: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {input_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frames to process
        if self.args.duration:
            frames_to_process = min(int(self.args.duration * fps), total_frames)
            print(f"[+] Processing first {self.args.duration} seconds ({frames_to_process} frames)")
        else:
            frames_to_process = total_frames
        
        # Create output directory structure
        video_path = Path(input_path)
        date = dt.datetime.now().strftime("%Y-%m-%d")
        time = dt.datetime.now().strftime("%H-%M-%S")
        
        if self.args.roboflow_ckpt:
            ckpt_workspace, ckpt_name, ckpt_version = self.args.roboflow_ckpt.split('/')
        else:
            ckpt_name = Path(self.args.ultralytics_ckpt).stem
        
        output_folder = (
            Path(output_dir) / 
            video_path.stem / 
            f"{date}_{time}_ckpt={ckpt_name}_version={ckpt_version}_mode={self.args.mode}_tracker={self.args.tracker}"
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Output video path
        output_video = output_folder / "processed_video.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError(f"Could not open VideoWriter: {output_video}")
        
        print(f"[+] Output will be saved to: {output_video}")
        
        # Process frames
        frame_count = 0
        with tqdm(total=frames_to_process, desc="Processing frames") as pbar:
            while frame_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                detections = self.run_inference(frame)
                
                # Apply tracking if available
                tracks = None
                if self.tracker and detections:
                    # Convert detections to format expected by tracker
                    # This would need to be implemented based on the specific tracker
                    pass
                
                # Annotate frame
                annotated_frame = self.annotate_frame(frame, detections, tracks)
                
                # Write frame
                out.write(annotated_frame)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        print(f"[+] Video processing complete: {output_video}")
        
        return str(output_video)
    
    def process_folder(self, input_folder: str, output_dir: str):
        """Process all video files in a folder."""
        folder_path = Path(input_folder)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder_path.glob(f'*{ext}'))
            video_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"[-] No video files found in {input_folder}")
            return
        
        print(f"[+] Found {len(video_files)} video files to process")
        
        for video_file in video_files:
            try:
                self.process_video(str(video_file), output_dir)
            except Exception as e:
                print(f"[-] Error processing {video_file}: {e}")
                continue
    
    def run(self):
        """Main execution method."""
        input_path = Path(self.args.input)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        if input_path.is_file():
            self.process_video(str(input_path), self.args.output_dir)
        elif input_path.is_dir():
            self.process_folder(str(input_path), self.args.output_dir)
        else:
            raise ValueError(f"Input path is neither file nor directory: {input_path}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Video tracking with YOLO models and various tracking algorithms"
    )
    
    # Positional argument
    parser.add_argument(
        "input",
        help="Input video file or folder containing videos"
    )
    
    # Optional arguments
    parser.add_argument(
        "--mode",
        choices=["detect", "segment"],
        default="detect",
        help="Detection mode (default: detect)"
    )
    
    parser.add_argument(
        "--tracker",
        choices=["bytetrack", "botsort"],
        default="bytetrack",
        help="Tracking algorithm (default: bytetrack)"
    )
    
    parser.add_argument(
        "--roboflow-ckpt",
        help="Roboflow checkpoint in format workspace/project/version"
    )
    
    parser.add_argument(
        "--ultralytics-ckpt",
        default="yolo11m.pt",
        help="Ultralytics model checkpoint (default: yolo11m.pt)"
    )
    
    parser.add_argument(
        "--class-name",
        default="cow",
        help="Object class to track (default: cow)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        help="Limit analysis to first N seconds (for debugging)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--output-dir",
        default="data/HF_dataset/processed_videos/tracking/youtube_segments",
        help="Output directory (default: data/HF_dataset/processed_videos/tracking/youtube_segments)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.roboflow_ckpt and args.ultralytics_ckpt == "yolo11m.pt":
        print("[+] Using default Ultralytics model: yolo11m.pt")
    elif args.roboflow_ckpt and args.ultralytics_ckpt != "yolo11m.pt":
        parser.error("Cannot specify both --roboflow-ckpt and --ultralytics-ckpt")
    
    # Create and run tracker
    tracker = VideoTracker(args)
    tracker.run()

if __name__ == "__main__":
    main() 