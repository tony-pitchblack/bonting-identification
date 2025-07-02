#!/usr/bin/env python
"""
Identify timestamps where each uniquely tracking cow is visible.

Examples
--------
# single video, pixel-accurate masks, ByteTrack IDs
python track_animals.py --input videos/pen1.mp4 --mode segment --tracker bytetrack

# process every video in a folder, bbox-only, BoTSORT IDs
python track_animals.py --input videos/ --mode detect --tracker botsort

# use roboflow model
python track_animals.py --input videos/pen1.mp4 --roboflow-ckpt my-model/version --class-name cow
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
import supervision as sv
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

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


class ModelWrapper:
    """Wrapper to handle both YOLO and Roboflow models with unified interface."""
    
    def __init__(self, roboflow_ckpt: str | None, ultralytics_ckpt: str, mode: str):
        self.is_roboflow = roboflow_ckpt is not None
        self.mode = mode
        self.tracker = sv.ByteTrack() if self.is_roboflow else None
        
        if self.is_roboflow:
            self._load_roboflow_model(roboflow_ckpt)
        else:
            self.model = self._load_ultralytics_model(ultralytics_ckpt, mode)
    
    def _load_roboflow_model(self, roboflow_ckpt: str):
        """Load Roboflow model using new interface."""
        try:
            from roboflow import Roboflow
        except ImportError as e:
            raise RuntimeError(
                "Roboflow package not installed. Install with 'pip install roboflow'"
            ) from e
        
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Environment variable ROBOFLOW_API_KEY not set. It is required to "
                "use Roboflow models."
            )
        
        rf = Roboflow(api_key=api_key)
        
        # Parse ckpt like "workspace/project/3" or "project/3"
        parts = roboflow_ckpt.split("/")
        if len(parts) == 3:
            workspace, project_slug, version = parts
            project = rf.workspace(workspace).project(project_slug)
        elif len(parts) == 2:
            project_slug, version = parts
            project = rf.workspace().project(project_slug)
        else:
            raise ValueError(
                "--roboflow-ckpt must be 'project_slug/version' or "
                "'workspace/project_slug/version'"
            )
        
        self.rf_model = project.version(int(version)).model
        print(f"Loaded Roboflow model: {roboflow_ckpt}")
        
        # Get model class names by making a dummy prediction
        import tempfile
        import numpy as np
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Create a small dummy image
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, dummy_img)
            try:
                result = self.rf_model.predict(tmp.name, confidence=1, overlap=30).json()
                # Extract unique class names from the model
                self.names = {}
                if 'predictions' in result:
                    for pred in result['predictions']:
                        if 'class' in pred:
                            # We'll build the names dict as we encounter classes
                            pass
                # For now, assume standard classes - this will be updated during processing
                self.names = {0: 'cow', 1: 'ear-tag'}  # Default mapping
            except:
                self.names = {0: 'cow', 1: 'ear-tag'}  # Fallback
            finally:
                os.unlink(tmp.name)
    
    def _load_ultralytics_model(self, ultralytics_ckpt: str, mode: str) -> YOLO:
        """Load Ultralytics YOLO model."""
        ckpt_dir = Path("ckpt")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        if mode == "segment" and not ultralytics_ckpt.endswith("-seg.pt"):
            if ultralytics_ckpt.endswith(".pt"):
                model_file = ultralytics_ckpt.replace(".pt", "-seg.pt")
            else:
                model_file = f"{ultralytics_ckpt}-seg.pt"
        else:
            model_file = ultralytics_ckpt if ultralytics_ckpt.endswith(".pt") else f"{ultralytics_ckpt}.pt"
        
        local_weights = ckpt_dir / Path(model_file).name

        if local_weights.exists():
            print(f"Loading YOLO model from local weights: {local_weights}")
            model = YOLO(str(local_weights))
        else:
            print(f"Downloading {model_file} to {ckpt_dir}")
            model = YOLO(model_file)
            src_path = Path(model.ckpt_path)
            if src_path.exists():
                shutil.copy2(src_path, local_weights)
                print(f"Moved weights to {local_weights}")
                src_path.unlink()
                print(f"Deleted source weights at {src_path}")
                model = YOLO(str(local_weights))
            else:
                print(f"Warning: Could not find downloaded weights at {src_path}")

        print("YOLO model loaded.")
        return model
    
    def predict_frame(self, frame, confidence: float = 0.25, overlap: float = 0.30):
        """Predict on a single frame and return standardized detections."""
        if self.is_roboflow:
            # Save frame temporarily for roboflow prediction
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                try:
                    result = self.rf_model.predict(tmp.name, confidence=int(confidence*100), overlap=int(overlap*100)).json()
                    detections = sv.Detections.from_inference(result)
                    
                    # Update names dict with classes found in this prediction
                    if 'predictions' in result:
                        for i, pred in enumerate(result['predictions']):
                            if 'class' in pred:
                                class_name = pred['class']
                                if class_name not in self.names.values():
                                    # Find next available index
                                    next_idx = max(self.names.keys()) + 1 if self.names else 0
                                    self.names[next_idx] = class_name
                    
                    return detections, result
                finally:
                    os.unlink(tmp.name)
        else:
            # Use YOLO prediction
            results = self.model(frame, verbose=False)
            return results[0], None
    
    def track(self, source: str, **kwargs):
        """Track objects across video frames."""
        if self.is_roboflow:
            raise NotImplementedError("Roboflow models require frame-by-frame processing")
        else:
            return self.model.track(source=source, **kwargs)


def _load_model(roboflow_ckpt: str | None, ultralytics_ckpt: str, mode: str) -> ModelWrapper:
    """Load model wrapper for either Roboflow or Ultralytics models."""
    return ModelWrapper(roboflow_ckpt, ultralytics_ckpt, mode)


def _validate_class_name(model: ModelWrapper, class_name: str) -> None:
    """Validate that the class name is supported by the model."""
    names = model.names if model.is_roboflow else model.model.names
    if class_name not in names.values():
        supported_classes = list(names.values())
        raise ValueError(
            f"Class '{class_name}' is not supported by the model. "
            f"Supported classes: {supported_classes}"
        )


def _draw_filtered_annotations(frame, detections, model: ModelWrapper, class_name: str, tracker_ids=None):
    """Draw bounding boxes only for the specified class."""
    annotated = frame.copy()
    
    if detections is None:
        return annotated
    
    if model.is_roboflow:
        # Handle supervision detections from roboflow
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Filter detections for the specified class
        if hasattr(detections, 'class_id') and detections.class_id is not None:
            class_names = [model.names.get(cls_id, f"class_{cls_id}") for cls_id in detections.class_id]
            mask = [name == class_name for name in class_names]
            if any(mask):
                filtered_detections = detections[mask]
                labels = [f"{class_name} ID:{tracker_ids[i] if tracker_ids and i < len(tracker_ids) else 'N/A'}" 
                         for i in range(len(filtered_detections))]
                annotated = box_annotator.annotate(scene=annotated, detections=filtered_detections)
                annotated = label_annotator.annotate(scene=annotated, detections=filtered_detections, labels=labels)
    else:
        # Handle YOLO detections
        if detections.boxes is None or len(detections.boxes) == 0:
            return annotated
        
        boxes_np = detections.boxes.xyxy.cpu().numpy()
        classes_np = detections.boxes.cls.cpu().numpy().astype(int)
        ids_np = detections.boxes.id.cpu().numpy().astype(int) if detections.boxes.id is not None else [None] * len(boxes_np)
        
        for box, cls_id, obj_id in zip(boxes_np, classes_np, ids_np):
            detected_class = model.model.names[cls_id]
            if detected_class == class_name:
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw class name and ID
                label = f"{detected_class}"
                if obj_id is not None:
                    label += f" ID:{obj_id}"
                
                # Calculate label position
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
                
                # Draw label background
                cv2.rectangle(annotated, (x1, label_y - label_size[1] - 5), 
                             (x1 + label_size[0] + 5, label_y + 5), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(annotated, label, (x1 + 2, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated


# --------------------------------------------------------------------------- #
# main routine                                                                #
# --------------------------------------------------------------------------- #
def process_video(
    video_path: Path,
    mode: str,
    tracker_name: str,
    roboflow_ckpt: str | None,
    ultralytics_ckpt: str,
    class_name: str,
    duration_s: float | None = None,
    device: str = "cpu",
    out_root: Path = Path("data/HF_dataset/processed_videos/tracking"),
) -> None:
    # ------------------------------------------------------------------ model
    model = _load_model(roboflow_ckpt, ultralytics_ckpt, mode)
    _validate_class_name(model, class_name)
    print(f"Validated class name: {class_name}")

    # ----------------------------------------------------------- video meta
    print("Reading video metadata")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    # ---------------------------------------------------------- run tracker
    print(f"Running tracker ({tracker_name})")
    id_frames: Dict[int, List[int]] = {}
    frame_i = -1

    # ------------------------------------------------------------- outputs
    time_tag = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_abs_path = video_path.resolve()
    
    category_folder = "unknown"
    if "source_videos" in video_abs_path.parts:
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

    # Initialize video writer
    writer = None
    out_video_path = out_dir / "processed_video.mp4"

    if duration_s is not None and fps > 0:
        max_frames = min(frame_count, int(duration_s * fps))
        print(f"Processing first {duration_s} seconds (~{max_frames} frames)")
    else:
        max_frames = frame_count
        print("Processing frames and tracking objects")

    if model.is_roboflow:
        # Process frame by frame for Roboflow models
        cap = cv2.VideoCapture(str(video_path))
        tracker = sv.ByteTrack() if tracker_name == "bytetrack" else sv.ByteTrack()  # Default to ByteTrack
        
        for frame_i in tqdm(range(max_frames), desc=f"Tracking {video_path.stem}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections from Roboflow
            detections, _ = model.predict_frame(frame)
            
            # Update tracker
            detections = tracker.update_with_detections(detections)
            
            # Filter for specified class and collect tracking data
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                names = model.names if model.is_roboflow else model.model.names
                for i, cls_id in enumerate(detections.class_id):
                    detected_class = names.get(cls_id, f"class_{cls_id}")
                    if detected_class == class_name and hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                        track_id = detections.tracker_id[i]
                        if track_id >= 0:  # Valid tracking ID
                            id_frames.setdefault(track_id, []).append(frame_i)
            
            # Create annotation
            tracker_ids = detections.tracker_id if hasattr(detections, 'tracker_id') else None
            annotated = _draw_filtered_annotations(frame, detections, model, class_name, tracker_ids)
            
            # Initialize video writer
            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video_path), fourcc, fps if fps > 0 else 30, (w, h))
            
            writer.write(annotated)
        
        cap.release()
    else:
        # Use YOLO's built-in tracking for Ultralytics models
        results = model.track(
            source=str(video_path),
            tracker=f"{tracker_name}.yaml",
            stream=True,
            imgsz=640,
            verbose=False,
        )

        for r in tqdm(results, desc=f"Tracking {video_path.stem}", total=max_frames):
            frame_i += 1
            
            # Create custom annotation with only the specified class
            annotated = _draw_filtered_annotations(r.orig_img, r, model, class_name)
            
            # Initialize video writer with actual frame dimensions
            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video_path), fourcc, fps if fps > 0 else 30, (w, h))
            
            writer.write(annotated)
            
            # Track only the specified class
            if r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int).tolist()
                classes = r.boxes.cls.cpu().numpy().astype(int).tolist()
                
                for cid, cls_id in zip(ids, classes):
                    detected_class = model.model.names[cls_id]
                    if detected_class == class_name:
                        id_frames.setdefault(cid, []).append(frame_i)

            if duration_s is not None and frame_i + 1 >= max_frames:
                break

    if writer is not None:
        writer.release()

    print(f"Tracking complete. Processed {frame_i + 1} frames")
    print(f"Found {len(id_frames)} unique tracking IDs for class '{class_name}'")

    # ---------------------------------------------------------- build timestamps
    timestamps_df = _slice_segments(id_frames, fps)

    print("Saving tracking timestamps CSV")
    # — timestamps CSV
    timestamps_df.to_csv(out_dir / "tracking_timestamps.csv", index=False)


def cli() -> None:
    print("Starting animal tracking")
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
        "--roboflow-ckpt",
        help="Roboflow model checkpoint (e.g., 'username/model-id/version')",
    )
    p.add_argument(
        "--ultralytics-ckpt",
        default="yolo11m",
        help="Ultralytics model checkpoint (default: yolo11m)",
    )
    p.add_argument(
        "--class-name",
        default="cow",
        help="Name of object class to track (default: cow)",
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

    if args.roboflow_ckpt and args.ultralytics_ckpt != "yolo11m":
        raise ValueError("Cannot specify both --roboflow-ckpt and --ultralytics-ckpt")

    print("Configuration:")
    print(f"  Input: {args.input}")
    print(f"  Mode: {args.mode}")
    print(f"  Tracker: {args.tracker}")
    print(f"  Roboflow checkpoint: {args.roboflow_ckpt}")
    print(f"  Ultralytics checkpoint: {args.ultralytics_ckpt}")
    print(f"  Class name: {args.class_name}")
    print(f"  Duration: {args.duration} seconds")

    videos = _collect_videos(Path(args.input))
    if not videos:
        raise SystemExit("❌ No video files found.")

    print(f"\nProcessing {len(videos)} video(s)")
    for i, vid in enumerate(videos, 1):
        print(f"\n--- Video {i}/{len(videos)} ---")
        process_video(
            vid,
            args.mode,
            args.tracker,
            args.roboflow_ckpt,
            args.ultralytics_ckpt,
            args.class_name,
            args.duration,
            args.device,
        )

    print("All videos processed successfully.")


if __name__ == "__main__":
    cli()
