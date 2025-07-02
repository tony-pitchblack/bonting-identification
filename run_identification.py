#!/usr/bin/env python
"""Track cows and identify them by ear tags."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
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
    for cid, frames in id_frames.items():
        frames.sort()
        start = frames[0]
        for i in range(1, len(frames)):
            if frames[i] != frames[i - 1] + 1:
                rows.append((cid, start / fps, frames[i - 1] / fps))
                start = frames[i]
        rows.append((cid, start / fps, frames[-1] / fps))
    return pd.DataFrame(rows, columns=["id", "start_ts", "end_ts"]).sort_values(["id", "start_ts"])


def _collect_videos(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXT:
        return [path]
    if path.is_dir():
        return [p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXT]
    raise FileNotFoundError(f"{path} is neither a video nor a folder with videos.")


def _find_cow_for_tag(tag_box, cows: List[Tuple[List[float], int]]) -> int | None:
    cx = (tag_box[0] + tag_box[2]) / 2
    cy = (tag_box[1] + tag_box[3]) / 2
    for box, cid in cows:
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            return cid
    return None


class ModelWrapper:
    """Wrapper to handle both YOLO and Roboflow models with unified interface."""
    
    def __init__(self, roboflow_ckpt: str | None, ultralytics_ckpt: str):
        self.is_roboflow = roboflow_ckpt is not None
        self.tracker = sv.ByteTrack() if self.is_roboflow else None
        
        if self.is_roboflow:
            self._load_roboflow_model(roboflow_ckpt)
        else:
            self.model = self._load_ultralytics_model(ultralytics_ckpt)
    
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
    
    def _load_ultralytics_model(self, ultralytics_ckpt: str) -> YOLO:
        """Load Ultralytics YOLO model."""
        ckpt_dir = Path("ckpt")
        ckpt_dir.mkdir(exist_ok=True)
        
        weight_name = Path(ultralytics_ckpt).name if ultralytics_ckpt.endswith('.pt') else f"{ultralytics_ckpt}.pt"
        local_weights = ckpt_dir / weight_name
        
        if Path(ultralytics_ckpt).exists():
            model = YOLO(str(ultralytics_ckpt))
        elif local_weights.exists():
            model = YOLO(str(local_weights))
        else:
            print(f"Downloading {ultralytics_ckpt} to {local_weights}")
            tmp_model = YOLO(ultralytics_ckpt if ultralytics_ckpt.endswith('.pt') else f"{ultralytics_ckpt}.pt")
            src = Path(tmp_model.ckpt_path)
            if src.exists():
                shutil.copy2(src, local_weights)
                src.unlink()
            model = YOLO(str(local_weights))
        print("YOLO model loaded")
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


def _load_model(roboflow_ckpt: str | None, ultralytics_ckpt: str) -> ModelWrapper:
    """Load model wrapper for either Roboflow or Ultralytics models."""
    return ModelWrapper(roboflow_ckpt, ultralytics_ckpt)


def _validate_class_names(model: ModelWrapper, class_names: List[str]) -> None:
    """Validate that all required class names are supported by the model."""
    names = model.names if model.is_roboflow else model.model.names
    supported_classes = list(names.values())
    
    for class_name in class_names:
        if class_name not in supported_classes:
            raise ValueError(
                f"Class '{class_name}' is not supported by the model. "
                f"Supported classes: {supported_classes}"
            )


def _draw_filtered_annotations(frame, detections, model: ModelWrapper, class_name: str, tag_numbers=None, tracker_ids=None):
    """Draw bounding boxes only for the specified class, with tag numbers if available."""
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
                labels = []
                for i in range(len(filtered_detections)):
                    label = f"{class_name}"
                    if tracker_ids and i < len(tracker_ids):
                        track_id = tracker_ids[i]
                        label += f" ID:{track_id}"
                        if tag_numbers and track_id in tag_numbers:
                            label += f" Tag:{tag_numbers[track_id]}"
                    labels.append(label)
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
                
                # Draw class name, ID, and tag number if available
                label = f"{detected_class}"
                if obj_id is not None:
                    label += f" ID:{obj_id}"
                    if tag_numbers and obj_id in tag_numbers:
                        label += f" Tag:{tag_numbers[obj_id]}"
                
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
    tracker_name: str,
    roboflow_ckpt: str | None,
    ultralytics_ckpt: str,
    trocr_ckpt: str,
    class_name: str,
    device: str = "cpu",
    duration_s: float | None = None,
    n_debug_images: int = 0,
    out_root: Path = Path("data/HF_dataset/processed_videos/identification"),
    gt_mapping_path: Path | None = None,
) -> None:
    print(f"\nProcessing video: {video_path.name}")
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------- load yolo
    model = _load_model(roboflow_ckpt, ultralytics_ckpt)
    # Note: For identification, we need both cow and tag classes. 
    # The class_name parameter specifies which objects to visualize (default: cow)
    # We still detect tags for OCR processing but may not visualize them
    required_classes = ["cow"]
    if class_name != "cow":
        required_classes.append(class_name)
    _validate_class_names(model, required_classes)
    print(f"Validated class names: {', '.join(required_classes)}")

    # ------------------------------------------------------------ load trocr
    print(f"Loading TrOCR model {trocr_ckpt}")
    
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(exist_ok=True)

    processor = TrOCRProcessor.from_pretrained(
        trocr_ckpt,
        cache_dir=str(ckpt_dir),
        use_fast=False,
    )
    trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_ckpt, cache_dir=str(ckpt_dir))
    trocr_model.to(device)
    trocr_model.eval()

    # -------------------------------------------------------------- video meta
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if duration_s is not None and fps > 0:
        max_frames = min(frame_count, int(duration_s * fps))
    else:
        max_frames = frame_count

    # output directory
    time_tag = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_abs = video_path.resolve()
    category_folder = "unknown"
    if "source_videos" in video_abs.parts:
        idx = video_abs.parts.index("source_videos")
        if idx + 1 < len(video_abs.parts):
            category_folder = video_abs.parts[idx + 1]
    out_dir = out_root / category_folder / video_path.stem / f"{time_tag}_tracker={tracker_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / "processed_video.mp4"

    first_frame = None
    writer = None

    id_frames: Dict[int, List[int]] = {}
    tag_numbers: Dict[int, str] = {}
    frame_i = -1

    # We'll look for ear tags even if class_name is set to cow (for identification purposes)
    tag_class_candidates = ["ear-tag", "tag", "ear_tag", "eartag"]

    if model.is_roboflow:
        # Process frame by frame for Roboflow models
        cap = cv2.VideoCapture(str(video_path))
        tracker = sv.ByteTrack() if tracker_name == "bytetrack" else sv.ByteTrack()  # Default to ByteTrack
        
        for frame_i in tqdm(range(max_frames), desc=f"Processing {video_path.stem}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            orig = frame
            
            # Get detections from Roboflow
            detections, _ = model.predict_frame(frame)
            
            # Update tracker
            detections = tracker.update_with_detections(detections)
            
            cows: List[Tuple[List[float], int]] = []
            tag_boxes: List[List[float]] = []
            
            # Process detections
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                names = model.names
                tracker_ids = detections.tracker_id if hasattr(detections, 'tracker_id') else None
                
                for i, cls_id in enumerate(detections.class_id):
                    detected_class = names.get(cls_id, f"class_{cls_id}")
                    label_norm = detected_class.lower().replace("-", "").replace("_", "").replace(" ", "")
                    
                    if tracker_ids is not None and i < len(tracker_ids):
                        track_id = tracker_ids[i]
                        if track_id >= 0:  # Valid tracking ID
                            box = detections.xyxy[i].tolist()
                            
                            if "cow" in detected_class.lower():
                                cows.append((box, track_id))
                                id_frames.setdefault(track_id, []).append(frame_i)
                            elif any(tag_candidate.replace("-", "").replace("_", "") in label_norm for tag_candidate in tag_class_candidates):
                                tag_boxes.append(box)
            
            # Process ear tags for OCR
            for tag_box in tag_boxes:
                cid = _find_cow_for_tag(tag_box, cows)
                if cid is None:
                    continue
                if cid not in tag_numbers:
                    x1, y1, x2, y2 = map(int, tag_box)
                    crop = orig[y1:y2, x1:x2]
                    if crop.size:
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        with torch.no_grad():
                            pix = processor(images=pil_img, return_tensors="pt").pixel_values
                            gen_ids = trocr_model.generate(pix)
                            txt = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                        tag_numbers[cid] = txt.strip()
            
            # Create annotation
            tracker_ids = detections.tracker_id if hasattr(detections, 'tracker_id') else None
            annotated = _draw_filtered_annotations(orig, detections, model, class_name, tag_numbers, tracker_ids)
            
            # store the first annotated frame
            if frame_i == 0:
                first_frame = annotated.copy()

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
            save_crop=True,
            verbose=False,  # Disable YOLO's built-in progress bar
        )

        for r in tqdm(results, total=max_frames, desc=f"Processing {video_path.stem}"):
            frame_i += 1
            if duration_s is not None and frame_i >= max_frames:
                break
            orig = r.orig_img
            
            # Create custom annotation with only the specified class
            annotated = _draw_filtered_annotations(orig, r, model, class_name, tag_numbers)

            cows: List[Tuple[List[float], int]] = []
            tag_boxes: List[List[float]] = []

            ids_list = (
                []
                if r.boxes.id is None
                else r.boxes.id.cpu().numpy().astype(int).tolist()
            )
            for box, cls_id, obj_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                ids_list,
            ):
                name = model.model.names[int(cls_id)]
                label_norm = name.lower().replace("-", "").replace("_", "").replace(" ", "")
                if "cow" in name.lower():
                    if obj_id is not None:
                        cows.append((box.tolist(), obj_id))
                        id_frames.setdefault(obj_id, []).append(frame_i)
                elif any(tag_candidate.replace("-", "").replace("_", "") in label_norm for tag_candidate in tag_class_candidates):
                    tag_boxes.append(box.tolist())

            for tag_box in tag_boxes:
                cid = _find_cow_for_tag(tag_box, cows)
                if cid is None:
                    continue
                if cid not in tag_numbers:
                    x1, y1, x2, y2 = map(int, tag_box)
                    crop = orig[y1:y2, x1:x2]
                    if crop.size:
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        with torch.no_grad():
                            pix = processor(images=pil_img, return_tensors="pt").pixel_values
                            gen_ids = trocr_model.generate(pix)
                            txt = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                        tag_numbers[cid] = txt.strip()

            # store the first annotated frame
            if frame_i == 0:
                first_frame = annotated.copy()

            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video_path), fourcc, fps if fps > 0 else 30, (w, h))
            writer.write(annotated)

    if writer is not None:
        writer.release()
    
    # ------------------------------------------------------ save first frame after processing
    debug_path = Path("tmp/debug_trocr/first_frame.png")
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    if first_frame is not None:
        cv2.imwrite(str(debug_path), first_frame)

    timestamps_df = _slice_segments(id_frames, fps)
    timestamps_df.to_csv(out_dir / "tracking_timestamps.csv", index=False)

    mapping = pd.DataFrame(
        [
            {"id": cid, "ear_tag": tag_numbers.get(cid, "")}
            for cid in sorted(id_frames.keys())
        ]
    )
    mapping.to_csv(out_dir / "id_to_tag.csv", index=False)

    # ------------------------------------------------------ optional metric computation
    metrics_rows = []
    n_total_ids = len(mapping)
    n_recognized = int(mapping["ear_tag"].fillna("").str.len().gt(0).sum())
    recognition_rate = n_recognized / n_total_ids if n_total_ids else 0.0
    metrics_rows.append({"metric": "n_total_ids", "value": n_total_ids})
    metrics_rows.append({"metric": "n_recognized_ids", "value": n_recognized})
    metrics_rows.append({"metric": "recognition_rate", "value": recognition_rate})

    # If ground-truth mapping is supplied and readable, compute accuracy.
    if gt_mapping_path is not None and gt_mapping_path.exists():
        try:
            gt_df = pd.read_csv(gt_mapping_path)
            if {"id", "ear_tag"}.issubset(gt_df.columns):
                merged = pd.merge(gt_df, mapping, on="id", how="left", suffixes=("_gt", "_pred"))
                merged["correct"] = merged["ear_tag_gt"].astype(str).str.strip() == merged["ear_tag_pred"].astype(str).str.strip()
                accuracy = merged["correct"].mean() if len(merged) else 0.0
                metrics_rows.append({"metric": "accuracy", "value": accuracy})
        except Exception:
            pass

    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)


def cli() -> None:
    p = argparse.ArgumentParser(description="Track cattle and identify ear tags")
    p.add_argument("input", help="Video file or folder of videos")
    p.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    p.add_argument("--roboflow-ckpt", help="Roboflow model checkpoint (e.g., 'username/model-id/version')")
    p.add_argument("--ultralytics-ckpt", default="yolo11m", help="Ultralytics YOLO checkpoint (default: yolo11m)")
    p.add_argument("--trocr", default="microsoft/trocr-base-handwritten", help="TrOCR model checkpoint")
    p.add_argument("--duration", type=float, help="Process only first N seconds of each video")
    p.add_argument("--class-name", default="cow", help="Name of object class to visualize in output video (default: cow)")
    p.add_argument("--n_debug_images", type=int, default=0, help="Save every Nth annotated frame for debugging TrOCR processing (0 = disabled)")
    p.add_argument("--out-root", default="data/CEID-D/processed_videos/identification", help="Root output directory compliant with CEID-D dataset structure")
    p.add_argument("--gt-mapping", help="CSV with ground-truth id→ear_tag mapping to compute accuracy metrics")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to use for TrOCR processing")
    args = p.parse_args()

    if args.roboflow_ckpt and args.ultralytics_ckpt != "yolo11m":
        raise ValueError("Cannot specify both --roboflow-ckpt and --ultralytics-ckpt")

    videos = _collect_videos(Path(args.input))
    if not videos:
        raise SystemExit("No video files found")

    for i, vid in enumerate(videos, 1):
        if len(videos) > 1:
            print(f"\nProcessing video {i}/{len(videos)}: {vid.name}")
        process_video(
            vid,
            args.tracker,
            args.roboflow_ckpt,
            args.ultralytics_ckpt,
            args.trocr,
            args.class_name,
            args.device,
            args.duration,
            n_debug_images=args.n_debug_images,
            out_root=Path(args.out_root),
            gt_mapping_path=Path(args.gt_mapping) if args.gt_mapping else None,
        )


if __name__ == "__main__":
    cli()
