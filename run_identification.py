#!/usr/bin/env python
"""Track cows and identify them by ear tags."""
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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

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


# --------------------------------------------------------------------------- #
# main routine                                                                #
# --------------------------------------------------------------------------- #

def process_video(
    video_path: Path,
    tracker_name: str,
    yolo_ckpt: str,
    trocr_ckpt: str,
    det_obj_name: str,
    duration_s: float | None = None,
    n_debug_images: int = 0,
    out_root: Path = Path("data/HF_dataset/processed_videos/identification"),
    dry_run: bool = False,
) -> None:
    print(f"\nProcessing video: {video_path.name}")
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(exist_ok=True)

    if not dry_run:
        # ------------------------------------------------------------- load yolo
        weight_name = Path(yolo_ckpt).name
        local_weights = ckpt_dir / weight_name
        if Path(yolo_ckpt).exists():
            model = YOLO(str(yolo_ckpt))
        elif local_weights.exists():
            model = YOLO(str(local_weights))
        else:
            print(f"Downloading {yolo_ckpt} to {local_weights} ...")
            tmp_model = YOLO(yolo_ckpt)
            src = Path(tmp_model.ckpt_path)
            if src.exists():
                shutil.copy2(src, local_weights)
                src.unlink()
            model = YOLO(str(local_weights))
        print("YOLO model loaded")

        # ------------------------------------------------------------ load trocr
        print(f"Loading TrOCR model {trocr_ckpt} ...")
        processor = TrOCRProcessor.from_pretrained(trocr_ckpt, cache_dir=str(ckpt_dir))
        trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_ckpt, cache_dir=str(ckpt_dir))
        trocr_model.to("cpu")
        trocr_model.eval()
        print("TrOCR loaded")
    else:
        model = None
        processor = None
        trocr_model = None

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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps if fps > 0 else 30, (width, height))

    if dry_run:
        cap = cv2.VideoCapture(str(video_path))
        id_frames: Dict[int, List[int]] = {}
        tag_numbers: Dict[int, str] = {}
        for frame_i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, "dummy", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            writer.write(frame)
            id_frames.setdefault(0, []).append(frame_i)
            tag_numbers[0] = "0"
        cap.release()
        writer.release()
        timestamps_df = _slice_segments(id_frames, fps)
        timestamps_df.to_csv(out_dir / "tracking_timestamps.csv", index=False)
        pd.DataFrame([{"id": 0, "ear_tag": tag_numbers.get(0, "")}]).to_csv(out_dir / "id_to_tag.csv", index=False)
        print(f"Outputs saved to {out_dir}")
        return

    results = model.track(
        source=str(video_path),
        tracker=f"{tracker_name}.yaml",
        stream=True,
        imgsz=640,
        verbose=True,
    )

    id_frames: Dict[int, List[int]] = {}
    tag_numbers: Dict[int, str] = {}
    frame_i = -1

    obj_key = det_obj_name.lower().replace("-", "").replace("_", "").replace(" ", "")

    # prepare debug directory if requested
    if n_debug_images > 0:
        dbg_dir = Path("tmp/debug_trocr") / video_path.stem
        dbg_dir.mkdir(parents=True, exist_ok=True)

    for r in tqdm(results, total=max_frames, desc=f"Tracking {video_path.stem}"):
        frame_i += 1
        if duration_s is not None and frame_i >= max_frames:
            break
        orig = r.orig_img
        annotated = r.plot(line_width=1)

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
            name = model.names[int(cls_id)]
            label_norm = name.lower().replace("-", "").replace("_", "").replace(" ", "")
            if "cow" in name.lower():
                if obj_id is not None:
                    cows.append((box.tolist(), obj_id))
                    id_frames.setdefault(obj_id, []).append(frame_i)
            elif obj_key in label_norm:
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
            if cid in tag_numbers:
                cv2.putText(
                    annotated,
                    tag_numbers[cid],
                    (int(tag_box[0]), int(tag_box[1]) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
        writer.write(annotated)

        # ------------------------------------------------------ debug logging
        if n_debug_images > 0 and frame_i % n_debug_images == 0:
            dbg_path = dbg_dir / f"frame_{frame_i:06d}.jpg"
            cv2.imwrite(str(dbg_path), annotated)

    writer.release()

    timestamps_df = _slice_segments(id_frames, fps)
    timestamps_df.to_csv(out_dir / "tracking_timestamps.csv", index=False)

    mapping = pd.DataFrame(
        [
            {"id": cid, "ear_tag": tag_numbers.get(cid, "")}
            for cid in sorted(id_frames.keys())
        ]
    )
    mapping.to_csv(out_dir / "id_to_tag.csv", index=False)
    print(f"Outputs saved to {out_dir}")


def cli() -> None:
    p = argparse.ArgumentParser(description="Track cattle and identify ear tags")
    p.add_argument("--input", required=True, help="Video file or folder of videos")
    p.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    p.add_argument("--yolo-ckpt", default="yolo11m.pt", help="YOLO weights path or name")
    p.add_argument("--trocr", default="microsoft/trocr-base-handwritten", help="TrOCR model checkpoint")
    p.add_argument("--duration", type=float, help="Process only first N seconds of each video")
    p.add_argument("--det-obj-name", default="ear-tag", help="Name of object class for tags")
    p.add_argument("--n_debug_images", type=int, default=0, help="Save every Nth annotated frame for debugging TrOCR processing (0 = disabled)")
    p.add_argument("--dry-run", action="store_true", help="Skip model loading for tests")
    args = p.parse_args()

    videos = _collect_videos(Path(args.input))
    if not videos:
        raise SystemExit("No video files found")

    for vid in videos:
        process_video(
            vid,
            args.tracker,
            args.yolo_ckpt,
            args.trocr,
            args.det_obj_name,
            args.duration,
            n_debug_images=args.n_debug_images,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    cli()
