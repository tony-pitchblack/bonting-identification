#!/usr/bin/env python3
"""
roboflow_head_demo.py

Detect cow heads in a directory of images, compute distance using corresponding EXR depth maps, and save annotated images.
"""
from pathlib import Path
import argparse
import os
import cv2
import numpy as np
import OpenEXR
import Imath
import array
from tqdm import tqdm
import yaml
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import load_dotenv
from datetime import datetime

# -----------------------------------------------------------------------------
# Depth handling
# -----------------------------------------------------------------------------

def read_exr_depth(filepath: Path) -> np.ndarray:
    """Read single-channel depth data from an EXR file (returns meters)."""
    exr = OpenEXR.InputFile(str(filepath))
    dw = exr.header()["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = exr.header()["channels"].keys()
    channel = "Z" if "Z" in channels else ("R" if "R" in channels else list(channels)[0])

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr.channel(channel, pt)
    depth = np.array(array.array("f", depth_str), dtype=np.float32)
    depth = depth.reshape((height, width))
    return depth

# -----------------------------------------------------------------------------
# Annotation helpers
# -----------------------------------------------------------------------------

def annotate_image(img_bgr: np.ndarray, preds: list, depth: np.ndarray, font_scale: float) -> np.ndarray:
    """Draw bounding boxes, center dot and distance text for each prediction."""
    h, w = img_bgr.shape[:2]
    coords_texts = []
    for pred in preds:
        x1 = int(pred["x"] - pred["width"] / 2)
        y1 = int(pred["y"] - pred["height"] / 2)
        x2 = x1 + int(pred["width"])
        y2 = y1 + int(pred["height"])

        cx = int(pred["x"])
        cy = int(pred["y"])
        if not (0 <= cx < w and 0 <= cy < h):
            continue  # skip predictions falling outside frame
        coords_texts.append(f"({cx}, {cy})")

        distance = float(depth[cy, cx])
        label = pred.get("class", "cow-head")
        text = f"{label} ({distance:.2f} m)"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img_bgr, (cx, cy), 4, (0, 0, 255), -1)
        # Coordinate text inside bottom-left of bbox
        coord_text = f"({cx}, {cy})"
        ct_size, _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ct_w, ct_h = ct_size
        coord_x = max(0, min(w - ct_w, x1 + 2))
        coord_y = min(h - 1, y2 - 2)
        # shadow for coordinate text
        cv2.putText(img_bgr, coord_text, (coord_x + 1, coord_y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_bgr, coord_text, (coord_x, coord_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        margin = 15  # gap between bbox and text in pixels
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_w, text_h = text_size

        # Determine y coordinate for text to keep it within frame
        if y1 - margin - text_h >= 0:
            # draw above the bbox
            text_y = y1 - margin
        elif y2 + margin + text_h <= h:
            # draw below the bbox
            text_y = y2 + margin + text_h
        else:
            # fallback to keep text inside frame
            text_y = max(text_h, min(h - 1, y1))

        text_x = max(0, min(x1, w - text_w))
        text_pos = (text_x, text_y)

        # shadow
        cv2.putText(
            img_bgr,
            text,
            (text_pos[0] + 1, text_pos[1] + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        # actual text
        cv2.putText(
            img_bgr,
            text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    # -----------------------------------------------------------------
    # Display absolute coordinates of red centre point(s) bottom-left
    # -----------------------------------------------------------------
    if False:  # disabled global coordinate overlay
        overlay_margin = 10
        # Determine text height for spacing
        _, text_height = cv2.getTextSize(coords_texts[0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        y_start = h - overlay_margin
        for i, coord in enumerate(coords_texts):
            pos = (overlay_margin, y_start - i * (text_height + 5))
            # shadow
            cv2.putText(img_bgr, coord, (pos[0] + 1, pos[1] + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
            # text
            cv2.putText(img_bgr, coord, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr

# -----------------------------------------------------------------------------
# Main processing routine
# -----------------------------------------------------------------------------

def process_images(
    model_id: str,
    api_key: str,
    img_dir: Path,
    depth_dir: Path,
    output_dir: Path,
    api_url: str,
    conf: float,
    iou_threshold: float,
    font_scale: float,
):
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    client.configure(InferenceConfiguration(confidence_threshold=conf, iou_threshold=iou_threshold))

    output_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(img_dir.glob("*.png"))

    processed_files = []
    for img_path in tqdm(img_paths, desc="Processing images"):
        depth_path = depth_dir / img_path.with_suffix(".exr").name
        if not depth_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        preds = client.infer(img_bgr, model_id=model_id).get("predictions", [])
        depth_map = read_exr_depth(depth_path)
        annotated = annotate_image(img_bgr, preds, depth_map, font_scale)

        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), annotated)
        processed_files.append(save_path)
        if len(processed_files) == 1:
            print(f"First frame saved to: {save_path}")

    # Compile video from processed images
    if processed_files:
        first_id = int(Path(processed_files[0]).stem.lstrip("0") or 0)
        last_id = int(Path(processed_files[-1]).stem.lstrip("0") or 0)
        video_path = output_dir / f"output-{first_id}-{last_id}.mp4"

        first_frame = cv2.imread(str(processed_files[0]))
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))

        for frame_path in processed_files:
            frame = cv2.imread(str(frame_path))
            writer.write(frame)
        writer.release()
        print(f"Video saved to: {video_path}")

# -----------------------------------------------------------------------------
# Config loader
# -----------------------------------------------------------------------------

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cow head detection on image directory with depth")
    parser.add_argument("--config", type=str, default="config_roboflow_head_demo.yml", help="Path to YAML configuration file")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    img_dir = Path(cfg["input_img_dir"]).expanduser()
    depth_dir = Path(cfg["input_depth_dir"]).expanduser()

    font_scale = cfg["defaults"]["font_size"]
    conf = cfg["defaults"]["confidence_threshold"]
    iou_th = cfg["defaults"]["iou_threshold"]

    model_id = cfg["model"]["model_id"]
    api_url = cfg["model"]["api_url"]

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY environment variable not set")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        remove_prefix = cfg["output"].get("remove_input_prefix", "")
        rel_path = str(img_dir)
        if remove_prefix and remove_prefix in rel_path:
            rel_path = rel_path.split(remove_prefix, 1)[1]
        rel_path = rel_path.lstrip("/")
        out_root = Path(cfg["output"]["base_dir"])
        model_name = model_id.replace("/", "_")
        out_dir = out_root / rel_path / f"{time_tag}_model={model_name}"

    process_images(
        model_id=model_id,
        api_key=api_key,
        img_dir=img_dir,
        depth_dir=depth_dir,
        output_dir=out_dir,
        api_url=api_url,
        conf=conf,
        iou_threshold=iou_th,
        font_scale=font_scale,
    )