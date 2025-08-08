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
import json
import math
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
# Camera parameter helpers
# -----------------------------------------------------------------------------

def load_camera_params(scene_config_path: Path, camera_name: str):
    """Return sensor size and focal length values for a named camera.

    Returns
    -------
    tuple(float, float, float, float)
        sensor_width_mm, sensor_height_mm, focal_length_x_mm, focal_length_y_mm
    """
    with open(scene_config_path, "r") as f:
        data = json.load(f)

    cams = data.get("cameras", {})
    if camera_name not in cams:
        available = ", ".join(cams.keys())
        raise ValueError(
            f"Camera '{camera_name}' not found in {scene_config_path}. Available: {available}"
        )
    cam = cams[camera_name]

    sensor_w_mm = float(cam["sensor_mm"]["width"])
    sensor_h_mm = float(cam["sensor_mm"]["height"])

    focal_val = cam.get("focal_length_mm")
    # `focal_length_mm` can be either a single float **or** a dict with axis keys.
    if isinstance(focal_val, dict):
        focal_mm = float(focal_val.get("x", next(iter(focal_val.values()))))
    else:
        focal_mm = float(focal_val)

    focal_x_mm = focal_mm
    focal_y_mm = focal_mm  # identical – avoids double-scaling

    return sensor_w_mm, sensor_h_mm, focal_x_mm, focal_y_mm

# -----------------------------------------------------------------------------
# Annotation helpers
# -----------------------------------------------------------------------------

def annotate_image(
    img_bgr: np.ndarray,
    preds: list,
    depth: np.ndarray,
    font_scale: float,
    sensor_w_mm: float,
    sensor_h_mm: float,
    focal_x_mm: float,
    focal_y_mm: float,
    camera_pitch_deg: float = 36.0,
) -> np.ndarray:
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

                # Depth value gives distance along the camera ray; convert to true Euclidean distance
        z_val = float(depth[cy, cx])  # metres
        u = cx - w / 2.0
        v = cy - h / 2.0
        fx_px = focal_x_mm * w / sensor_w_mm
        fy_px = focal_y_mm * h / sensor_h_mm
        scale = math.sqrt(1.0 + (u / fx_px) ** 2 + (v / fy_px) ** 2)
        distance = z_val * scale

        # -----------------------------------------------------------------
        # Compute projected distance onto room OX axis (horizontal ground plane)
        # -----------------------------------------------------------------
        yaw_rad = math.atan2(u, fx_px)
        pitch_rad = math.atan2(-v, fy_px)  # negative v so upward is positive
        ab = distance * math.cos(yaw_rad)
        alpha_deg = 90.0 - (math.degrees(pitch_rad) + camera_pitch_deg)
        dist_ox = ab * math.cos(math.radians(alpha_deg))

        label = pred.get("class", "cow-head")
        text = f"{label} (Eu {distance:.2f}m) (OX {dist_ox:.2f}m)"

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
    sensor_w_mm: float,
    sensor_h_mm: float,
    focal_x_mm: float,
    focal_y_mm: float,
    camera_pitch_deg: float = 36.0,
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
        annotated = annotate_image(
            img_bgr,
            preds,
            depth_map,
            font_scale,
            sensor_w_mm,
            sensor_h_mm,
            focal_x_mm,
            focal_y_mm,
            camera_pitch_deg,
        )

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

    defaults_cfg = cfg.get("defaults", {})
    font_scale = defaults_cfg.get("font_size", 0.5) * defaults_cfg.get("font_scale_multiplier", 1.2)
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

    # ---------------------------------------------------------------------
    # Load camera parameters for true Euclidean distance calculation
    # ---------------------------------------------------------------------
    scene_config_path = Path(cfg["scene_config_path"]).expanduser()
    camera_name = cfg.get("camera_name", "Camera.front.upper")
    sensor_w_mm, sensor_h_mm, focal_x_mm, focal_y_mm = load_camera_params(
        scene_config_path, camera_name
    )

    camera_pitch_deg = cfg.get("camera_pitch_deg", 36.0)

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
        sensor_w_mm=sensor_w_mm,
        sensor_h_mm=sensor_h_mm,
        focal_x_mm=focal_x_mm,
        focal_y_mm=focal_y_mm,
        camera_pitch_deg=camera_pitch_deg,
    )