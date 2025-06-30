#!/usr/bin/env python
"""Fine-tune YOLO detection model for cow ear tags."""
from __future__ import annotations

import argparse
from pathlib import Path

import os

from ultralytics import YOLO


DATASET_HANDLE = "fandaoerji/cow-eartag-detection-dataset"


def download_dataset() -> Path | None:
    """Download dataset from Kaggle if possible."""
    try:
        import kagglehub
    except Exception as e:
        print(f"kagglehub not installed: {e}")
        return None
    try:
        path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"Path to dataset files: {path}")
        return Path(path)
    except Exception as e:
        print(f"Dataset download failed: {e}")
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO on ear-tag dataset")
    p.add_argument("--data", type=str, help="Path to dataset directory or dataset.yaml")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--workers", type=int, default=2, help="Data loader workers")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.data:
        data_path = Path(args.data)
        yaml_path = data_path if data_path.suffix == ".yaml" else data_path / "dataset.yaml"
    else:
        env_dir = os.environ.get("EAR_TAG_DATASET")
        if env_dir:
            yaml_path = Path(env_dir) / "dataset.yaml"
        else:
            d = download_dataset()
            if d is None:
                print("Dataset unavailable. Exiting.")
                return
            yaml_path = d / "dataset.yaml"

    model = YOLO("yolov8n.pt")
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=640, batch=args.batch,
                device=args.device, workers=args.workers)


if __name__ == "__main__":
    main()
