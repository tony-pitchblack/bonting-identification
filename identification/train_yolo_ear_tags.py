#!/usr/bin/env python
"""Fine-tune YOLO detection model for cow ear tags."""
from __future__ import annotations

import argparse
from pathlib import Path

import os
import shutil

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


def _resolve_yaml(path: Path) -> Path:
    """Return a YAML file path given either a file or directory."""
    if path.suffix in {".yaml", ".yml"}:  # direct file input
        return path

    # search common filenames first
    for fname in ("dataset.yaml", "data.yaml"):
        candidate = path / fname
        if candidate.exists():
            return candidate

    # fallback: first *.yaml / *.yml file in the directory (non-recursive)
    for pattern in ("*.yaml", "*.yml"):
        candidates = list(path.glob(pattern))
        if candidates:
            return candidates[0]

    # recursive search as a last resort
    for pattern in ("*.yaml", "*.yml"):
        candidates = list(path.rglob(pattern))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"No YAML dataset description found in {path}")


def main() -> None:
    args = parse_args()
    if args.data:
        yaml_path = _resolve_yaml(Path(args.data))
    else:
        env_dir = os.environ.get("EAR_TAG_DATASET")
        if env_dir:
            yaml_path = _resolve_yaml(Path(env_dir))
        else:
            d = download_dataset()
            if d is None:
                print("Dataset unavailable. Exiting.")
                return
            yaml_path = _resolve_yaml(d)

    # -------------------------------------------------------------- weights
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(exist_ok=True)

    model_file = "yolov8n.pt"
    local_weights = ckpt_dir / model_file

    if local_weights.exists():
        print(f"Loading YOLO model from local weights: {local_weights} ...")
        model = YOLO(str(local_weights))
    else:
        print(f"Downloading {model_file} to {ckpt_dir} ...")
        # Download to a temp location first – Ultralytics handles this internally
        model = YOLO(model_file)

        # Move the downloaded file into our ckpt directory for future runs
        src_path = Path(model.ckpt_path)
        if src_path.exists():
            shutil.copy2(src_path, local_weights)
            print(f"Moved weights to {local_weights}")
            # Delete the original file to save space
            if src_path != local_weights:
                src_path.unlink(missing_ok=True)
                print(f"Deleted source weights at {src_path}")
            # Reload model with the moved weights to ensure consistent path
            model = YOLO(str(local_weights))
        else:
            print(f"Warning: Could not find downloaded weights at {src_path}")

    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=640, batch=args.batch,
                device=args.device, workers=args.workers)


if __name__ == "__main__":
    main()
