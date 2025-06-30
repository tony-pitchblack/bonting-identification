#!/usr/bin/env python
"""Test fine-tuning with a tiny dummy dataset on CPU."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


THIS_DIR = Path(__file__).resolve().parent
TMP_ROOT = THIS_DIR / "_dummy_data"


def make_dummy_dataset() -> Path:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    (TMP_ROOT / "images/train").mkdir(parents=True)
    (TMP_ROOT / "images/val").mkdir(parents=True)
    (TMP_ROOT / "labels/train").mkdir(parents=True)
    (TMP_ROOT / "labels/val").mkdir(parents=True)

    for split in ["train", "val"]:
        img_path = TMP_ROOT / f"images/{split}/img0.jpg"
        img = Image.new("RGB", (640, 640), color="black")
        draw = ImageDraw.Draw(img)
        draw.rectangle([280, 280, 360, 360], outline="white", fill="white")
        img.save(img_path)
        # YOLO format: class x_center y_center width height (normalized)
        box = [0.5, 0.5, 80 / 640, 80 / 640]
        label_path = TMP_ROOT / f"labels/{split}/img0.txt"
        with open(label_path, "w") as f:
            f.write(f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n")

    yaml_path = TMP_ROOT / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {TMP_ROOT}\ntrain: images/train\nval: images/val\nnames: ['ear-tag']\n"
        )
    return yaml_path


def main() -> None:
    yaml_path = make_dummy_dataset()
    cmd = [
        "python",
        str(THIS_DIR / "train_yolo_ear_tags.py"),
        "--data",
        str(yaml_path),
        "--epochs",
        "1",
        "--batch",
        "1",
        "--device",
        "cpu",
        "--workers",
        "0",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
