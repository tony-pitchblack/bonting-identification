#!/usr/bin/env python
"""Run track_identify_animals with a tiny dummy video."""
from __future__ import annotations

import subprocess
from pathlib import Path
import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
TMP_VIDEO = THIS_DIR / "_dummy_vid.mp4"


def make_dummy_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (320, 240))
    for _ in range(3):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(img)
    writer.release()


def main() -> None:
    make_dummy_video(TMP_VIDEO)
    cmd = [
        "python",
        str(THIS_DIR / "track_identify_animals.py"),
        "--input",
        str(TMP_VIDEO),
        "--yolo-ckpt",
        "yolov8n.pt",
        "--trocr",
        "microsoft/trocr-small-printed",
        "--duration",
        "0.5",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
