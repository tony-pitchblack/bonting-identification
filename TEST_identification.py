#!/usr/bin/env python
"""Run run_identification.py with a tiny dummy video."""
from __future__ import annotations

import subprocess
from pathlib import Path
import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
TMP_VIDEO = THIS_DIR / "_dummy_vid.mp4"


def make_dummy_video(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from {path}")

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (w, h), True)
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    cv2.putText(frame, "dummy", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    writer.write(frame)

    for _ in range(3):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()
    cap.release()


def main() -> None:
    make_dummy_video(TMP_VIDEO)
    cmd = [
        "python",
        str(THIS_DIR / "run_identification.py"),
        "--input",
        str(TMP_VIDEO),
        "--yolo-ckpt",
        "yolov8n.pt",
        "--trocr",
        "microsoft/trocr-small-printed",
        "--det-obj-name",
        "ear-tag",
        "--duration",
        "0.5",
        "--dry-run",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
