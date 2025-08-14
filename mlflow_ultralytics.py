#!/usr/bin/env python3
import os
import shutil
import tempfile
from typing import Optional

import mlflow
from mlflow.models import Model
from ultralytics import YOLO


def save_model(weights_path: str, path: str, task: str = "detect") -> str:
    """
    Save an Ultralytics YOLO model as a custom MLflow flavor.

    - weights_path: path to a .pt weights file compatible with YOLO()
    - path: target directory to create the MLflow model (contains MLmodel + data/)
    - task: optional task metadata (e.g., 'detect')

    Returns the directory where the model was saved (path).
    """
    os.makedirs(path, exist_ok=True)
    data_dir = os.path.join(path, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Store weights under a stable relative path
    rel_weights = os.path.join("data", "model.pt")
    dst_weights = os.path.join(path, rel_weights)
    shutil.copy2(weights_path, dst_weights)

    # Write MLmodel file with custom flavor metadata
    mlmodel = Model()
    mlmodel.add_flavor(
        "ultralytics",
        weights=rel_weights,
        task=task,
        framework="ultralytics",
    )
    mlmodel.save(os.path.join(path, "MLmodel"))
    return path


def log_model(weights_path: str, artifact_path: str = "model", task: str = "detect") -> None:
    """
    Log an Ultralytics YOLO model to the active MLflow run using the custom flavor.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("No active MLflow run. Start or attach to a run before calling log_model().")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "ultralytics_model")
        save_model(weights_path=weights_path, path=model_dir, task=task)
        mlflow.log_artifacts(model_dir, artifact_path=artifact_path)


def load_model(model_uri: str) -> YOLO:
    """
    Load an Ultralytics YOLO model from an MLflow model URI saved with this flavor.

    Returns a YOLO() instance initialized with the stored weights.
    """
    local_path = mlflow.artifacts.download_artifacts(model_uri)
    mlmodel = Model.load(os.path.join(local_path, "MLmodel"))
    flavor = mlmodel.flavors.get("ultralytics")
    if not flavor or "weights" not in flavor:
        raise ValueError("The specified MLflow model does not contain the 'ultralytics' flavor.")

    weights_rel = flavor["weights"]
    weights_abs = os.path.join(local_path, weights_rel)
    if not os.path.isfile(weights_abs):
        raise FileNotFoundError(f"Weights not found at expected path: {weights_abs}")
    return YOLO(weights_abs)


__all__ = [
    "save_model",
    "log_model",
    "load_model",
]

