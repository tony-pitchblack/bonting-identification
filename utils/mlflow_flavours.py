#!/usr/bin/env python3
import os
import tempfile
from typing import Any

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from ultralytics import YOLO


class YoloUltralyticsFlavor:
    FLAVOR_NAME = "yolo_ultralytics"

    class _PyfuncModel(pyfunc.PythonModel):
        def __init__(self) -> None:
            self._yolo = None
            self._weights_path = None

        def load_context(self, context: pyfunc.PythonModelContext) -> None:
            self._weights_path = context.artifacts["weights"]
            self._yolo = YOLO(self._weights_path)

        def predict(self, context: pyfunc.PythonModelContext, model_input: Any) -> Any:
            return self._yolo(model_input)

    @classmethod
    def save_model(
        cls,
        path: str,
        *,
        weights_path: str,
        task: str = "detect",
        **kwargs,
    ) -> None:
        os.makedirs(path, exist_ok=True)
        # Persist as a pyfunc model with weights as an artifact (no loader_module required)
        pyfunc.save_model(
            path=path,
            python_model=cls._PyfuncModel(),
            artifacts={"weights": weights_path},
        )
        # Append custom flavor metadata
        mlmodel = Model.load(os.path.join(path, MLMODEL_FILE_NAME))
        mlmodel.add_flavor(
            cls.FLAVOR_NAME,
            # Older models may have stored only the artifacts/weights path (a file without extension).
            # Keep pointing to the artifact key path; loader will handle both file and directory cases.
            weights=os.path.join("artifacts", "weights"),
            task=task,
            framework="ultralytics",
        )
        mlmodel.save(os.path.join(path, MLMODEL_FILE_NAME))

    # Backward-compatible convenience alias
    @classmethod
    def save(cls, weights_path: str, path: str, task: str = "detect") -> str:
        cls.save_model(path=path, weights_path=weights_path, task=task)
        return path

    @classmethod
    def log_model(
        cls,
        artifact_path: str,
        *,
        weights_path: str,
        task: str = "detect",
        **kwargs,
    ) -> Model:
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Start or attach to a run before calling log_model().")
        return Model.log(
            artifact_path=artifact_path,
            flavor=cls,  # calls cls.save_model under the hood
            weights_path=weights_path,
            task=task,
            **kwargs,
        )

    # Backward-compatible convenience alias
    @classmethod
    def log(cls, weights_path: str, artifact_path: str = "model", task: str = "detect") -> None:
        cls.log_model(artifact_path=artifact_path, weights_path=weights_path, task=task)

    @classmethod
    def load_model(cls, model_uri: str) -> YOLO:
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        mlmodel = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))
        flavor = mlmodel.flavors.get(cls.FLAVOR_NAME)
        if not flavor or "weights" not in flavor:
            raise ValueError(f"The specified MLflow model does not contain the '{cls.FLAVOR_NAME}' flavor.")
        weights_rel = flavor["weights"]
        weights_abs = cls._find_weights(local_path, weights_rel)
        if not weights_abs:
            raise FileNotFoundError(
                f"Weights not found. Tried hint '{weights_rel}' and common fallbacks within: {local_path}"
            )
        return YOLO(weights_abs)

    # Helper used by load_model to locate weights robustly
    @classmethod
    def _find_weights(cls, base_dir: str, hint_rel: str) -> str:
        candidate = os.path.join(base_dir, hint_rel)
        # If exact path is a file, use it directly
        if os.path.isfile(candidate):
            return candidate
        # If it's a directory, search for common weight filenames inside
        if os.path.isdir(candidate):
            preferred_names = [
                "model.pt",
                "best.pt",
                "weights.pt",
                "model.pth",
                "best.pth",
                "weights.pth",
            ]
            for name in preferred_names:
                p = os.path.join(candidate, name)
                if os.path.isfile(p):
                    return p
            # Fallback: any .pt or .pth
            for root, _, files in os.walk(candidate):
                for f in files:
                    if f.endswith((".pt", ".pth")):
                        return os.path.join(root, f)
        # Try other common locations within the model folder
        common_relatives = [
            os.path.join("artifacts", "weights", "model.pt"),
            os.path.join("artifacts", "weights", "best.pt"),
            os.path.join("data", "model.pt"),
            os.path.join("data", "model.pth"),
        ]
        for rel in common_relatives:
            p = os.path.join(base_dir, rel)
            if os.path.isfile(p):
                return p
        # Final fallback: any .pt or .pth anywhere under base_dir
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith((".pt", ".pth")):
                    return os.path.join(root, f)
        return ""


__all__ = ["YoloUltralyticsFlavor"]

