#!/usr/bin/env python
"""Fine-tune YOLO detection model for cow ear tags."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
from typing import Dict, Any, Optional
import re

import mlflow
from dotenv import load_dotenv
from ultralytics import YOLO


DATASET_HANDLE = "fandaoerji/cow-eartag-detection-dataset"


# --------------------------- logging helpers ---------------------------------

def setup_logging(logging_mode: str, experiment_name: str) -> object:
    """Configure MLflow based on the selected logging mode.

    Parameters
    ----------
    logging_mode : str
        One of {'mlflow', 'databricks'}
    experiment_name : str
        Desired experiment name (will be auto-prefixed for Databricks if needed)

    Returns
    -------
    mlflow module for logging.
    """
    logging_mode = logging_mode.lower()

    if logging_mode == "databricks":
        # Load credentials from .env
        load_dotenv()
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")

        if not host or not token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set in .env for Databricks logging.")

        mlflow.set_tracking_uri(tracking_uri)
        # Auto-prefix experiment path if not absolute
        if not experiment_name.startswith("/"):
            experiment_name = f"/Shared/{experiment_name}"
        print(f"Using Databricks MLflow tracking at {host} (experiment '{experiment_name}')")
        # Disable global autologging to avoid Spark init and background threads
        mlflow.autolog(disable=True)

    elif logging_mode == "mlflow":
        # Pure MLflow (local or custom HTTP) – NEVER fall back to Databricks here.
        load_dotenv()
        uri = os.getenv("MLFLOW_TRACKING_URI")

        if uri and uri.lower() != "databricks":
            # Respect user-provided URI that is NOT the magic 'databricks'
            mlflow.set_tracking_uri(uri)
            print(f"Using MLflow tracking URI: {uri} (experiment '{experiment_name}')")
        else:
            # Default to local file store under ./mlruns
            default_uri = f"file:{Path('mlruns').absolute()}"
            mlflow.set_tracking_uri(default_uri)
            print(f"Using local MLflow file store at {default_uri} (experiment '{experiment_name}')")

        # Disable global autologging integrations to avoid unwanted Spark init
        mlflow.autolog(disable=True)
    else:
        raise ValueError("--logging must be one of 'mlflow', 'databricks'")

    # Ensure experiment exists / select it
    mlflow.set_experiment(experiment_name)
    return mlflow


def _sanitize_tag(name: str) -> str:
    """Sanitize MLflow metric/param names to allowed characters."""
    return re.sub(r"[^0-9a-zA-Z _\-\./:]", "_", name)


def log_metrics(mlflow_mod: object | None, metrics: Dict[str, Any]) -> None:
    """Log numeric metrics if MLflow is active, with name sanitization."""
    if mlflow_mod is None:
        return
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        safe_k = _sanitize_tag(k)
        try:
            mlflow_mod.log_metric(safe_k, v)
        except Exception as e:
            print(f"Skipping metric {k}: {e}")


# ----------------------------------------------------------------------------


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

    p.add_argument(
        "--logging",
        type=str,
        choices=["mlflow", "databricks"],
        default="mlflow",
        help="Logging backend: mlflow | databricks",
    )
    p.add_argument("--experiment", type=str, default="EarTagYOLO", help="MLflow experiment name")
    p.add_argument(
        "--fraction",
        type=float,
        help="Fraction of the training dataset to use (0.0 < f ≤ 1.0)",
    )
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


# --------------------------- main -------------------------------------------


def main() -> None:
    args = parse_args()

    # Configure logging / MLflow
    mlflow_mod = setup_logging(args.logging, args.experiment)

    # Resolve dataset YAML
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

    # --------------- helper to run training -----------------
    def _run_training() -> tuple[object, "YOLO"]:  # type: ignore
        script_dir = Path(__file__).resolve().parent
        ckpt_dir = Path("ckpt")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_file = "yolov8n.pt"
        local_weights = ckpt_dir / model_file

        if local_weights.exists():
            print(f"Loading YOLO model from local weights: {local_weights} ...")
            model = YOLO(str(local_weights))
        else:
            print(f"Downloading {model_file} to {ckpt_dir} ...")
            model = YOLO(model_file)
            src_path = Path(model.ckpt_path)
            if src_path.exists():
                shutil.copy2(src_path, local_weights)
                print(f"Moved weights to {local_weights}")
                if src_path != local_weights:
                    src_path.unlink(missing_ok=True)
                    print(f"Deleted source weights at {src_path}")
                model = YOLO(str(local_weights))
            else:
                print(f"Warning: Could not find downloaded weights at {src_path}")

        results = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            fraction=args.fraction if args.fraction else 1.0,
        )
        return results, model

    # --------------- run with or without MLflow ----------------
    with mlflow_mod.start_run():
        # Log parameters
        mlflow_mod.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch,
            "device": args.device,
            "workers": args.workers,
            "fraction": args.fraction if args.fraction else 1.0,
            "dataset": str(yaml_path),
        })

        results, model = _run_training()
        if hasattr(results, "results_dict"):
            log_metrics(mlflow_mod, results.results_dict)  # type: ignore[arg-type]
        # Ultralytics already logs weights; skip manual artifact to avoid duplicate uploads


if __name__ == "__main__":
    main()
    print('done')
