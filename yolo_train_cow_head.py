#!/usr/bin/env python3
import os
from datetime import datetime
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO, settings
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
from utils.mlflow_flavours import YoloUltralyticsFlavor

load_dotenv()

def main(
    rf_project_name="cow-heads-varied-heoza",
    rf_version=2,
    epochs=100,
    imgsz=640,
):
    # Ensure Ultralytics uses MLflow and that we keep its run open after training
    settings.update({"mlflow": True})
    original_cwd = os.getcwd()
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("defaultworkspace-z97gj").project(rf_project_name)
    version = project.version(rf_version)
    download_dir = os.path.join(original_cwd, "tmp", "roboflow", "datasets", rf_project_name, str(rf_version))
    dataset = version.download(model_format="yolov12", location=download_dir, overwrite=True)

    data_yaml = os.path.join(dataset.location, "data.yaml")
    run_base = "yolo_cow_head"
    os.makedirs(run_base, exist_ok=True)
    run_name = "yolo12_cow_head_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    model = YOLO("yolo12n.pt")

    # Configure MLflow run metadata for Ultralytics callbacks via env variables
    # Ultralytics reads these in on_pretrain_routine_end and starts the run
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "yolo_cow_head")
    os.environ["MLFLOW_RUN"] = run_name
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "True"
    runs_dir = os.path.join(original_cwd, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", os.path.join(runs_dir, "mlflow"))

    try:
        os.chdir(runs_dir)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            project=run_base,
            name=run_name,
        )

        run_dirs = [
            os.path.join(run_base, d)
            for d in os.listdir(run_base)
            if os.path.isdir(os.path.join(run_base, d))
        ]
        if not run_dirs:
            raise FileNotFoundError("No training run directories found")
        latest_run = max(run_dirs, key=lambda d: os.path.getmtime(d))
        best_weights_path = os.path.join(latest_run, "weights", "best.pt")
        if not os.path.isfile(best_weights_path):
            raise FileNotFoundError("best.pt not found after training")

        # Do not start a new MLflow run here; reuse the Ultralytics one
        trained = YOLO(best_weights_path)
        torch_model = trained.model

        # Attach to the same active run Ultralytics created
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError(
                "No active MLflow run found after training. Ensure Ultralytics MLflow is enabled and set MLFLOW_KEEP_RUN_ACTIVE=True."
            )

        # Log and register to the same run using custom Ultralytics flavor
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("imgsz", imgsz)
        best_weights_abs = os.path.abspath(best_weights_path)
        YoloUltralyticsFlavor.log(weights_path=best_weights_abs, artifact_path="model", task="detect")
        model_uri = f"runs:/{active_run.info.run_id}/model"
        registered_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "yolo12_cow_head")
        mlflow.register_model(model_uri=model_uri, name=registered_name)
        client = MlflowClient()
        client.set_registered_model_tag(registered_name, "task_name", "det_cow_head")
        client.set_registered_model_tag(registered_name, "flavour", YoloUltralyticsFlavor.FLAVOR_NAME)

        # Close the Ultralytics-created run that we kept open
        mlflow.end_run()
    finally:
        # Always restore original working directory, even on Ctrl+C
        try:
            os.chdir(original_cwd)
        except Exception:
            pass


if __name__ == "__main__":
    main()

