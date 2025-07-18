"""Minimal hook that logs input datasets to MLflow once per run.
Assumes every dataset dict contains:
    'data_root': <path>

and
    'metainfo': {'dataset_name': <str>}
    # or dataset objects that provide the same key directly

Place the file anywhere on the Python path and add
    custom_hooks = [dict(type='MlflowDatasetHook', priority='LOW')]
inside the training config.
"""
# mmocr_custom/hooks/mlflow_dataset_hook.py

import os
import pandas as pd
import mlflow
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.dataset import ConcatDataset
from pathlib import Path

@HOOKS.register_module()
class MlflowDatasetHook(Hook):
    """Log train/test datasets via MLflow Data API so they appear in the Datasets filter."""

    priority = 'LOW'

    def _create_metadata_dataset(self, dataset):
        """Create a metadata-only MLflow dataset for the given dataset object."""
        # Import here to avoid mypy issues with missing stubs for ``mlflow.data``
        import mlflow.data as mldata  # type: ignore

        # Expect dataset_name to be stored at the top-level of metainfo
        name = dataset.metainfo["dataset_name"]
        source = dataset.data_root

        # A metadata-only dataset is created by passing an **empty** dataframe
        return mldata.from_pandas(pd.DataFrame(), source=source, name=name)  # type: ignore[attr-defined]

    def _log_dataset(self, ds, context):
        """Log the provided dataset (or its sub-datasets) to MLflow."""
        if isinstance(ds, ConcatDataset):
            for sub_ds in ds.datasets:
                ml_ds = self._create_metadata_dataset(sub_ds)
                mlflow.log_input(ml_ds, context=context)
            return

        ml_ds = self._create_metadata_dataset(ds)
        mlflow.log_input(ml_ds, context=context)

    def before_train(self, runner):
        # Log cfg file name as a simple MLflow param
        cfg = getattr(runner, 'cfg', None)
        if cfg and getattr(cfg, 'filename', None):
            cfg_name = Path(cfg.filename).name
            mlflow.log_param("config_filename", cfg_name)

        self._log_dataset(runner.train_dataloader.dataset, context='training')

    def before_test(self, runner):
        self._log_dataset(runner.test_dataloader.dataset, context='test')
