from __future__ import annotations

# mmocr_custom/hooks/mlflow_checkpoint_hook.py
# noqa: D401, E501
"""Checkpoint hook that also uploads checkpoints to MLflow.

It inherits from :class:`mmengine.hooks.CheckpointHook` and performs the
standard checkpointing *plus* an ``mlflow.log_artifact`` call so the saved
model files are stored as run artefacts.

The hook re-uses (and lightly re-implements) the environment-initialisation
logic found in :class:`mmengine.visualization.MLflowVisBackend` so it can be
used even when that backend is not enabled.
"""

import os.path as osp
from typing import Any, Dict, Optional

import mlflow  # type: ignore
from mmengine.hooks import CheckpointHook
from mmengine.registry import HOOKS
from mmengine.dist.utils import is_main_process

# Locking to the master rank is important so parallel processes don't all
# attempt to log the same file concurrently which can lead to MLflow errors.


@HOOKS.register_module()
class MlflowCheckpointHook(CheckpointHook):
    """Extension of :class:`CheckpointHook` that uploads checkpoints to MLflow.

    Additional keyword arguments (all optional and identical to those accepted
    by :class:`mmengine.visualization.MLflowVisBackend`) control how the
    *tracking* run is created.

    Args:
        tracking_uri (str, optional): MLflow tracking URI. If *None*, a local
            file store under ``work_dir/mlruns`` is used.
        exp_name (str): MLflow experiment name. Defaults to ``"Default"``.
        run_name (str, optional): Name of the run. If *None*, the default name
            assigned by MLflow is kept.
        tags (dict, optional): Tags to set on the run.
        params (dict, optional): Parameters to log once at the beginning.
        artifact_location (str, optional): Location to store run artefacts.
            Passed directly to :pyfunc:`mlflow.create_experiment` when a new
            experiment is created.
        artifact_subdir (str): Sub-directory inside the run to store the
            checkpoint files. Defaults to ``"checkpoints"``.

        **kwargs: All remaining keyword arguments are forwarded verbatim to
            the parent :class:`CheckpointHook`.
    """

    priority = "VERY_LOW"

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        exp_name: str = "Default",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        artifact_location: Optional[str] = None,
        artifact_subdir: str = "checkpoints",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # First, initialise the base CheckpointHook – this sets attributes like
        # ``self.last_ckpt`` which we will re-use later.
        super().__init__(*args, **kwargs)

        # MLflow initialisation is done *once*, ideally only on the main rank
        # to avoid creating multiple runs.
        if not is_main_process():
            # Non-master ranks do not touch MLflow; they will simply skip the
            # logging calls later.
            self.mlflow = None  # type: ignore
            return

        # ---------- MLflow environment setup ----------
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use a local *file* store inside the current working directory as
            # the default when nothing is specified.
            default_store = f"file://{osp.abspath('./mlruns')}"
            mlflow.set_tracking_uri(default_store)

        # Create (if needed) and set the experiment.
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(exp_name, artifact_location=artifact_location)
        mlflow.set_experiment(exp_name)

        # Make sure *some* active run exists so that artefacts can be logged.
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=run_name)

        # Apply optional metadata.
        if run_name is not None:
            mlflow.set_tag("mlflow.runName", run_name)
        if tags is not None:
            mlflow.set_tags(tags)
        if params is not None:
            mlflow.log_params(params)

        self.mlflow = mlflow
        self._artifact_subdir = artifact_subdir

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    def _log_artifact(self, file_path: Optional[str]) -> None:
        """Upload *file_path* as an artefact (master rank only)."""
        if self.mlflow is None:  # Non-master ranks
            return
        if file_path and osp.isfile(file_path):
            self.mlflow.log_artifact(file_path, artifact_path=self._artifact_subdir)

    # ---------------------------------------------------------------------
    # Overridden checkpointing hooks
    # ---------------------------------------------------------------------

    def _save_checkpoint(self, runner) -> None:  # type: ignore[override]
        """Save the *latest* checkpoint and upload it to MLflow."""
        # Delegate to the parent implementation first.
        super()._save_checkpoint(runner)

        if is_main_process():
            self._log_artifact(self.last_ckpt)

    def _save_best_checkpoint(self, runner, metrics):  # type: ignore[override]
        """Save *best* checkpoints and upload them to MLflow."""
        # Let the parent do its full logic (select, delete, save, etc.) first.
        super()._save_best_checkpoint(runner, metrics)

        if not is_main_process():
            return

        # Depending on configuration there may be one or multiple best paths.
        if hasattr(self, "best_ckpt_path") and self.best_ckpt_path is not None:
            self._log_artifact(self.best_ckpt_path)

        if hasattr(self, "best_ckpt_path_dict"):
            for path in self.best_ckpt_path_dict.values():
                self._log_artifact(path) 