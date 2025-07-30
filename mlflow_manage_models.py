    #!/usr/bin/env python3
"""
MLflow Registration Utility
===========================

This standalone CLI tool lets you keep the MLflow Model Registry tidy and up to date in **one command**.

Supported actions
-----------------
1. **Delete** registered models – remove *all* models or only a comma-separated subset.
2. **Register** the **latest logged model** from **every active run** inside a given experiment (optionally filtered to a subset of model names).
3. **Tag** newly registered models / versions with an optional `task` label for easy filtering.
4. Both operations can be combined; deletion is performed first, followed by registration.

Typical usage
-------------

Delete everything:
    python mlflow_manage_models.py --delete all

Delete a subset:
    python mlflow_manage_models.py --delete "DBNet,PANet"

# Register the freshest checkpoints from an experiment (all models):
    python mlflow_manage_models.py --experiment mmocr_det --register all

# Register only DBNet & PANet from the experiment:
    python mlflow_manage_models.py --experiment mmocr_det --register "DBNet,PANet"

Delete old models **and** re-register the latest ones in a single shot:
    python mlflow_manage_models.py \
        --delete "DBNet,PANet" \
        --from-experiment mmocr_det \
        --task textrecog
"""

import argparse
import os
import sys
import json
from typing import List, Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from dotenv import load_dotenv


def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    
    # Set MLflow tracking URI if provided in env
    if os.getenv('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))


# -----------------------------------------------------------------------------
# Deletion helpers
# -----------------------------------------------------------------------------


def _models_for_experiment(client: MlflowClient, experiment_ids: list[str]) -> set[str]:
    """Return a set of registered model *names* that have at least one version whose
    source_run_id belongs to the supplied experiments."""
    df = mlflow.search_logged_models(experiment_ids=experiment_ids)
    if df.empty:
        return set()
    return set(df["name"].unique())


def delete_registered_models(
    client: MlflowClient,
    models_to_delete: str,
    experiment_ids: list[str] | None = None,
):
    """
    Delete registered models
    
    Args:
        client: MLflow client instance
        models_to_delete: Either 'all' or comma-separated list of model names
    """
    try:
        # Determine the candidate model list depending on experiment filter
        if experiment_ids:
            experiment_models = _models_for_experiment(client, experiment_ids)
            print(
                f"Experiment filter active – {len(experiment_models)} model(s) linked"
                " to the given experiment(s) will be considered for deletion."
            )
        else:
            experiment_models = None  # means no filtering

        if models_to_delete.lower() == 'all':
            # Delete all (within scope)
            registered_models = client.search_registered_models()
            model_names = [model.name for model in registered_models]
            if experiment_models is not None:
                model_names = [m for m in model_names if m in experiment_models]
            print(f"Found {len(model_names)} registered models to delete")
        else:
            # Parse comma-separated list of model names
            requested = [name.strip() for name in models_to_delete.split(',') if name.strip()]
            if experiment_models is not None:
                model_names = [m for m in requested if m in experiment_models]
            else:
                model_names = requested
            print(f"Models to delete: {model_names}")
        
        if not model_names:
            print("No models to delete")
            return
        
        # Delete each model
        for model_name in model_names:
            try:
                print(f"Deleting registered model: {model_name}")
                client.delete_registered_model(model_name)
                print(f"Successfully deleted model: {model_name}")
            except MlflowException as e:
                print(f"Error deleting model {model_name}: {e}")
                
    except Exception as e:
        print(f"Error in delete operation: {e}")
        sys.exit(1)


def register_models_from_experiment(
    client: MlflowClient,
    experiment_name: str,
    *,
    task_tag: str | None = None,
    filter_model_names: list[str] | None = None,
):
    """
    Register last model checkpoints from each run in the specified experiment
    using mlflow.search_logged_models()
    
    Args:
        client: MLflow client instance
        experiment_name: Name of the experiment to process
    """
    try:
        # Get experiment by name
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            sys.exit(1)
        
        print(f"Processing experiment: {experiment_name} (ID: {experiment.experiment_id})")
        
        # Search for all logged models in this experiment
        print("Searching for logged models in experiment...")
        logged_models_df = mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id]
        )
        
        if logged_models_df.empty:
            print(f"No logged models found in experiment '{experiment_name}'")
            return
        
        print(f"Found {len(logged_models_df)} total logged models in experiment")
        
        # Group by source_run_id and get the latest model per run
        # Sort by creation_timestamp descending and take first per run
        latest_models_per_run = (
            logged_models_df
            .sort_values('creation_timestamp', ascending=False)
            .groupby('source_run_id')
            .first()
            .reset_index()
        )
        
        print(f"Found {len(latest_models_per_run)} unique runs with models")
        
        registered_count = 0
        for _, model_row in latest_models_per_run.iterrows():
            run_id = model_row['source_run_id']
            model_name = model_row['name']
            model_uri = model_row['artifact_location']
            model_id = model_row['model_id']
            
            # Get run info for run name display
            try:
                run = client.get_run(run_id)
                # Skip runs that are not active
                if getattr(run.info, 'lifecycle_stage', 'active').lower() != 'active':
                    print(f"  Skipping run {run_id[:8]} - lifecycle_stage is '{run.info.lifecycle_stage}'")
                    continue
                run_name = run.data.tags.get('mlflow.runName', f'run_{run_id[:8]}')
            except Exception as e:
                print(f"  Warning: Could not get run info for {run_id}: {e}")
                run_name = f'run_{run_id[:8]}'
            
            # Optional model-name whitelist filtering
            if filter_model_names is not None and model_name not in filter_model_names:
                continue

            # Use model name as the registered model name
            registered_model_name = model_name.replace(' ', '_').replace('-', '_')
            
            print(f"\nProcessing run: {run_name} ({run_id})")
            print(f"  Model: {model_name} (ID: {model_id})")
            print(f"  Model URI: {model_uri}")
            
            try:
                print(f"  Registering as: {registered_model_name}")
                
                # Create registered model if it doesn't exist
                try:
                    client.create_registered_model(registered_model_name)
                    print(f"  Created new registered model: {registered_model_name}")
                except MlflowException as ce:
                    if "already exists" not in str(ce).lower():
                        raise
                    print(f"  Registered model already exists: {registered_model_name}")
                
                # Create new model version
                model_version = client.create_model_version(
                    name=registered_model_name,
                    source=model_uri,
                    run_id=run_id
                )

                # Apply task tag if provided
                if task_tag:
                    try:
                        client.set_registered_model_tag(registered_model_name, 'task_name', task_tag)
                        client.set_model_version_tag(registered_model_name, model_version.version, 'task_name', task_tag)
                        print(f"  Tagged with task='{task_tag}'")
                    except Exception as tag_e:
                        print(f"  Warning: failed to set task tag: {tag_e}")
                
                print(f"  Successfully registered model version: {registered_model_name} (version {model_version.version})")
                registered_count += 1
                
            except MlflowException as e:
                print(f"  Error registering model {registered_model_name}: {e}")
                continue
            except Exception as e:
                print(f"  Unexpected error registering model {registered_model_name}: {e}")
                continue
        
        print(f"\nRegistration complete. Total models registered: {registered_count}")
        
    except Exception as e:
        print(f"Error in registration operation: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and execute operations"""
    parser = argparse.ArgumentParser(
        description="MLflow Model Registration and Deletion Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Delete all registered models:
    python mlflow_manage_models.py --delete all
    
  Delete specific models:
    python mlflow_manage_models.py --delete "ABINet,ASTER,TrOCR"
    
  Register models from experiment:
    python mlflow_manage_models.py --experiment "text_recognition_experiment"
        """
    )
    
    parser.add_argument(
        '--delete',
        type=str,
        default=None,
        help="Delete registered models. Use 'all' to delete all models or provide comma-separated list of model names"
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help="Target experiment name for deletion/registration operations"
    )

    parser.add_argument(
        '--register',
        type=str,
        default=None,
        help="Register models from the given experiment. Use 'all' or comma-separated list of model names"
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help="Optional task tag to attach to registered models and versions (e.g., 'textrecog')"
    )
    
    args = parser.parse_args()
    
    # Validate that at least one major operation is specified
    if args.delete is None and args.register is None:
        print("Error: At least one of --delete or --register must be specified")
        parser.print_help()
        sys.exit(1)

    if args.register is not None and args.experiment is None:
        print("Error: --register requires --experiment to be specified")
        sys.exit(1)
    
    # Load environment variables
    print("Loading environment variables from .env file...")
    load_environment()
    
    # Initialize MLflow client
    try:
        client = MlflowClient()
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    except Exception as e:
        print(f"Error initializing MLflow client: {e}")
        sys.exit(1)
    
    # Execute operations in order: delete first (if any), then register (if any)
    if args.delete is not None:
        print(f"Starting delete operation for: {args.delete}")
        exp_ids_for_delete: list[str] | None = None
        if args.experiment is not None:
            # If deletion is combined with registration, limit deletion to that experiment
            exp_obj = client.get_experiment_by_name(args.experiment)
            if exp_obj:
                exp_ids_for_delete = [exp_obj.experiment_id]
        delete_registered_models(client, args.delete, experiment_ids=exp_ids_for_delete)
 
    if args.register is not None:
        print(f"Starting registration operation for experiment: {args.experiment}")

        model_filter: list[str] | None
        if args.register.lower() == 'all':
            model_filter = None
        else:
            model_filter = [m.strip() for m in args.register.split(',') if m.strip()]

        register_models_from_experiment(
            client,
            args.experiment,
            task_tag=args.task,
            filter_model_names=model_filter,
        )
    
    print("Operation completed successfully!")


if __name__ == "__main__":
    main() 