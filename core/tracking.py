"""
Tracking – Developer Guide

Utilities for ML experiment tracking using MLflow.

--------------------------------------------------------------------------------
Callable Functions:
--------------------------------------------------------------------------------
init_mlflow()
    - Sets the MLflow tracking URI to local `./mlruns`.

start_run(run_name: str)
    - Starts an MLflow run with a given name.

log_params(params: dict)
    - Logs parameters to the active MLflow run.

log_metrics(metrics: dict)
    - Logs metrics to the active MLflow run.

end_run()
    - Ends the current MLflow run.
"""

import mlflow

def init_mlflow():
    mlflow.set_tracking_uri("file:./mlruns")


def start_run(run_name: str):
    mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    mlflow.log_params(params)


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def end_run():
    mlflow.end_run()