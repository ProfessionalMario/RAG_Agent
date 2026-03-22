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