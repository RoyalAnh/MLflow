import mlflow
import os

def init_mlflow(experiment_name: str = "MARL-Experiment"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_training(params: dict, metrics: dict, model_dir: str):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(model_dir)
