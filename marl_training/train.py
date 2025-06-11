import ray
import mlflow
import os
import shutil
from agents.ppo_agent import get_ppo_trainer

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "MARL-Experiment"

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        trainer = get_ppo_trainer("GridWorld-v0", num_agents=2)
        # Log hyperparameters
        mlflow.log_params({
            "num_agents": 2,
            "algo": "PPO",
            "env": "GridWorld-v0"
        })
        for i in range(20): # Số vòng lặp huấn luyện
            result = trainer.train()
            reward = result.get("env_runners", {}).get("episode_return_mean", None)
            if reward is not None:
                print(f"Iter {i}: reward = {reward}") # , info = {result}
            else:
                print(f"Iter {i}: reward info not available. Keys: {list(result.keys())}")


        checkpoint_result = trainer.save()
        checkpoint_dir = checkpoint_result.checkpoint.path
        mlflow.log_param("checkpoint_path", checkpoint_dir)

        # Nén checkpoint để log artifact
        artifact_path = "checkpoints"
        zip_path = shutil.make_archive("ppo_checkpoint", 'zip', checkpoint_dir)
        mlflow.log_artifact(zip_path, artifact_path=artifact_path)

    ray.shutdown()