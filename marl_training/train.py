import ray
import mlflow
import os
import shutil
import json
from agents.ppo_agent import get_ppo_trainer

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "MARL-Experiment"

def get_training_data_from_logfile(log_file_path, limit=1000):
    """Đọc dữ liệu huấn luyện từ file log do kafka_consumer.py tạo ra."""
    data = []
    if not os.path.exists(log_file_path):
        print(f"[WARN] Không tìm thấy file log {log_file_path}")
        return data
    with open(log_file_path, "r") as fin:
        for i, line in enumerate(fin):
            if i >= limit: break
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"[ERROR] Lỗi đọc dòng {i}: {e}")
    return data

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        trainer = get_ppo_trainer("GridWorld-v0", num_agents=2)
        mlflow.log_params({
            "num_agents": 2,
            "algo": "PPO",
            "env": "GridWorld-v0"
        })
        # Đọc dữ liệu từ file log của kafka_consumer.py
        log_file = os.path.join(os.path.dirname(__file__), "../kafka_to_cassandra_log.jsonl")
        train_data = get_training_data_from_logfile(log_file, limit=1000)
        train_data_path = "train_data.json"
        with open(train_data_path, "w") as f:
            json.dump(train_data, f, indent=2)
        mlflow.log_artifact(train_data_path, artifact_path="input_data")
        # (Nếu muốn train offline RL thực sự, cần sửa trainer, còn ở đây chỉ log lại flow.)
        episode_logs = []
        for i in range(20):
            result = trainer.train()
            reward = result.get("env_runners", {}).get("episode_return_mean", None)
            log_entry = {
                "iter": i,
                "reward": reward,
                "actions": result.get("actions", None),
                "observations": result.get("observations", None)
            }
            episode_logs.append(log_entry)
            print(f"Iter {i}: reward = {reward}")
        log_path = "episode_log.json"
        with open(log_path, "w") as f:
            json.dump(episode_logs, f, indent=2)
        mlflow.log_artifact(log_path, artifact_path="training_logs")
        checkpoint_result = trainer.save()
        checkpoint_dir = checkpoint_result.checkpoint.path
        mlflow.log_param("checkpoint_path", checkpoint_dir)
        artifact_path = "checkpoints"
        zip_path = shutil.make_archive("ppo_checkpoint", 'zip', checkpoint_dir)
        mlflow.log_artifact(zip_path, artifact_path=artifact_path)
        print("Run ID:", run.info.run_id)

    ray.shutdown()

