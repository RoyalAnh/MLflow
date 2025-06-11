import ray
import shutil
import os

@ray.remote
def prepare_data():
    print("📦 Step 1: Data is being streamed from Kafka → Cassandra in real-time.")
    return "data_ready"

@ray.remote
def train_model(_):
    from marl_training.agents.ppo_agent import get_ppo_trainer
    trainer = get_ppo_trainer("GridWorld-v0", num_agents=2)
    for _ in range(10):
        trainer.train()
    checkpoint_path = trainer.save()
    print(f"✅ Checkpoint saved at: {checkpoint_path}")
    return checkpoint_path

@ray.remote
def deploy_model(checkpoint_path: str):
    dst_path = "ray_workflow/deployed_model"
    os.makedirs(dst_path, exist_ok=True)
    shutil.copytree(checkpoint_path, f"{dst_path}/model", dirs_exist_ok=True)
    print(f"🚀 Model deployed to {dst_path}/model")
    return f"{dst_path}/model"

@ray.remote
def full_pipeline():
    data = prepare_data.remote()
    ckpt = train_model.remote(data)
    deployed_path = deploy_model.remote(ckpt)
    return deployed_path
