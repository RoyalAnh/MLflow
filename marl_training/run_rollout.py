import ray
import os
import numpy as np
import torch
import json
import mlflow

from agents.ppo_agent import get_ppo_trainer
from environment.gridworld_env import GridWorldEnv
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from torch.distributions import Categorical 

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "MARL-Experiment"

RUN_ID = os.environ.get("MLFLOW_RUN_ID")  # Truyền run_id qua biến môi trường hoặc tham số 

if __name__ == "__main__":
    ray.init()
    ckpt_path = os.path.abspath("mlartifacts/807185155446277740/92a0ff0310544dd0a99f02bd66021f1d/artifacts/checkpoints")
    trainer = get_ppo_trainer("GridWorld-v0", num_agents=2)
    trainer.restore(ckpt_path)
    env = GridWorldEnv(config={"num_agents": 2})
    obs, _ = env.reset()
    rollout_logs = []
    for step in range(20):
        actions = {}
        for agent_id in obs:
            module = trainer.get_module(agent_id)
            obs_tensor = {"obs": torch.tensor(np.expand_dims(obs[agent_id], axis=0), dtype=torch.float32)}
            action_out = module.forward_inference(obs_tensor)
            logits = action_out["action_dist_inputs"]
            action_dist = Categorical(logits=logits)
            sampled_action = action_dist.sample().item()
            actions[agent_id] = sampled_action
        obs, rewards, dones, _, _ = env.step(actions)
        print(f"Step {step} | Actions: {actions} | Rewards: {rewards}")
        rollout_logs.append({
            "step": step,
            "actions": actions,
            "rewards": rewards
        })
    rollout_path = "rollout_log.json"
    with open(rollout_path, "w") as f:
        json.dump(rollout_logs, f, indent=2)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    if RUN_ID:
        mlflow.start_run(run_id=RUN_ID)
    else:
        mlflow.start_run(run_name="rollout_eval")
    mlflow.log_artifact(rollout_path, artifact_path="rollout_logs")
    mlflow.end_run()
    ray.shutdown()
