import ray
import os
import numpy as np
import torch

from agents.ppo_agent import get_ppo_trainer
from environment.gridworld_env import GridWorldEnv

# Tạo action distribution từ logits
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from torch.distributions import Categorical 

if __name__ == "__main__":
    ray.init()
    ckpt_path = os.path.abspath("mlartifacts/807185155446277740/5e08cd366a9841a2bae28354d5115082/artifacts/checkpoints")

    trainer = get_ppo_trainer("GridWorld-v0", num_agents=2)
    trainer.restore(ckpt_path)

    env = GridWorldEnv(config={"num_agents": 2})
    obs, _ = env.reset()

    for step in range(20):
        actions = {}
        for agent_id in obs:
            module = trainer.get_module(agent_id)
            obs_tensor = {"obs": torch.tensor(np.expand_dims(obs[agent_id], axis=0), dtype=torch.float32)}
            action_out = module.forward_inference(obs_tensor)
            print(f"[DEBUG] Output from forward_inference: {action_out}")

            logits = action_out["action_dist_inputs"]
            action_dist = Categorical(logits=logits)
            sampled_action = action_dist.sample().item()  # Lấy hành động từ phân phối
            actions[agent_id] = sampled_action

        obs, rewards, dones, _, _ = env.step(actions)
        print(f"Step {step} | Actions: {actions} | Rewards: {rewards}")

    ray.shutdown()
