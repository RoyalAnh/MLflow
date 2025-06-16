import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import ray
from ray import serve
import torch
import numpy as np
from marl_training.agents.ppo_agent import get_ppo_trainer

@serve.deployment(num_replicas=1)
class InferenceModel:
    def __init__(self, checkpoint_path: str):
        self.trainer = get_ppo_trainer("GridWorld-v0", num_agents=2)
        self.trainer.restore(checkpoint_path)

    async def __call__(self, request):
        try:
            if request.method == "GET":
                return {
                    "message": "POST JSON: {\"agent_0\": [x, y], \"agent_1\": [x, y]}"
                }
            data = await request.json()
            actions = {}
            details = {}
            for agent_id, obs in data.items():
                module = self.trainer.get_module(agent_id)
                obs_tensor = {"obs": torch.tensor(np.expand_dims(np.array(obs), axis=0), dtype=torch.float32)}
                action_out = module.forward_inference(obs_tensor)
                action = int(action_out["action_dist_inputs"].argmax())
                actions[agent_id] = action
                details[agent_id] = {
                    "obs": obs,
                    "action": action,
                    "logits": action_out["action_dist_inputs"].tolist()
                }
            return {
                "inference_result": details,
                "actions": actions
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
