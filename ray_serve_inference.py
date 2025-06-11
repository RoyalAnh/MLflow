'''import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))'''

import ray
from ray import serve
from marl_training.agents.ppo_agent import get_ppo_trainer
import numpy as np
import os

@serve.deployment(num_replicas=1)
class InferenceModel:
    def __init__(self, checkpoint_path: str):
        checkpoint_path = "mlartifacts/807185155446277740/5e08cd366a9841a2bae28354d5115082/artifacts/checkpoints" # gán cứng 
        self.trainer = get_ppo_trainer("GridWorld-v0", num_agents=2)
        self.trainer.restore(os.path.abspath(checkpoint_path))  # 🛠️ Load đúng checkpoint

    async def __call__(self, request):
        data = await request.json()
        obs = np.array(data.get("obs", [2, 3]))
        agent_id = data.get("agent_id", "agent_0")
        action = self.trainer.compute_single_action(obs, policy_id=agent_id)
        return {"action": int(action)}

