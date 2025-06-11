import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ray import serve
from marl_training.agents.ppo_agent import get_ppo_trainer
import numpy as np

@serve.deployment(num_replicas=1)
class InferenceModel:
    def __init__(self, checkpoint_path: str):
        self.trainer = get_ppo_trainer("GridWorld-v0", num_agents=2)
        self.trainer.restore(checkpoint_path)

    async def __call__(self, request):
        print("Headers:", request.headers)
        print("Body:", await request.body())
        data = await request.json()
        obs = np.array(data.get("obs", [2, 3]))
        agent_id = data.get("agent_id", "agent_0")
        action = self.trainer.compute_single_action(obs, policy_id=agent_id)
        return {"action": int(action)}
