from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.grid_size = 5
        self._num_agents = 2 if config is None else config.get("num_agents", 2)
        self.agents = ["agent_0", "agent_1"]  # danh sách các agent hiện đang hoạt động trong env
        self.possible_agents = ["agent_0", "agent_1"]  # danh sách các agent có thể có trong toàn bộ episode
        self.action_space = spaces.Discrete(4)
        self.action_spaces = {f"agent_{i}": self.action_space for i in range(self._num_agents)}
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.float32)
        self.observation_spaces = {f"agent_{i}": self.observation_space for i in range(self._num_agents)}
        self.agent_positions = self._generate_agent_positions()
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
        self.steps = 0
        self.max_steps = 100

    def _generate_agent_positions(self):
        return {f"agent_{i}": np.array([0, 0]) for i in range(self._num_agents)}

    def reset(self, *, seed=None, options=None):
        self.agent_positions = self._generate_agent_positions()
        self.steps = 0
        obs = {agent_id: pos.astype(np.float32) for agent_id, pos in self.agent_positions.items()}
        info = {}
        return obs, info

    def step(self, action_dict):
        self.steps += 1
        obs, rewards, infos = {}, {}, {}
        terminateds, truncateds = {}, {}

        for agent_id, action in action_dict.items():
            # ...move agent...
            if action == 0:  # up
                self.agent_positions[agent_id][1] = max(0, self.agent_positions[agent_id][1] - 1)
            elif action == 1:  # down
                self.agent_positions[agent_id][1] = min(self.grid_size - 1, self.agent_positions[agent_id][1] + 1)
            elif action == 2:  # left
                self.agent_positions[agent_id][0] = max(0, self.agent_positions[agent_id][0] - 1)
            elif action == 3:  # right
                self.agent_positions[agent_id][0] = min(self.grid_size - 1, self.agent_positions[agent_id][0] + 1)

            obs[agent_id] = self.agent_positions[agent_id].astype(np.float32)
            if np.array_equal(self.agent_positions[agent_id], self.goal_pos):
                rewards[agent_id] = 100  # hoặc 10 
                terminateds[agent_id] = True
            else:
                rewards[agent_id] = -1
                terminateds[agent_id] = self.steps >= self.max_steps

            infos[agent_id] = {}

            # terminateds[agent_id] = self.steps >= self.max_steps
            truncateds[agent_id] = False
        # print(self.agent_positions)  log vị trí agent mỗi vòng lặp để kiểm tra:

        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = False

        return obs, rewards, terminateds, truncateds, infos