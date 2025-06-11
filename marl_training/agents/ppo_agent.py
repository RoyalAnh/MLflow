from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.multi_agent_env import make_multi_agent

def get_ppo_trainer(env_name, num_agents=2, config_overrides=None):
    def env_creator(config):
        from marl_training.environment.gridworld_env import GridWorldEnv
        return GridWorldEnv(config)

    import ray
    from ray import tune
    from ray.tune.registry import register_env

    register_env(env_name, env_creator)

    dummy_env = env_creator({"num_agents": num_agents})
    policies = {
        f"agent_{i}": (None, dummy_env.observation_space, dummy_env.action_space, {}) 
        for i in range(num_agents)
    }


    def policy_mapping_fn(agent_id, episode=None):
        return agent_id

    config = {
        "env": env_name,
        "env_config": {
            "num_agents": num_agents
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": "torch",
        "num_workers": 0,
    }


    if config_overrides:
        config.update(config_overrides)

    return PPO(config=config)
