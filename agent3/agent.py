"""
Agent 3 — Human BC pretraining + PPO fine-tuning with reward annealing.

BCWarmStartModel must be registered before the checkpoint is restored.
This is done at the top of this file via the local model.py import.

"""

import os
import pickle
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface
from .model import BCWarmStartModel

# Register custom model before restoring checkpoint
ModelCatalog.register_custom_model("BCWarmStartModel", BCWarmStartModel)

ALGORITHM   = "PPO"
POLICY_NAME = "default"

# TODO: fill in after training — replace <trial_id> and <best_N>
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ray_results",
    "PPO_agent3_full",
    "PPO_Soccer_<trial_id>",
    "checkpoint_<best_N>",
    "checkpoint-<N>",
)


class RayAgent(AgentInterface):
    """
    Agent 3 — BC warm-start + PPO fine-tuning.
    Policy is warm-started from human demonstrations (signed importance-weighted BC),
    then fine-tuned with PPO + dense reward shaping + linear reward annealing.
    Novel learning concept: combines imitation learning with self-play RL.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True)

        config_dir  = os.path.dirname(CHECKPOINT_PATH)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"params.pkl not found near {CHECKPOINT_PATH}. "
                "Ensure the trial directory was copied alongside the checkpoint."
            )

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        config["num_workers"] = 0
        config["num_gpus"]    = 0

        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        cls   = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        agent.restore(CHECKPOINT_PATH)
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions