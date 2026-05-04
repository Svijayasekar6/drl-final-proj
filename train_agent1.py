"""
Agent 1 training — Vanilla PPO + self-play, no reward shaping.
"""

import argparse
import os

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils import create_rllib_env

# ── Mode flag ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["smoke", "full"], default="full",
                    help="smoke: 2M steps / 2h; full: 15M steps, no time cap")
parser.add_argument("--restore", default="",
                    help="Checkpoint path to resume from (optional)")
args = parser.parse_args()

if args.mode == "smoke":
    RUN_NAME        = "PPO_agent1_smoke"
    STOP_CONFIG     = {"timesteps_total": 2_000_000, "time_total_s": 7_200}
    CHECKPOINT_FREQ = 50
else:
    RUN_NAME        = "PPO_agent1_full"
    STOP_CONFIG     = {"timesteps_total": 15_000_000}
    CHECKPOINT_FREQ = 200

SCRATCH             = os.environ.get("SCRATCH_DIR", "./ray_results")
NUM_ENVS_PER_WORKER = 3


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "default"
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.50, 0.25, 0.125, 0.125],
    )


class SelfPlayCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        result  = info["result"]
        trainer = info["trainer"]
        if result["episode_reward_mean"] > 0.5:
            print("── Rotating opponent snapshots ──")
            trainer.set_weights({
                "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                "opponent_1": trainer.get_weights(["default"])["default"],
            })


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env  = create_rllib_env({"num_envs_per_worker": 1})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    restore = args.restore if args.restore else None

    analysis = tune.run(
        "PPO",
        name=RUN_NAME,
        restore=restore,
        config={
            # ── System ──────────────────────────────────────────────────────
            "num_gpus":            1,
            "num_workers":         8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level":           "INFO",
            "framework":           "torch",
            "callbacks":           SelfPlayCallback,
            # ── Environment ──────────────────────────────────────────────────
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "shaped_rewards":      False,
                "defensive_reward":    False,
            },
            # ── Multi-agent self-play ─────────────────────────────────────────
            "multiagent": {
                "policies": {
                    "default":    (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            # ── Model ────────────────────────────────────────────────────────
            "model": {
                "fcnet_hiddens":    [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers":  True,
            },
            # ── PPO hyperparameters ───────────────────────────────────────────
            "rollout_fragment_length": 5000,
            "batch_mode":              "complete_episodes",
            "train_batch_size":        40000,
            "sgd_minibatch_size":      4000,
            "num_sgd_iter":            10,
            "lr":                      5e-4,
            "clip_param":              0.2,
            "entropy_coeff":           0.01,
            "vf_loss_coeff":           0.5,
            "gamma":                   0.99,
            "lambda":                  0.95,
        },
        stop=STOP_CONFIG,
        checkpoint_freq=CHECKPOINT_FREQ,
        checkpoint_at_end=True,
        local_dir=SCRATCH,
    )

    best_trial      = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print("Best checkpoint:", best_checkpoint)