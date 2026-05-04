"""
Agent 3 training — BC warm-start + PPO fine-tuning with reward annealing.
Uses BCWarmStartModel (registered in utils.py) to load pretrained weights
from bc_pretrain.py before PPO starts.
"""

import argparse
import os

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils import create_rllib_env, BCWarmStartModel

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["smoke", "full"], default="full")
parser.add_argument(
    "--bc-checkpoint",
    default="bc_checkpoint.pth",
    help="Path to BC pretrained weights (output of bc_pretrain.py)",
)
parser.add_argument("--restore", default="",
                    help="RLLib checkpoint to resume from (optional)")
args = parser.parse_args()

BC_PATH = os.path.abspath(args.bc_checkpoint)

if args.mode == "smoke":
    RUN_NAME        = "PPO_agent3_smoke"
    STOP_CONFIG     = {"timesteps_total": 1_000_000, "time_total_s": 3_600}
    CHECKPOINT_FREQ = 50
else:
    RUN_NAME        = "PPO_agent3_full"
    STOP_CONFIG     = {"timesteps_total": 7_000_000}
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
    temp_env  = create_rllib_env({
        "num_envs_per_worker": 1,
        "shaped_rewards":       True,
        "defensive_reward":     True,
        "anneal_shaped_rewards": True,
    })
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
                "num_envs_per_worker":  NUM_ENVS_PER_WORKER,
                "shaped_rewards":       True,
                "defensive_reward":     True,
                "anneal_shaped_rewards": True,
                "anneal_steps":         4_000_000,
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
            # ── BCWarmStartModel injects BC pretrained weights ────────────────
            "model": {
                "custom_model": "BCWarmStartModel",
                "custom_model_config": {
                    "bc_checkpoint": BC_PATH,
                },
                "fcnet_hiddens":    [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers":  False,  # value head separate — BC only informs actor
            },
            # ── PPO hyperparameters (tuned for warm-start) ────────────────────
            "rollout_fragment_length": 5000,
            "batch_mode":              "complete_episodes",
            "train_batch_size":        40000,
            "sgd_minibatch_size":      4000,
            "num_sgd_iter":            10,
            "lr":                      3e-4,    # lower than A1/A2 — preserve BC init
            "clip_param":              0.2,
            "entropy_coeff":           0.005,   # lower — BC policy is less random
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