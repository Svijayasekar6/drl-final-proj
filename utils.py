from random import uniform as randfloat

import gym
import numpy as np
import os
import torch
import torch.nn as nn
from ray.rllib import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """A RLLib wrapper so our env can inherit from MultiAgentEnv."""
    pass

SCORING_AXIS   = 0
BLUE_GOAL      = np.array([-14.0, 0.0])   # agents 0,1 defend this end
ORANGE_GOAL    = np.array([ 14.0, 0.0])   # agents 2,3 defend this end
FIELD_DIAGONAL = np.sqrt(28.0**2 + 10.0**2)   


def _own_goal(agent_id):
    return BLUE_GOAL if agent_id in (0, 1) else ORANGE_GOAL


def _opp_goal(agent_id):
    return ORANGE_GOAL if agent_id in (0, 1) else BLUE_GOAL


def _opp_goal_dir(agent_id):
    """Sign of attack direction along the scoring (x) axis."""
    return 1.0 if agent_id in (0, 1) else -1.0


def _teammate_id(agent_id):
    return {0: 1, 1: 0, 2: 3, 3: 2}[agent_id]


def _opp_ids(agent_id):
    return (2, 3) if agent_id in (0, 1) else (0, 1)


# ── Reward shaping coefficients ───────────────────────────────────────────────
# Terms that fire every step (A, F, G) must stay very small.
# Calibrate by running validate_reward.py
ALPHA  = 0.003   # Term A: delta ball approach
BETA   = 0.02    # Term B: shot quality (gated)
GAMMA  = 0.01    # Term C: defensive clearance (gated)
ETA    = 0.02    # Term D2: opponent shot penalty
ZETA   = 0.01    # Term F: team possession margin
XI     = 0.005   # Term G: ball progression (smallest — fires every step)
KAPPA  = 0.01    # Term H: anti-clustering penalty
MU     = 0.015   # Term K: goalie/defender positioning

PROXIMITY_THRESHOLD    = 2.0   # world units — gate for Term B
DANGER_THRESHOLD       = 3.0   # world units — gate for Term C
OPP_PRESSURE_THRESHOLD = 2.5   # world units — gate for D2 / F / K
CLUSTER_THRESHOLD      = 2.0   # world units — gate for Term H
LANE_WIDTH             = 1.5   # world units — corridor for Term K


def compute_shaped_reward_offline(
    agent_id,
    player_pos,       # np.array [x, y]
    ball_pos,         # np.array [x, y]
    ball_vel,         # np.array [vx, vy]
    sparse_reward,    # float: ±1 or 0
    all_player_pos=None,   # dict {agent_id: np.array [x, y]} — needed for D2/F/K/H
    prev_ball_dist=None,
    prev_ball_own_goal_dist=None,
    shaped_rewards=True,
    defensive_reward=True,
):
    """
    Standalone shaped reward for offline use (bc_pretrain, validate_reward).
    Returns (total_reward, components_dict) so callers can log each term.
    """
    p  = np.asarray(player_pos, dtype=np.float64)
    b  = np.asarray(ball_pos,   dtype=np.float64)
    bv = np.asarray(ball_vel,   dtype=np.float64)

    ball_speed = np.linalg.norm(bv)
    bv_unit    = bv / (ball_speed + 1e-8)
    curr_dist  = np.linalg.norm(b - p)

    opp_dists = []
    if all_player_pos:
        for oid in _opp_ids(agent_id):
            if oid in all_player_pos:
                opp_dists.append(np.linalg.norm(b - np.asarray(all_player_pos[oid])))
    opp_min_dist = min(opp_dists) if opp_dists else FIELD_DIAGONAL
    opp_pressure = max(0.0, 1.0 - opp_min_dist / OPP_PRESSURE_THRESHOLD)

    r_approach = r_shot = r_clear = r_opp_shot = 0.0
    r_possession = r_progress = r_cluster = r_goalie = 0.0

    if shaped_rewards:
        # Term A
        _prev_d  = prev_ball_dist if prev_ball_dist is not None else curr_dist
        r_approach = ALPHA * (_prev_d - curr_dist)

        # Term B
        to_opp      = _opp_goal(agent_id) - b
        to_opp_unit = to_opp / (np.linalg.norm(to_opp) + 1e-8)
        toward_opp  = float(np.dot(bv_unit, to_opp_unit))
        if curr_dist < PROXIMITY_THRESHOLD:
            r_shot = BETA * ball_speed * max(0.0, toward_opp)

        # Term C
        curr_own_dist = np.linalg.norm(b - _own_goal(agent_id))
        _prev_od = prev_ball_own_goal_dist if prev_ball_own_goal_dist is not None else curr_own_dist
        if curr_own_dist < DANGER_THRESHOLD:
            r_clear = GAMMA * (curr_own_dist - _prev_od)

        # Term D2
        to_own      = _own_goal(agent_id) - b
        to_own_unit = to_own / (np.linalg.norm(to_own) + 1e-8)
        ball_danger = ball_speed * max(0.0, float(np.dot(bv_unit, to_own_unit)))
        r_opp_shot  = -ETA * opp_pressure * ball_danger

        # Term F
        possession_margin = opp_min_dist - curr_dist
        r_possession = ZETA * max(0.0, possession_margin) / FIELD_DIAGONAL

        # Term G
        ball_progress_vel = bv[SCORING_AXIS] * _opp_goal_dir(agent_id)
        r_progress = XI * max(0.0, float(ball_progress_vel))

        # Term H
        if all_player_pos:
            tid = _teammate_id(agent_id)
            if tid in all_player_pos:
                teammate_dist = np.linalg.norm(p - np.asarray(all_player_pos[tid]))
                if teammate_dist < CLUSTER_THRESHOLD:
                    r_cluster = -KAPPA * (1.0 - teammate_dist / CLUSTER_THRESHOLD)

    if defensive_reward and opp_pressure > 0.0:
        ball_to_goal      = _own_goal(agent_id) - b
        ball_to_goal_dist = np.linalg.norm(ball_to_goal)
        ball_to_goal_unit = ball_to_goal / (ball_to_goal_dist + 1e-8)
        agent_from_ball   = p - b
        proj              = float(np.dot(agent_from_ball, ball_to_goal_unit))
        proj_fraction     = float(np.clip(proj / (ball_to_goal_dist + 1e-8), 0.0, 1.0))
        lateral           = np.linalg.norm(agent_from_ball - proj * ball_to_goal_unit)
        in_lane           = proj_fraction * max(0.0, 1.0 - lateral / LANE_WIDTH)
        r_goalie          = MU * opp_pressure * in_lane

    total = (r_approach + r_shot + r_clear + r_opp_shot
             + r_possession + r_progress + r_cluster + r_goalie
             + sparse_reward)

    components = dict(
        A=r_approach, B=r_shot, C=r_clear, D2=r_opp_shot,
        F=r_possession, G=r_progress, H=r_cluster, K=r_goalie,
        goal=sparse_reward,
    )
    return total, components


class RewardShapingWrapper(RLLibWrapper):
    """
    Augments per-step rewards using position/velocity aux info from the env.

    env_config keys:
        shaped_rewards   (bool): Enables Terms A-H (attack + coordination).
        defensive_reward (bool): Enables Term K (goalie/defender positioning).
    Both default to False — set True for Agents 2 and 3.

    """

    def __init__(self, env, env_config=None):
        super().__init__(env)
        cfg = env_config or {}
        self.shaped_rewards   = cfg.get("shaped_rewards",   False)
        self.defensive_reward = cfg.get("defensive_reward", False)
        self._needs_aux       = self.shaped_rewards or self.defensive_reward
        self._warned_no_aux   = False

        self._prev_ball_dist:          dict = {}
        self._prev_ball_own_goal_dist: dict = {}

    def reset(self):
        obs = super().reset()
        self._prev_ball_dist.clear()
        self._prev_ball_own_goal_dist.clear()
        return obs

    def step(self, action):
        obs, rewards, dones, infos = super().step(action)
        if self._needs_aux:
            rewards = self._apply_shaping(rewards, infos)
        return obs, rewards, dones, infos

    def _apply_shaping(self, rewards: dict, infos: dict) -> dict:
        shaped = {}

        for agent_id, sparse_reward in rewards.items():
            info = infos.get(agent_id, {})

            if not info:
                if not self._warned_no_aux:
                    print(
                        "[RewardShapingWrapper] WARNING: aux_info empty. "
                        "Training binary may send 336-dim obs (not 345-dim). "
                        "Reward shaping is disabled. Verify binary version."
                    )
                    self._warned_no_aux = True
                shaped[agent_id] = sparse_reward
                continue

            p  = np.asarray(info["player_info"]["position"],  dtype=np.float64)
            b  = np.asarray(info["ball_info"]["position"],    dtype=np.float64)
            bv = np.asarray(info["ball_info"]["velocity"],    dtype=np.float64)

            ball_speed = np.linalg.norm(bv)
            bv_unit    = bv / (ball_speed + 1e-8)
            curr_dist  = np.linalg.norm(b - p)

            opp_dists = []
            for oid in _opp_ids(agent_id):
                oinfo = infos.get(oid, {})
                if oinfo:
                    op = np.asarray(oinfo["player_info"]["position"], dtype=np.float64)
                    opp_dists.append(np.linalg.norm(b - op))
            opp_min_dist = min(opp_dists) if opp_dists else FIELD_DIAGONAL
            opp_pressure = max(0.0, 1.0 - opp_min_dist / OPP_PRESSURE_THRESHOLD)

            r_approach = r_shot = r_clear = r_opp_shot = 0.0
            r_possession = r_progress = r_cluster = r_goalie = 0.0

            if self.shaped_rewards:
                prev_dist  = self._prev_ball_dist.get(agent_id, curr_dist)
                r_approach = ALPHA * (prev_dist - curr_dist)

                to_opp      = _opp_goal(agent_id) - b
                to_opp_unit = to_opp / (np.linalg.norm(to_opp) + 1e-8)
                toward_opp  = float(np.dot(bv_unit, to_opp_unit))
                if curr_dist < PROXIMITY_THRESHOLD:
                    r_shot = BETA * ball_speed * max(0.0, toward_opp)

                curr_own_dist = np.linalg.norm(b - _own_goal(agent_id))
                prev_own_dist = self._prev_ball_own_goal_dist.get(agent_id, curr_own_dist)
                if curr_own_dist < DANGER_THRESHOLD:
                    r_clear = GAMMA * (curr_own_dist - prev_own_dist)

                to_own      = _own_goal(agent_id) - b
                to_own_unit = to_own / (np.linalg.norm(to_own) + 1e-8)
                ball_danger = ball_speed * max(0.0, float(np.dot(bv_unit, to_own_unit)))
                r_opp_shot  = -ETA * opp_pressure * ball_danger

                possession_margin = opp_min_dist - curr_dist
                r_possession = ZETA * max(0.0, possession_margin) / FIELD_DIAGONAL

                ball_progress_vel = bv[SCORING_AXIS] * _opp_goal_dir(agent_id)
                r_progress = XI * max(0.0, float(ball_progress_vel))

                t_info = infos.get(_teammate_id(agent_id), {})
                if t_info:
                    t_pos = np.asarray(t_info["player_info"]["position"], dtype=np.float64)
                    teammate_dist = np.linalg.norm(p - t_pos)
                    if teammate_dist < CLUSTER_THRESHOLD:
                        r_cluster = -KAPPA * (1.0 - teammate_dist / CLUSTER_THRESHOLD)

            if self.defensive_reward and opp_pressure > 0.0:
                ball_to_goal      = _own_goal(agent_id) - b
                ball_to_goal_dist = np.linalg.norm(ball_to_goal)
                ball_to_goal_unit = ball_to_goal / (ball_to_goal_dist + 1e-8)
                agent_from_ball   = p - b
                proj              = float(np.dot(agent_from_ball, ball_to_goal_unit))
                proj_fraction     = float(np.clip(proj / (ball_to_goal_dist + 1e-8), 0.0, 1.0))
                lateral           = np.linalg.norm(agent_from_ball - proj * ball_to_goal_unit)
                in_lane           = proj_fraction * max(0.0, 1.0 - lateral / LANE_WIDTH)
                r_goalie          = MU * opp_pressure * in_lane

            shaped[agent_id] = (
                r_approach + r_shot + r_clear + r_opp_shot
                + r_possession + r_progress + r_cluster + r_goalie
                + sparse_reward
            )

        for agent_id in rewards:
            info = infos.get(agent_id, {})
            if info:
                b = np.asarray(info["ball_info"]["position"], dtype=np.float64)
                p = np.asarray(info["player_info"]["position"], dtype=np.float64)
                self._prev_ball_dist[agent_id]          = float(np.linalg.norm(b - p))
                self._prev_ball_own_goal_dist[agent_id] = float(
                    np.linalg.norm(b - _own_goal(agent_id))
                )

        return shaped


class AnnealingRewardWrapper(RewardShapingWrapper):
    """
    RewardShapingWrapper with linear annealing of shaped terms.
    Shaped contribution decays from full weight to zero over `anneal_steps` local steps.
    Used by Agent 3 to gradually shift from shaped rewards toward pure sparse signal.
    """

    def __init__(self, env, env_config=None):
        super().__init__(env, env_config)
        cfg = env_config or {}
        self.anneal_shaped = cfg.get("anneal_shaped_rewards", False)
        self.anneal_steps  = cfg.get("anneal_steps", 4_000_000)
        self._local_steps  = 0

    def _annealing_factor(self):
        if not self.anneal_shaped:
            return 1.0
        return max(0.0, 1.0 - self._local_steps / self.anneal_steps)

    def _apply_shaping(self, rewards, infos):
        shaped  = super()._apply_shaping(rewards, infos)
        factor  = self._annealing_factor()
        if factor < 1.0:
            for aid in shaped:
                sparse         = rewards[aid]
                shaped_contrib = shaped[aid] - sparse
                shaped[aid]    = sparse + factor * shaped_contrib
        self._local_steps += 1
        return shaped


class BCWarmStartModel(TorchModelV2, nn.Module):
    """
    RLLib-compatible policy model that loads BC pretrained weights at init.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        hidden  = model_config.get("fcnet_hiddens", [256, 256])
        obs_dim = obs_space.shape[0]

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim,   hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        )
        self.action_heads = nn.ModuleList([nn.Linear(hidden[1], 3) for _ in range(3)])
        self.value_head   = nn.Linear(hidden[1], 1)
        self._last_feat   = None

        bc_cfg  = model_config.get("custom_model_config") or {}
        bc_path = bc_cfg.get("bc_checkpoint", "")
        if bc_path and os.path.exists(bc_path):
            state = torch.load(bc_path, map_location="cpu")
            missing, unexpected = self.load_state_dict(state, strict=False)
            print(f"[BCWarmStartModel] Loaded weights from {bc_path}")
            if missing:
                print(f"  Missing keys (random init): {missing}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {unexpected}")
        else:
            print(
                f"[BCWarmStartModel] No BC checkpoint at '{bc_path}' — "
                "using random initialisation."
            )

    def forward(self, input_dict, state, seq_lens):
        obs             = input_dict["obs"].float()
        self._last_feat = self.backbone(obs)
        logits          = torch.cat(
            [head(self._last_feat) for head in self.action_heads], dim=-1
        )
        return logits, state

    def value_function(self):
        return self.value_head(self._last_feat).squeeze(1)

try:
    ModelCatalog.register_custom_model("BCWarmStartModel", BCWarmStartModel)
except AttributeError:
    from ray.tune.registry import _global_registry, RLLIB_MODEL
    _global_registry.register(RLLIB_MODEL, "BCWarmStartModel", BCWarmStartModel)


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment. Selects wrapper based on env_config flags.

    env_config flags:
        shaped_rewards        (bool): Enable reward shaping Terms A-H.
        defensive_reward      (bool): Enable Term K (goalie positioning).
        anneal_shaped_rewards (bool): Enable reward annealing (Agent 3 only).
        anneal_steps          (int):  Steps over which to anneal shaped rewards.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )

    needs_shaping   = env_config.get("shaped_rewards",        False)
    needs_defensive = env_config.get("defensive_reward",      False)
    needs_annealing = env_config.get("anneal_shaped_rewards", False)

    _wrapper_keys = {"shaped_rewards", "defensive_reward",
                     "anneal_shaped_rewards", "anneal_steps"}
    make_config = {k: v for k, v in env_config.items() if k not in _wrapper_keys}

    env = soccer_twos.make(**make_config)

    if needs_shaping or needs_defensive:
        if needs_annealing:
            return AnnealingRewardWrapper(env, env_config)
        return RewardShapingWrapper(env, env_config)

    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s