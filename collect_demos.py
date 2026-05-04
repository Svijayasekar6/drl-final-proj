"""
  Player 1: WASD (move) + QE (rotate)
  Player 2: Arrow keys (move) + ,. (rotate)
  ESC: quit early

"""

import argparse
import importlib
import os
import sys

import numpy as np
import soccer_twos

from input_handlers import (
    InputHandler, KeyboardInputHandler, RandomInputHandler,
    PLAYER1_KEYS, PLAYER2_KEYS,
)

# ── Config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--scored",   type=int, default=50,
                    help="Target episodes where ≥1 goal was scored")
parser.add_argument("--draws",    type=int, default=20,
                    help="Target timeout draw episodes")
parser.add_argument("--out-dir",  default="demos")
parser.add_argument("--random",   action="store_true",
                    help="Use random input handlers instead of keyboard (for testing)")
parser.add_argument("--worker-id", type=int, default=1,
                    help="Soccer-twos worker id (change if port conflict)")
args = parser.parse_args()

TARGET_SCORED   = args.scored
TARGET_DRAWS    = args.draws
DEMOS_DIR       = args.out_dir
EPISODE_TIMEOUT = 5000

os.makedirs(DEMOS_DIR, exist_ok=True)


def load_baseline_agent(env):
    """Import and instantiate the ceia_baseline_agent."""
    module = importlib.import_module("ceia_baseline_agent")
    cls    = getattr(module, "Agent", None) or getattr(module, "RayAgent")
    return cls(env)


def run_collection(player1_handler: InputHandler, player2_handler: InputHandler):
    try:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((400, 120))
        pygame.display.set_caption("Demo Collector  |  ESC to quit")
        use_pygame = True
    except Exception:
        use_pygame = False
        print("pygame not available — running headless (no keyboard input possible).")

    env      = soccer_twos.make(watch=True, worker_id=args.worker_id)
    baseline = load_baseline_agent(env)

    scored_ep = 0
    draw_ep   = 0
    ep_idx    = 0

    print(f"Target: {TARGET_SCORED} scored + {TARGET_DRAWS} draw episodes")
    if not args.random:
        print("Player 1: WASD + QE  |  Player 2: Arrow keys + ,.")
    print("Collecting...\n")

    while scored_ep < TARGET_SCORED or draw_ep < TARGET_DRAWS:
        obs  = env.reset()
        done = False
        step = 0

        ep_data = {agent_id: dict(
            observations=[], actions=[], sparse_rewards=[],
            player_pos=[], player_vel=[], player_rot=[],
            ball_pos=[], ball_vel=[],
        ) for agent_id in (0, 1)}

        blue_rew   = 0.0
        orange_rew = 0.0

        while not done and step < EPISODE_TIMEOUT:
            if use_pygame:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        _finish(env, use_pygame)
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        _finish(env, use_pygame)
                        return

            a0 = player1_handler.get_action()
            a1 = player2_handler.get_action()

            baseline_obs     = {0: obs[2], 1: obs[3]}
            baseline_actions = baseline.act(baseline_obs)
            a2               = baseline_actions[0]
            a3               = baseline_actions[1]

            actions = {0: a0, 1: a1, 2: a2, 3: a3}
            next_obs, rewards, dones, infos = env.step(actions)

            for agent_id in (0, 1):
                info = infos.get(agent_id, {})
                ep_data[agent_id]["observations"].append(obs[agent_id].copy())
                ep_data[agent_id]["actions"].append(actions[agent_id].copy())
                ep_data[agent_id]["sparse_rewards"].append(float(rewards[agent_id]))
                ep_data[agent_id]["player_pos"].append(
                    np.asarray(info["player_info"]["position"]).copy()
                    if info else np.zeros(2)
                )
                ep_data[agent_id]["player_vel"].append(
                    np.asarray(info["player_info"]["velocity"]).copy()
                    if info else np.zeros(2)
                )
                ep_data[agent_id]["player_rot"].append(
                    float(info["player_info"]["rotation_y"]) if info else 0.0
                )
                ep_data[agent_id]["ball_pos"].append(
                    np.asarray(info["ball_info"]["position"]).copy()
                    if info else np.zeros(2)
                )
                ep_data[agent_id]["ball_vel"].append(
                    np.asarray(info["ball_info"]["velocity"]).copy()
                    if info else np.zeros(2)
                )

            blue_rew   += rewards[0] + rewards[1]
            orange_rew += rewards[2] + rewards[3]
            obs  = next_obs
            done = dones.get("__all__", False)
            step += 1

        if blue_rew > 0:
            outcome = 1
            scored_ep += 1
        elif blue_rew < 0:
            outcome = -1
            scored_ep += 1
        else:
            outcome = 0
            draw_ep += 1

        label = {1: "WIN", -1: "LOSS", 0: "DRAW"}[outcome]
        print(
            f"  ep {ep_idx:04d} | {label:4s} | steps={step:5d} "
            f"| blue={blue_rew:+.1f} org={orange_rew:+.1f} "
            f"| scored={scored_ep}/{TARGET_SCORED} draws={draw_ep}/{TARGET_DRAWS}"
        )

        for agent_id in (0, 1):
            d    = ep_data[agent_id]
            path = os.path.join(DEMOS_DIR, f"ep{ep_idx:04d}_agent{agent_id}.npz")
            np.savez_compressed(
                path,
                observations   = np.array(d["observations"],   dtype=np.float32),
                actions        = np.array(d["actions"],        dtype=np.int32),
                sparse_rewards = np.array(d["sparse_rewards"], dtype=np.float32),
                player_pos     = np.array(d["player_pos"],     dtype=np.float32),
                player_vel     = np.array(d["player_vel"],     dtype=np.float32),
                player_rot     = np.array(d["player_rot"],     dtype=np.float32),
                ball_pos       = np.array(d["ball_pos"],       dtype=np.float32),
                ball_vel       = np.array(d["ball_vel"],       dtype=np.float32),
                outcome        = np.array(outcome,             dtype=np.int32),
            )

        ep_idx += 1

    _finish(env, use_pygame)
    print(f"\nDone. {ep_idx} episodes saved to {DEMOS_DIR}/")
    print(f"Transfer to PACE and run: python bc_pretrain.py --demos-dir {DEMOS_DIR}/")


def _finish(env, use_pygame):
    env.close()
    if use_pygame:
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    if args.random:
        p1 = RandomInputHandler()
        p2 = RandomInputHandler()
        print("Using random input handlers for testing.")
    else:
        p1 = KeyboardInputHandler(PLAYER1_KEYS)
        p2 = KeyboardInputHandler(PLAYER2_KEYS)

    run_collection(p1, p2)