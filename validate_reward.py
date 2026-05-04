"""
Reward function validation script.

Steps a random policy through the env for N steps and prints a per-step
breakdown of each shaped reward term. Use this to:
"""

import argparse

import numpy as np
import soccer_twos

from utils import compute_shaped_reward_offline, _own_goal, _opp_goal

parser = argparse.ArgumentParser()
parser.add_argument("--steps",     type=int, default=1000)
parser.add_argument("--worker-id", type=int, default=2)
parser.add_argument("--no-render", action="store_true")
args = parser.parse_args()


def main():
    env = soccer_twos.make(
        render=not args.no_render,
        worker_id=args.worker_id,
    )
    obs  = env.reset()
    step = 0
    done = False

    # Accumulators for each term
    totals = dict(A=0.0, B=0.0, C=0.0, D2=0.0, F=0.0, G=0.0, H=0.0, K=0.0, goal=0.0)
    fires  = {k: 0 for k in totals}
    nan_count = 0

    # Delta state for agents
    prev_ball_dist      = {}
    prev_ball_own_dist  = {}

    print(f"Validating reward shaping over {args.steps} steps...\n")
    print(f"{'step':>5} {'agent':>5} | {'A':>8} {'B':>8} {'C':>8} {'D2':>8} "
          f"{'F':>8} {'G':>8} {'H':>8} {'K':>8} {'goal':>6} | {'total':>9}")
    print("-" * 110)

    while not done and step < args.steps:
        actions     = {i: env.action_space.sample() for i in obs}
        next_obs, rewards, dones, infos = env.step(actions)

        for agent_id, sparse_reward in rewards.items():
            info = infos.get(agent_id, {})
            if not info:
                if step == 0:
                    print(
                        f"  [WARNING] infos[{agent_id}] is empty at step {step}. "
                    )
                continue

            player_pos = np.asarray(info["player_info"]["position"])
            ball_pos   = np.asarray(info["ball_info"]["position"])
            ball_vel   = np.asarray(info["ball_info"]["velocity"])

            all_player_pos = {}
            for aid in infos:
                ainfo = infos.get(aid, {})
                if ainfo:
                    all_player_pos[aid] = np.asarray(ainfo["player_info"]["position"])

            total, comps = compute_shaped_reward_offline(
                agent_id=agent_id,
                player_pos=player_pos,
                ball_pos=ball_pos,
                ball_vel=ball_vel,
                sparse_reward=sparse_reward,
                all_player_pos=all_player_pos,
                prev_ball_dist=prev_ball_dist.get(agent_id),
                prev_ball_own_goal_dist=prev_ball_own_dist.get(agent_id),
            )

            if np.isnan(total) or any(np.isnan(v) for v in comps.values()):
                nan_count += 1
                print(f"  [NaN] step={step} agent={agent_id} comps={comps}")

            for k in totals:
                totals[k] += comps[k]
                if abs(comps[k]) > 1e-9:
                    fires[k] += 1

            if step < 50 or sparse_reward != 0.0:
                print(
                    f"{step:>5} {agent_id:>5} | "
                    f"{comps['A']:>8.4f} {comps['B']:>8.4f} {comps['C']:>8.4f} "
                    f"{comps['D2']:>8.4f} {comps['F']:>8.4f} {comps['G']:>8.4f} "
                    f"{comps['H']:>8.4f} {comps['K']:>8.4f} {comps['goal']:>6.1f} | "
                    f"{total:>9.4f}"
                )

            prev_ball_dist[agent_id]     = float(np.linalg.norm(ball_pos - player_pos))
            prev_ball_own_dist[agent_id] = float(
                np.linalg.norm(ball_pos - _own_goal(agent_id))
            )

        obs  = next_obs
        done = dones.get("__all__", False)
        step += 1

    env.close()

    print("\n" + "=" * 110)
    print(f"{'SUMMARY':^110}")
    print("=" * 110)
    print(f"Steps run: {step} | NaN occurrences: {nan_count}")
    print()
    print(f"{'Term':<8} {'Total':>12} {'Mean/step':>12} {'Fire count':>12} {'Fire %':>10}")
    print("-" * 60)
    n_agents = 4
    total_agent_steps = step * n_agents
    for k in ["A", "B", "C", "D2", "F", "G", "H", "K", "goal"]:
        v    = totals[k]
        fc   = fires[k]
        mean = v / max(total_agent_steps, 1)
        pct  = 100.0 * fc / max(total_agent_steps, 1)
        print(f"  {k:<6} {v:>12.4f} {mean:>12.6f} {fc:>12d} {pct:>9.1f}%")

    print()
    sparse_total = totals["goal"]
    shaped_total = sum(totals[k] for k in ["A", "B", "C", "D2", "F", "G", "H", "K"])
    print(f"Shaped total (all terms): {shaped_total:.4f}")
    print(f"Sparse total (goals):     {sparse_total:.4f}")
    if sparse_total != 0:
        print(f"Shaped / Sparse ratio:    {abs(shaped_total / sparse_total):.2f}x")
    print()
    if nan_count == 0:
        print("[OK] No NaN values detected.")
    else:
        print(f"[FAIL] {nan_count} NaN values — check reward coefficient for instability.")

    zero_terms = [k for k in ["A", "B", "C", "D2", "F", "G", "H", "K"] if fires[k] == 0]
    if zero_terms:
        print(f"[WARN] Terms never fired: {zero_terms} — check gates/thresholds or run more steps.")
    else:
        print("[OK] All reward terms fired at least once.")


if __name__ == "__main__":
    main()