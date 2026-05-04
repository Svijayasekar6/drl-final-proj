"""

Loads .npz demo files from collect_demos.py, computes signed importance weights
from discounted shaped returns, then trains a BCPolicy (same architecture as
BCWarmStartModel in utils.py) with weighted cross-entropy loss.

Output: bc_checkpoint.pth — loaded by train_agent3.py via BCWarmStartModel.

"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_shaped_reward_offline, _own_goal

OBS_DIM     = 336
HIDDEN_DIM  = 256
ACTION_DIMS = [3, 3, 3]


class BCPolicy(nn.Module):
    """
    Standalone BC policy.  Architecture exactly mirrors BCWarmStartModel:
    - backbone: Linear(336,256) ReLU → Linear(256,256) ReLU
    - action_heads: 3 × Linear(256,3)  (one per MultiDiscrete branch)
    - value_head: Linear(256,1)
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN_DIM),   nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        )
        self.action_heads = nn.ModuleList(
            [nn.Linear(HIDDEN_DIM, n) for n in ACTION_DIMS]
        )
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, obs):
        feat   = self.backbone(obs)
        logits = [head(feat) for head in self.action_heads]
        value  = self.value_head(feat)
        return logits, value


def compute_weights(demo_path, gamma=0.99, beta=1.0):
    """
    Load one .npz demo file and return (obs, actions, weights) arrays.

    Weights: w_t = sign(G_t) * exp(|G_t| / beta)
      w_t > 0 → imitate strongly
      w_t ≈ 0 → neutral
      w_t < 0 → actively avoid

    Returns arrays of shape (T, 336), (T, 3), (T,).
    """
    data = np.load(demo_path, allow_pickle=True)
    obs          = data["observations"]           # (T, 336)
    actions      = data["actions"]                # (T, 3)
    sparse_rews  = data["sparse_rewards"]         # (T,)
    player_pos   = data["player_pos"]             # (T, 2)
    ball_pos     = data["ball_pos"]               # (T, 2)
    ball_vel     = data["ball_vel"]               # (T, 2)

    # Infer agent_id from filename (ep????_agent{N}.npz)
    fname    = os.path.basename(demo_path)
    agent_id = int(fname.split("_agent")[1].split(".")[0])

    T      = len(obs)
    shaped = np.zeros(T, dtype=np.float64)
    prev_bd  = None
    prev_bod = None

    for t in range(T):
        shaped[t], _ = compute_shaped_reward_offline(
            agent_id=agent_id,
            player_pos=player_pos[t],
            ball_pos=ball_pos[t],
            ball_vel=ball_vel[t],
            sparse_reward=float(sparse_rews[t]),
            prev_ball_dist=prev_bd,
            prev_ball_own_goal_dist=prev_bod,
        )
        prev_bd  = float(np.linalg.norm(ball_pos[t] - player_pos[t]))
        prev_bod = float(np.linalg.norm(ball_pos[t] - _own_goal(agent_id)))

    # Discounted returns (backward pass)
    G       = np.zeros(T, dtype=np.float64)
    running = 0.0
    for t in reversed(range(T)):
        running = shaped[t] + gamma * running
        G[t]    = running

    # Normalise to stabilise beta scaling
    G_std  = G.std() + 1e-8
    G_norm = G / G_std

    # Signed exponential weights
    w = np.sign(G_norm) * np.exp(np.abs(G_norm) / beta)
    w = np.clip(w, -10.0, 10.0).astype(np.float32)

    return obs.astype(np.float32), actions.astype(np.int64), w


def weighted_ce_loss(logits_list, actions, weights):
    """
    Factored weighted cross-entropy across MultiDiscrete([3,3,3]) branches.
    logits_list: list of 3 tensors each (B, 3)
    actions:     (B, 3) int64
    weights:     (B,)  float  — sign drives imitation vs avoidance
    """
    loss = torch.zeros(actions.shape[0], device=actions.device)
    for dim, logits in enumerate(logits_list):
        ce   = F.cross_entropy(logits, actions[:, dim], reduction="none")
        loss = loss + ce
    weighted = weights * loss
    return weighted.mean()


def train_bc(
    demos_dir="demos",
    out_path="bc_checkpoint.pth",
    epochs=50,
    lr=3e-4,
    beta=1.0,
    gamma=0.99,
    batch_size=256,
):
    demo_files = sorted(glob.glob(os.path.join(demos_dir, "*.npz")))
    if not demo_files:
        raise FileNotFoundError(
            f"No .npz files found in '{demos_dir}/'. "
            "Run collect_demos.py first."
        )

    print(f"Loading {len(demo_files)} demo files from {demos_dir}/...")
    all_obs, all_actions, all_weights = [], [], []

    for path in demo_files:
        obs, acts, w = compute_weights(path, gamma=gamma, beta=beta)
        all_obs.append(obs)
        all_actions.append(acts)
        all_weights.append(w)

    obs_t     = torch.FloatTensor(np.concatenate(all_obs))
    actions_t = torch.LongTensor(np.concatenate(all_actions))
    weights_t = torch.FloatTensor(np.concatenate(all_weights))

    N = len(obs_t)
    print(f"Dataset: {N:,} transitions")
    print(
        f"Weight stats: min={weights_t.min():.2f}  max={weights_t.max():.2f}  "
        f"mean={weights_t.mean():.2f}  std={weights_t.std():.2f}"
    )
    print(
        "If max weight >> 10 before clipping, increase --beta. "
        "If most weights ≈ 0, decrease --beta."
    )

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = BCPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    obs_t     = obs_t.to(device)
    actions_t = actions_t.to(device)
    weights_t = weights_t.to(device)

    print(f"\nTraining BC for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        model.train()
        perm       = torch.randperm(N, device=device)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx              = perm[i:i + batch_size]
            logits_list, _   = model(obs_t[idx])
            loss             = weighted_ce_loss(logits_list, actions_t[idx], weights_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        print(f"  Epoch {epoch+1:3d}/{epochs} | loss={epoch_loss/N:.5f}")

    torch.save(model.state_dict(), out_path)
    print(f"\nBC checkpoint saved → {out_path}")
    print("Run: python train_agent3.py --bc-checkpoint", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos-dir", default="demos")
    parser.add_argument("--out",       default="bc_checkpoint.pth")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--beta",      type=float, default=1.0,
                        help="Weight temperature: lower=sharper contrast")
    parser.add_argument("--gamma",     type=float, default=0.99)
    parser.add_argument("--batch-size", type=int,  default=256)
    args = parser.parse_args()

    train_bc(
        demos_dir  = args.demos_dir,
        out_path   = args.out,
        epochs     = args.epochs,
        lr         = args.lr,
        beta       = args.beta,
        gamma      = args.gamma,
        batch_size = args.batch_size,
    )