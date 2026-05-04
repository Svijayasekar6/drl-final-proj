"""
Self-contained BCWarmStartModel for agent3 packaging.
This file is a copy of the BCWarmStartModel defined in utils.py.
It must be self-contained so agent3/ can be loaded without the training repo.

The bc_checkpoint.pth is NOT needed at evaluation time — weights are baked
into the RLLib checkpoint. This file only needs to exist so RLLib can
deserialise the model architecture from the checkpoint.
"""

import os

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class BCWarmStartModel(TorchModelV2, nn.Module):
    """
    Factored-head policy model matching BCPolicy in bc_pretrain.py.
    Architecture: Linear(obs,256) ReLU → Linear(256,256) ReLU → 3×Linear(256,3)
    Value head: Linear(256,1) independent.
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

        # At evaluation time bc_checkpoint is not needed — skip loading
        bc_cfg  = model_config.get("custom_model_config") or {}
        bc_path = bc_cfg.get("bc_checkpoint", "")
        if bc_path and os.path.exists(bc_path):
            state = torch.load(bc_path, map_location="cpu")
            self.load_state_dict(state, strict=False)

    def forward(self, input_dict, state, seq_lens):
        obs             = input_dict["obs"].float()
        self._last_feat = self.backbone(obs)
        logits          = torch.cat(
            [head(self._last_feat) for head in self.action_heads], dim=-1
        )
        return logits, state

    def value_function(self):
        return self.value_head(self._last_feat).squeeze(1)