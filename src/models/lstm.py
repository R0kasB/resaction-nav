import torch
import torch.nn as nn


class PolicyLSTM(nn.Module):
    """
    Recurrent policy network for the dynamic-resolution navigation agent.

    Input at each step (flat vector):
      - visual features : vis_dim   (DinoV2 CLS token, or raw-frame CNN features)
      - GPS position    : 3         (x, y, z)
      - compass         : 2         (yaw/360, camera_horizon/360)
      - previous action : n_actions (one-hot)
      - resolution level: 1         (current_downgrade / base_downgrade, normalised)
      - sensing budget  : 1         (remaining_budget / max_budget, normalised)
      - target object embedding: target_object_embed_dim

    Outputs:
      - action_logits : (batch, n_actions)
      - value          : (batch, 1)
      - hidden         : updated LSTM hidden state

    The LSTM gives the agent memory across timesteps, which matters because
    the scene is only partially observable at low resolution.
    """

    def __init__(
        self,
        vis_dim: int = 768,   # DinoV2-base CLS token dimension
        n_actions: int = 10,
        hidden_dim: int = 512,
        lstm_layers: int = 1,
        target_object_embed_dim: int = 8,
    ):
        super().__init__()
        input_dim = (
            vis_dim
            + 2                          # gps: (x, z)
            + 4                          # compass: sin/cos of yaw and horizon
            + n_actions                  # prev_action one-hot
            + 1                          # resolution level
            + 1                          # sensing budget
            + target_object_embed_dim    # target embedding (expanded from idx in AgentPolicy)
        )

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, hidden=None):
        """
        obs    : (batch, input_dim)  or  (batch, T, input_dim) for sequence input
        hidden : (h_n, c_n) LSTM state, or None to start fresh

        Returns: (action_logits, value, hidden)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, input_dim)

        out, hidden = self.lstm(obs, hidden)
        # single step: collapse T=1 → (batch, hidden_dim)
        # full sequence: keep all timesteps → (batch, T, hidden_dim)
        out = out.squeeze(1) if out.size(1) == 1 else out

        logits = self.policy_head(out)   # (batch, n_actions) or (batch, T, n_actions)
        value  = self.value_head(out)    # (batch, 1)         or (batch, T, 1)
        return logits, value, hidden
