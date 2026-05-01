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
    ):
        super().__init__()
        input_dim = vis_dim + 3 + 2 + n_actions + 1 + 1  # see docstring breakdown

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
        out = out[:, -1, :]  # last timestep

        logits = self.policy_head(out)   # (batch, n_actions)
        value  = self.value_head(out)    # (batch, 1)
        return logits, value, hidden
