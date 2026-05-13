import torch
import torch.nn as nn

from src.models.visual_encoder import DinoV2Encoder
from src.models.lstm import PolicyLSTM


class AgentPolicy(nn.Module):
    """
    Full agent policy:
        raw image + auxiliary features
        -> frozen visual encoder
        -> PolicyLSTM
        -> action logits + value
    """

    def __init__(
        self,
        n_actions: int,
        encoder_name: str = "dinov2_vitb14",
        hidden_dim: int = 512,
        lstm_layers: int = 1,
        target_object_embed_dim: int = 8,
        device: str | torch.device | None = None,
    ):
        super().__init__()

        self.visual_encoder = DinoV2Encoder(
            model_name=encoder_name,
            device=device,
        )

        self.policy_lstm = PolicyLSTM(
            vis_dim=self.visual_encoder.output_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            target_object_embed_dim=target_object_embed_dim,
        )

    def forward(
        self,
        image: torch.Tensor,
        aux_features: torch.Tensor,
        hidden=None,
    ):
        if aux_features.dim() == 1:
            aux_features = aux_features.unsqueeze(0)

        with torch.no_grad():
            visual_features = self.visual_encoder(image)

        obs = torch.cat(
            [visual_features, aux_features],
            dim=-1,
        )

        logits, value, hidden = self.policy_lstm(obs, hidden)

        return logits, value, hidden