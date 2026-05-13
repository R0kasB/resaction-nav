import torch
import torch.nn as nn

from src.models.agent_policy import AgentPolicy
from src.models.lstm import PolicyLSTM
from src.models.visual_encoder import DinoV2Encoder


class _DummyDino(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.ones(batch, 768, device=x.device)


def test_policy_lstm_emits_logits_and_value_shapes():
    model = PolicyLSTM(vis_dim=768, n_actions=10, hidden_dim=64, lstm_layers=1)
    obs = torch.randn(4, 768 + 3 + 2 + 10 + 1 + 1 + 8)
    logits, value, hidden = model(obs)
    assert tuple(logits.shape) == (4, 10)
    assert tuple(value.shape) == (4, 1)
    assert hidden is not None


def test_dinov2_encoder_is_frozen(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _DummyDino())
    encoder = DinoV2Encoder(model_name="dinov2_vitb14", device="cpu")

    params = list(encoder.model.parameters())
    if params:
        assert all(not p.requires_grad for p in params)


def test_agent_policy_stacks_encoder_and_lstm(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _DummyDino())
    policy = AgentPolicy(
        n_actions=10,
        encoder_name="dinov2_vitb14",
        hidden_dim=64,
        lstm_layers=1,
        device="cpu",
    )

    image = torch.rand(2, 3, 64, 64)
    aux = torch.randn(2, 3 + 2 + 10 + 1 + 1 + 8)
    logits, value, hidden = policy(image, aux, hidden=None)

    assert tuple(logits.shape) == (2, 10)
    assert tuple(value.shape) == (2, 1)
    assert hidden is not None
