import math

import torch
import torch.nn as nn

from src.agents.ppo_agent import PPOAgent


class DummyPolicy(nn.Module):
    def __init__(self, image_shape=(3, 8, 8), aux_dim=17, n_actions=5):
        super().__init__()
        self.image_dim = image_shape[0] * image_shape[1] * image_shape[2]
        self.aux_dim = aux_dim
        self.backbone = nn.Sequential(
            nn.Linear(self.image_dim + aux_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, image: torch.Tensor, aux_features: torch.Tensor, hidden=None):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if aux_features.dim() == 1:
            aux_features = aux_features.unsqueeze(0)

        image_flat = image.flatten(start_dim=1)
        x = torch.cat([image_flat, aux_features], dim=1)
        h = self.backbone(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value, hidden


def test_update_empty_buffer_returns_zero_metrics():
    policy = DummyPolicy()
    agent = PPOAgent(policy=policy, epochs=2)

    metrics = agent.update(
        next_image=torch.zeros(3, 8, 8),
        next_aux_features=torch.zeros(17),
        final_hidden=None,
    )

    assert metrics == {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}


def test_rollout_store_and_update_runs_and_clears_buffer():
    torch.manual_seed(0)
    policy = DummyPolicy(n_actions=4)
    agent = PPOAgent(policy=policy, epochs=2, clip_eps=0.2)

    hidden = None
    agent.store_initial_hidden(hidden)

    image = torch.randn(3, 8, 8)
    aux = torch.randn(17)
    for t in range(6):
        action, log_prob, value, hidden = agent.act(image=image, aux_features=aux, hidden=hidden)
        done = t == 5
        reward = float(t) * 0.1
        agent.store(
            image=image,
            aux_features=aux,
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
        )
        image = torch.randn(3, 8, 8)
        aux = torch.randn(17)

    result = agent.update(next_image=image, next_aux_features=aux, final_hidden=hidden)

    assert set(result.keys()) == {"policy_loss", "value_loss", "entropy"}
    assert math.isfinite(result["policy_loss"])
    assert math.isfinite(result["value_loss"])
    assert math.isfinite(result["entropy"])
    assert agent.buffer == []
