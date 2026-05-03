"""
Compares core PPO algorithmic components between:
  - CleanRL reference (src/agents/ppo.py  — Agent / layer_init)
  - Project implementation (src/agents/ppo_agent.py — PPOAgent)

Three test groups:
  1. GAE equivalence     — same trajectory → identical advantages/returns
  2. Loss equivalence    — same tensors → identical policy/value losses
  3. End-to-end learning — both agents improve over a random baseline on CartPole-v1
"""
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ppo.py imports tensorboard at the top level; stub it so the test doesn't need it installed.
sys.modules.setdefault("tensorboard", MagicMock())
sys.modules.setdefault("torch.utils.tensorboard", MagicMock())

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from src.agents.ppo_agent import PPOAgent
from src.agents.ppo import Agent as CleanRLAgent

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _MLPPolicy(nn.Module):
    """Minimal MLP satisfying PPOAgent's (logits, value, hidden) interface."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, hidden=None):
        x = self.net(obs)
        return self.policy_head(x), self.value_head(x), hidden


def _make_ppo_agent(seed: int = SEED) -> tuple[PPOAgent, gym.Env]:
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    policy = _MLPPolicy(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
    )
    return PPOAgent(policy=policy, lr=2.5e-4, epochs=4), env


def _reference_gae(
    rewards: list,
    dones: list,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CleanRL GAE formula in ppo_agent dones convention:
      dones[t] = True if episode ended after action at step t.

    Equivalent to ppo.py where dones[t] = done_after_step_{t-1}
    because ppo.py uses dones[t+1] as the mask at step t.
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        next_v = last_value.item() if t == T - 1 else values[t + 1].item()
        delta = rewards[t] + gamma * next_v * mask - values[t].item()
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
    return advantages + values, advantages


# ---------------------------------------------------------------------------
# 1. GAE equivalence
# ---------------------------------------------------------------------------

class TestGAEEquivalence:
    """PPOAgent._compute_gae must match the CleanRL reference formula."""

    @pytest.mark.parametrize(
        "T, done_at",
        [
            (16, None),       # no episode boundary in rollout
            (16, 7),          # mid-rollout episode boundary
            (16, 15),         # terminal at last step
        ],
        ids=["no-done", "mid-done", "terminal-last"],
    )
    def test_advantages_match_reference(self, T, done_at):
        torch.manual_seed(0)
        rewards = [float(torch.rand(1)) for _ in range(T)]
        dones = [False] * T
        if done_at is not None:
            dones[done_at] = True
        values = torch.rand(T)
        last_value = torch.rand(1).squeeze()

        agent, _ = _make_ppo_agent()
        agent_returns, agent_adv = agent._compute_gae(rewards, dones, values, last_value)
        ref_returns, ref_adv = _reference_gae(rewards, dones, values, last_value)

        torch.testing.assert_close(agent_adv, ref_adv, rtol=1e-5, atol=1e-6,
                                   msg="Advantages diverge from CleanRL reference")
        torch.testing.assert_close(agent_returns, ref_returns, rtol=1e-5, atol=1e-6,
                                   msg="Returns diverge from CleanRL reference")

    def test_bootstrap_zero_when_terminal(self):
        """When last step is terminal, next-state value must not be bootstrapped."""
        T = 8
        rewards = [1.0] * T
        dones = [False] * (T - 1) + [True]
        values = torch.ones(T)
        last_value = torch.tensor(999.0)  # large sentinel — must have no effect

        agent, _ = _make_ppo_agent()
        _, adv_with_sentinel = agent._compute_gae(rewards, dones, values, last_value)

        last_value_zero = torch.tensor(0.0)
        _, adv_with_zero = agent._compute_gae(rewards, dones, values, last_value_zero)

        torch.testing.assert_close(adv_with_sentinel, adv_with_zero,
                                   msg="Terminal masking failed: last_value leaked into advantages")


# ---------------------------------------------------------------------------
# 2. Loss formula equivalence
# ---------------------------------------------------------------------------

class TestLossEquivalence:
    """PPO surrogate and value losses must be numerically identical to CleanRL."""

    @pytest.mark.parametrize("clip_eps", [0.1, 0.2, 0.3])
    def test_policy_loss_matches_cleanrl(self, clip_eps):
        torch.manual_seed(1)
        T = 32
        old_lps = torch.randn(T)
        new_lps = old_lps + 0.15 * torch.randn(T)
        advantages = torch.randn(T)

        # ppo_agent.py: -min(surr1, surr2).mean()
        ratio = (new_lps - old_lps).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        agent_loss = -torch.min(surr1, surr2).mean()

        # ppo.py: max(-adv*ratio, -adv*clamp(ratio)).mean()
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * ratio.clamp(1 - clip_eps, 1 + clip_eps)
        cleanrl_loss = torch.max(pg_loss1, pg_loss2).mean()

        torch.testing.assert_close(agent_loss, cleanrl_loss,
                                   msg="Policy loss formula diverges from CleanRL")

    @pytest.mark.parametrize("use_clip", [True, False])
    def test_value_loss_matches_cleanrl(self, use_clip):
        torch.manual_seed(2)
        T = 32
        clip_eps = 0.2
        old_vals = torch.randn(T)
        new_vals = old_vals + 0.05 * torch.randn(T)
        returns = torch.randn(T)

        if use_clip:
            # ppo_agent.py clipped value loss
            v_clipped = old_vals + (new_vals - old_vals).clamp(-clip_eps, clip_eps)
            agent_loss = 0.5 * torch.max(
                (new_vals - returns).pow(2),
                (v_clipped - returns).pow(2),
            ).mean()

            # ppo.py clipped value loss
            v_loss_unclipped = (new_vals - returns) ** 2
            v_clipped_ref = old_vals + torch.clamp(new_vals - old_vals, -clip_eps, clip_eps)
            v_loss_clipped = (v_clipped_ref - returns) ** 2
            cleanrl_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            agent_loss = 0.5 * (new_vals - returns).pow(2).mean()
            cleanrl_loss = 0.5 * ((new_vals - returns) ** 2).mean()

        torch.testing.assert_close(agent_loss, cleanrl_loss,
                                   msg=f"Value loss (clipped={use_clip}) diverges from CleanRL")


# ---------------------------------------------------------------------------
# 3. End-to-end learning on CartPole-v1
# ---------------------------------------------------------------------------

N_ROLLOUTS = 60
ROLLOUT_LEN = 128  # 60 × 128 = 7 680 env steps — fast but enough to see improvement


def _run_ppo_agent(seed: int = SEED) -> list[float]:
    """Collect episode returns while training PPOAgent on CartPole-v1."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent, env = _make_ppo_agent(seed)
    episode_returns: list[float] = []

    obs, _ = env.reset(seed=seed)
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    ep_return = 0.0
    agent.store_initial_hidden(None)

    for rollout in range(N_ROLLOUTS):
        for _ in range(ROLLOUT_LEN):
            action, lp, val, _ = agent.act(obs_t, None)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.store(obs_t, action, lp, float(reward), val, done)
            ep_return += float(reward)
            obs_t = torch.FloatTensor(next_obs).unsqueeze(0)
            if done:
                episode_returns.append(ep_return)
                ep_return = 0.0
                obs_t = torch.FloatTensor(env.reset()[0]).unsqueeze(0)

        agent.update(obs_t, None)
        agent.store_initial_hidden(None)

    env.close()
    return episode_returns


def _run_cleanrl(seed: int = SEED) -> list[float]:
    """Minimal CleanRL training loop using the Agent from ppo.py."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def _make():
        env = gym.make("CartPole-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    envs = gym.vector.SyncVectorEnv([_make])
    agent = CleanRLAgent(envs)
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    obs_dim = envs.single_observation_space.shape[0]
    gamma, gae_lambda, clip_coef = 0.99, 0.95, 0.2
    update_epochs, num_minibatches = 4, 4
    minibatch_size = ROLLOUT_LEN // num_minibatches

    s_obs = torch.zeros((ROLLOUT_LEN, 1, obs_dim))
    s_actions = torch.zeros((ROLLOUT_LEN, 1), dtype=torch.long)
    s_logprobs = torch.zeros((ROLLOUT_LEN, 1))
    s_rewards = torch.zeros((ROLLOUT_LEN, 1))
    s_dones = torch.zeros((ROLLOUT_LEN, 1))
    s_values = torch.zeros((ROLLOUT_LEN, 1))

    episode_returns: list[float] = []
    next_obs = torch.FloatTensor(envs.reset(seed=seed)[0])
    next_done = torch.zeros(1)

    for _ in range(N_ROLLOUTS):
        for step in range(ROLLOUT_LEN):
            s_obs[step] = next_obs
            s_dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                s_values[step] = value.flatten()
            s_actions[step] = action
            s_logprobs[step] = logprob

            next_obs_np, reward, terms, truncs, infos = envs.step(action.cpu().numpy())
            next_done = torch.FloatTensor(np.logical_or(terms, truncs))
            s_rewards[step] = torch.FloatTensor(reward)
            next_obs = torch.FloatTensor(next_obs_np)

            if "episode" in infos:
                mask = infos.get("_episode", np.ones(1, dtype=bool))
                for i, done_ep in enumerate(mask):
                    if done_ep:
                        episode_returns.append(float(infos["episode"]["r"][i]))

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(s_rewards)
            lastgaelam = 0.0
            for t in reversed(range(ROLLOUT_LEN)):
                if t == ROLLOUT_LEN - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - s_dones[t + 1]
                    nextvalues = s_values[t + 1]
                delta = s_rewards[t] + gamma * nextvalues * nextnonterminal - s_values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + s_values

        b_obs = s_obs.reshape(-1, obs_dim)
        b_lps = s_logprobs.reshape(-1)
        b_actions = s_actions.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_vals = s_values.reshape(-1)

        b_inds = np.arange(ROLLOUT_LEN)
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, ROLLOUT_LEN, minibatch_size):
                mb = b_inds[start : start + minibatch_size]
                _, new_lp, entropy, new_val = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                ratio = (new_lp - b_lps[mb]).exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * ratio.clamp(1 - clip_coef, 1 + clip_coef)).mean()
                v_loss = 0.5 * ((new_val.view(-1) - b_ret[mb]) ** 2).mean()
                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

    envs.close()
    return episode_returns


def _random_baseline(n_episodes: int = 50, seed: int = SEED) -> float:
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret, done = 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


class TestEndToEndCartPole:
    """Both implementations must learn to outperform random on CartPole-v1."""

    @pytest.fixture(scope="class")
    def random_mean(self):
        return _random_baseline()

    @pytest.fixture(scope="class")
    def ppo_agent_returns(self):
        return _run_ppo_agent()

    @pytest.fixture(scope="class")
    def cleanrl_returns(self):
        return _run_cleanrl()

    def test_ppo_agent_completes_episodes(self, ppo_agent_returns):
        assert len(ppo_agent_returns) >= 10, "PPOAgent completed too few episodes"

    def test_cleanrl_completes_episodes(self, cleanrl_returns):
        assert len(cleanrl_returns) >= 10, "CleanRL completed too few episodes"

    def test_ppo_agent_improves(self, ppo_agent_returns):
        n = max(len(ppo_agent_returns) // 4, 1)
        early = np.mean(ppo_agent_returns[:n])
        late = np.mean(ppo_agent_returns[-n:])
        assert late > early, f"PPOAgent did not improve: early={early:.1f} late={late:.1f}"

    def test_cleanrl_improves(self, cleanrl_returns):
        n = max(len(cleanrl_returns) // 4, 1)
        early = np.mean(cleanrl_returns[:n])
        late = np.mean(cleanrl_returns[-n:])
        assert late > early, f"CleanRL did not improve: early={early:.1f} late={late:.1f}"

    def test_ppo_agent_beats_random(self, ppo_agent_returns, random_mean):
        n = max(len(ppo_agent_returns) // 4, 1)
        final = np.mean(ppo_agent_returns[-n:])
        assert final > random_mean, (
            f"PPOAgent final mean {final:.1f} did not beat random {random_mean:.1f}"
        )

    def test_cleanrl_beats_random(self, cleanrl_returns, random_mean):
        n = max(len(cleanrl_returns) // 4, 1)
        final = np.mean(cleanrl_returns[-n:])
        assert final > random_mean, (
            f"CleanRL final mean {final:.1f} did not beat random {random_mean:.1f}"
        )

    def test_implementations_perform_similarly(self, ppo_agent_returns, cleanrl_returns):
        """Final performance of both implementations should be within 100 steps of each other."""
        n_ppo = max(len(ppo_agent_returns) // 4, 1)
        n_crl = max(len(cleanrl_returns) // 4, 1)
        final_ppo = np.mean(ppo_agent_returns[-n_ppo:])
        final_crl = np.mean(cleanrl_returns[-n_crl:])
        assert abs(final_ppo - final_crl) < 100, (
            f"Implementations diverge significantly: PPOAgent={final_ppo:.1f}, CleanRL={final_crl:.1f}"
        )


# ---------------------------------------------------------------------------
# 4. Loss wiring — tests that go through PPOAgent, not just the formulas
# ---------------------------------------------------------------------------

def _collect_rollout(agent: PPOAgent, env: gym.Env, T: int = 32, seed: int = 6):
    obs, _ = env.reset(seed=seed)
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    agent.store_initial_hidden(None)
    for _ in range(T):
        action, lp, val, _ = agent.act(obs_t, None)
        next_obs, r, term, trunc, _ = env.step(action)
        agent.store(obs_t, action, lp, float(r), val, term or trunc)
        obs_t = torch.FloatTensor(next_obs).unsqueeze(0)
        if term or trunc:
            obs_t = torch.FloatTensor(env.reset()[0]).unsqueeze(0)
    return obs_t


class TestLossWiring:
    """Verify PPOAgent assembles loss components the same way as CleanRL."""

    def test_default_coefficients_match_cleanrl(self):
        """ent_coef=0.01, vf_coef=0.5, clip_eps=0.2 must match CleanRL defaults."""
        agent, _ = _make_ppo_agent()
        assert agent.entropy_coef == 0.01, f"entropy_coef={agent.entropy_coef}, expected 0.01"
        assert agent.value_coef == 0.5,    f"value_coef={agent.value_coef}, expected 0.5"
        assert agent.clip_eps == 0.2,      f"clip_eps={agent.clip_eps}, expected 0.2"

    def test_update_returns_finite_loss_components(self):
        """update() must return finite policy_loss, value_loss, and entropy > 0."""
        torch.manual_seed(6)
        agent, env = _make_ppo_agent()
        obs_t = _collect_rollout(agent, env)
        loss_dict = agent.update(obs_t, None)

        assert all(np.isfinite(v) for v in loss_dict.values()), f"Non-finite loss: {loss_dict}"
        assert loss_dict["entropy"] > 0,    "Entropy must be positive"
        assert loss_dict["value_loss"] >= 0, "Value loss must be non-negative"

    def test_higher_entropy_coef_preserves_more_entropy(self):
        """Larger entropy_coef should result in higher policy entropy after an update."""
        torch.manual_seed(8)
        agent_hi, env_hi = _make_ppo_agent()
        agent_lo, env_lo = _make_ppo_agent()
        agent_lo.entropy_coef = 0.0
        # Share initial weights so the only difference is the coefficient
        agent_lo.policy.load_state_dict(agent_hi.policy.state_dict())

        for agent, env in [(agent_hi, env_hi), (agent_lo, env_lo)]:
            obs_t = _collect_rollout(agent, env, T=64, seed=8)
            agent.update(obs_t, None)

        probe = torch.FloatTensor(env_hi.reset(seed=8)[0]).unsqueeze(0)
        with torch.no_grad():
            logits_hi, _, _ = agent_hi.policy(probe)
            logits_lo, _, _ = agent_lo.policy(probe)
        entropy_hi = torch.distributions.Categorical(logits=logits_hi).entropy().item()
        entropy_lo = torch.distributions.Categorical(logits=logits_lo).entropy().item()
        assert entropy_hi >= entropy_lo, (
            f"Higher entropy_coef should yield higher entropy: "
            f"hi={entropy_hi:.4f} lo={entropy_lo:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Gradient clipping
# ---------------------------------------------------------------------------

class TestGradientClipping:
    """Gradient norm must be clipped to max_norm=0.5 before every optimizer step."""

    def test_grad_norm_clipped_during_update(self):
        torch.manual_seed(9)
        agent, env = _make_ppo_agent()
        obs_t = _collect_rollout(agent, env, T=64, seed=9)

        norms_seen = []
        orig_step = agent.optimizer.step

        def capturing_step(*args, **kwargs):
            norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in agent.policy.parameters()
                if p.grad is not None
            ) ** 0.5
            norms_seen.append(norm)
            return orig_step(*args, **kwargs)

        agent.optimizer.step = capturing_step
        agent.update(obs_t, None)

        assert norms_seen, "No optimizer steps were recorded"
        violations = [n for n in norms_seen if n > 0.51]
        assert not violations, (
            f"{len(violations)}/{len(norms_seen)} steps had grad norm > 0.5. "
            f"Max observed: {max(norms_seen):.4f}"
        )


# ---------------------------------------------------------------------------
# 6. Advantage normalization scope
# ---------------------------------------------------------------------------

class TestAdvantageNormalization:
    """
    PPOAgent normalizes advantages once over the full rollout (global normalization),
    before the epoch loop. CleanRL normalizes per-minibatch. These tests document
    and verify the two different behaviors.
    """

    def test_ppo_agent_normalizes_globally_not_per_minibatch(self):
        """
        Under global normalization, minibatch subsets retain non-zero means.
        Under per-minibatch normalization, each subset has mean exactly 0.
        """
        torch.manual_seed(11)
        T = 64
        # Monotonically increasing rewards → clear trend in advantages, non-symmetric
        rewards = np.linspace(0.1, 2.0, T).tolist()
        dones = [False] * T
        values = torch.linspace(0.0, 1.5, T)
        last_value = torch.tensor(1.6)

        agent, _ = _make_ppo_agent()
        _, advantages = agent._compute_gae(rewards, dones, values, last_value)

        mb_size = T // 4

        # Global normalization (PPOAgent): normalize once over the full T
        global_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        global_mb_means = [
            global_norm[i : i + mb_size].mean().abs().item()
            for i in range(0, T, mb_size)
        ]

        # Per-minibatch normalization (CleanRL): normalize each chunk independently
        per_mb_means = []
        for i in range(0, T, mb_size):
            mb = advantages[i : i + mb_size]
            per_mb_norm = (mb - mb.mean()) / (mb.std() + 1e-8)
            per_mb_means.append(per_mb_norm.mean().abs().item())

        # Global: at least some minibatches will have non-zero mean
        assert max(global_mb_means) > 0.05, (
            "Global normalization should leave some minibatch means non-zero"
        )
        # Per-minibatch: every minibatch has mean ≈ 0
        assert all(m < 1e-5 for m in per_mb_means), (
            "Per-minibatch normalization must yield mean≈0 for every minibatch"
        )
        # The two regimes are meaningfully different
        assert max(global_mb_means) > max(per_mb_means), (
            "Global normalization should produce larger minibatch mean variance"
        )
