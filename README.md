# resaction-nav

## Overview

This project explores **dynamic resolution for efficient visual navigation** in AI2-THOR.

---

## Structure

```
resaction-nav/
├── cfgs/
│   └── train_rl.yaml            # W&B / HuggingFace logging config
├── scripts/
│   ├── train.py                 # Training entry point
│   └── image_resolution.py      # Standalone resolution degradation demo
├── src/
│   ├── agents/
│   │   └── ppo_agent.py         # PPOAgent — clipped surrogate + value loss
│   ├── models/
│   │   └── lstm.py              # PolicyLSTM — recurrent policy/value network
│   ├── simulation/
│   │   ├── thor_env.py          # ThorEnv — Gymnasium-style AI2-THOR environment
│   │   └── thor_camera.py       # ThorCamera — low-level camera wrapper
│   └── utils/
│       └── image_resolution.py  # degrade_resolution() utility
├── pyproject.toml
├── uv.lock
└── README.md

```
## Setup

```bash
uv sync
```

---

## Run

```bash
uv run python scripts/run_pipeline.py --smoke
uv run python scripts/run_pipeline.py --cfg cfgs/train_rl.yaml
uv run python scripts/train.py --cfg cfgs/train_rl.yaml
```

---

## Tests

```bash
make test       # run all tests in tests/
make test-e2e   # run end-to-end pipeline test only
make test-fast  # run fast core unit tests
```

---

## Notes

- PPO scirpt
- LSTM implementation
- training script 
- 

---

## Architecture

**Environment (`ThorEnv`)** — Gymnasium-compatible interface:
- `reset(scene, target_obj_type)` → initial observation
- `step(action_idx)` → `(obs, reward, terminated, truncated, info)`

**Actions (10 total):**

| Index | Action | Effect |
|-------|--------|--------|
| 0–3 | Move Ahead/Right/Left/Back | Navigate; resets resolution to worst level |
| 4–5 | Rotate Right/Left | Turn in place |
| 6–7 | Look Up/Down | Tilt camera |
| 8 | SENSE | Halves downgrade level (better resolution), costs 1 budget unit |
| 9 | DONE | Ends the episode |

**Observations:** Raw RGB frame as a `(3, H, W)` float tensor, degraded to the current resolution level.
> TODO: replace with DinoV2 CLS token + GPS + compass + previous-action embedding (see `PolicyLSTM` docstring).

**Policy (`PolicyLSTM`):** LSTM network that takes the flat observation vector and outputs action logits + value estimate. Memory across timesteps matters because the scene is only partially visible at low resolution.

**Agent (`PPOAgent`):** Standard PPO with clipped surrogate objective, value loss, and entropy regularization.

**Reward components:**

| Signal | Value |
|--------|-------|
| Distance progress | `+distance_scale × Δdist` |
| Step penalty | `−0.002` |
| SENSE penalty | `−0.02` |
| Over-budget SENSE | `−0.05` additional |
| Collision (bump) | `−0.03` |
| Timeout | `−1.0` |
| Success (DONE + visible + close) | `+5.0` |

All weights live in `RewardConfig` (`thor_env.py`) and can be overridden at construction time.
