# resaction-nav

## Overview

This project explores **dynamic resolution for efficient visual navigation** in AI2-THOR.

---

## TODO

P0: Before First Real Training
- [ ] Fix target conditioning: add target class encoding to `ThorEnv.get_aux_features()` or restrict `cfgs/train_rl.yaml` to a fixed `target_object_types` list.
- [ ] Update PolicyLSTM input dimension in `src/models/lstm.py` if target features are added.
- [ ] Add tests for target conditioning in `tests/test_thor_env_unit.py` and `tests/test_models_architecture_unit.py`.
- [ ] Add `target_object_types` / `fixed_target_object_type` config fields to `cfgs/train_rl.yaml`.
- [ ] Decide whether SENSE improvement should affect the current observation or next-step observation; current code applies it next step, while the report wording may imply immediate improvement.
- [ ] Run `uv run python scripts/run_pipeline.py --smoke` on the target machine before launching long jobs.
- [ ] Update `README.md` and `ARCHITECTURE.md` to match the current DINOv2 + aux-feature policy, since both still contain older/raw-observation notes.

P1: Training Pipeline
- [ ] Add parallel rollout support to `scripts/train.py` or move rollout collection into a new `src/training/rollout.py`.
- [ ] Extend PPOAgent in `src/agents/ppo_agent.py` to support multi-env rollout buffers, per-env done masks, and per-env LSTM hidden states.
- [ ] Add config fields: `num_envs`, `rollout_steps`, `minibatch_size`, `update_epochs`, and maybe `seed_start`.
- [ ] Log `success_rate`, `episode_length`, `num_sense_actions`, `final_downgrade`, `final_sensing_budget`, and PPO losses for every run.
- [ ] Save full training config into each checkpoint or output directory for reproducibility.
- [ ] Add a deterministic/eval action mode to `PPOAgent.act()` so evaluation can use argmax instead of sampling.

P1: Baselines
- [ ] Implement low-resolution baseline: never use SENSE.
- [ ] Implement high-resolution baseline: always observe at full resolution or force env downgrade to 0.
- [ ] Implement random-sensing baseline: sample SENSE with a configurable probability while budget remains.
- [ ] Implement fixed-schedule baseline: sense every k steps or at predefined timesteps.
- [ ] Put baseline policies in a new module such as src/agents/baselines.py.
- [ ] Add unit tests for baseline action behavior.

P1: Evaluation
- [ ] Create `scripts/evaluate.py` that loads trained checkpoints and runs fixed episodes across selected scenes.
- [ ] Add evaluation config, likely `cfgs/eval.yaml`, with scenes, seeds, target classes, number of episodes, and baseline list.
- [ ] Extend `ThorEnv.step()` info with `target_obj_type`, `distance_to_target`, `closest_distance`, and maybe `path_length`.
- [ ] Compute metrics: success rate, average return, average episode length, distance-to-goal reduction, sensing count, sensing cost, remaining budget, and final resolution.
- [ ] Add SPL only after deciding how to compute shortest-path distance in AI2-THOR.
- [ ] Write evaluation outputs as CSV/JSON under `output/eval/`.

P2: Reward Tuning
- [ ] Add named reward presets in config for sensing-cheap, sensing-expensive, sparse, and shaped variants.
- [ ] Run controlled sweeps over sense_penalty, oversensing_penalty, success_reward, and distance_scale.
- [ ] Record each reward config in W&B and in the output artifact.
- [ ] Add tests ensuring timeout, failed DONE, valid SENSE, invalid SENSE, and collision rewards match RewardConfig.

P2: Analysis And Plots
- [ ] Create `scripts/plot_results.py`.
- [ ] Plot reward evolution, success rate, episode length, sensing frequency, final downgrade, and remaining budget.
- [ ] Add sensing-behavior analysis: sense timing by episode fraction, after rotations, near target, before DONE, and as a function of remaining budget.
- [ ] Save plots under `output/plots/`.

P2: Documentation Cleanup
- [x] Fix README typo: “PPO scirpt”.
- [ ] Remove stale TODOs saying DINOv2/GPS/compass are not implemented.
- [ ] Document the exact current observation contract: image tensor plus auxiliary features.
- [ ] Add a short “How to run first controlled experiment” section with smoke, train, evaluate, and plot commands.

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

- PPO script
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
