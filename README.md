# resaction-nav

Reinforcement learning for **dynamic-resolution visual navigation** in AI2-THOR.
The agent navigates 3D indoor scenes to find target objects while learning *when* to spend a limited perception budget to sharpen its camera view.

Full write-up: [index.html](index.html)

---

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

---

## Running

### Smoke test (no GPU, no WandB)

```bash
uv run python scripts/run_pipeline.py --smoke
```

Runs 10 episodes, 8 steps each, logs to console only.

### Local training

```bash
uv run python scripts/train.py --cfg cfgs/train_rl.yaml
```

### Resume from checkpoint

```bash
uv run python scripts/train.py --cfg cfgs/train_rl.yaml --resume checkpoints/ep500.pt
```

### Run a specific agent type

```bash
uv run python scripts/run_pipeline.py --cfg cfgs/train_rl.yaml --agent high_res
```

Available agents: `adaptive` (default), `high_res`, `low_res`, `random_sensing`, `fixed_schedule`

### Cluster (SLURM / IZAR)

```bash
sbatch scripts/run_izar_smoke.sbatch
```

Headless GPU node. Reserves ports per rank, generates CloudRendering config automatically.
Logs to `logs/`.

---

## Config

The single source of truth is `cfgs/train_rl.yaml`. Key sections:

| Section | Purpose |
|---------|---------|
| `training` | Scenes, episode count, parallelism, checkpointing |
| `env` | Resolution, max steps, sensing budget, reward weights, action set |
| `agent` | PPO hyperparameters (lr, clip\_eps, entropy\_coef, epochs) |
| `model` | LSTM hidden dim, layers |
| `visual_encoder` | DinoV2 variant (`dinov2_vitb14` default) |
| `wandb` | Logging toggle, project, entity, run name |
| `trajectory_logging` | Per-step CSV output (off by default) |
| `huggingface` | Checkpoint push to HF Hub (off by default) |

Experiment-specific configs are in `cfgs/train_rl.rok_*.yaml`.

---

## Testing

```bash
# Fast unit tests (no AI2-THOR)
make test-fast

# All unit tests
make test

# End-to-end pipeline smoke
make test-e2e
```

---

## Project structure

```
resaction-nav/
├── cfgs/               ← YAML configs
├── scripts/
│   ├── train.py        ← main entry point
│   ├── run_pipeline.py ← unified launcher (smoke mode, agent override)
│   ├── make_sweep.py   ← generate hyperparameter sweep configs
│   └── run_izar_smoke.sbatch
├── src/
│   ├── agents/
│   │   ├── ppo_agent.py
│   │   └── baselines/  ← high_res, low_res, random_sensing, fixed_schedule
│   ├── models/
│   │   ├── lstm.py         ← PolicyLSTM (recurrent policy + value head)
│   │   └── agent_policy.py ← AgentPolicy (DinoV2 + target embedding + LSTM)
│   ├── simulation/
│   │   └── thor_env.py ← ThorEnv (AI2-THOR wrapper, dynamic resolution)
│   └── utils/
│       ├── image_resolution.py
│       └── trajectory_logger.py
└── tests/
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed component breakdown.
