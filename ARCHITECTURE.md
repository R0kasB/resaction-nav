# Architecture

## What this project does

An RL agent navigates AI2-THOR indoor environments to find a target object.
The core research idea is **dynamic resolution**: the agent always starts with a heavily downgraded (blurry) camera view and can spend a limited budget of `SENSE` actions to sharpen it — at a cost — before any movement resets the view back to blurry.

---

## File map

```
resaction-nav/
├── cfgs/
│   ├── train_rl.yaml                  ← canonical base config
│   ├── train_rl.rok_adaptive.yaml     ← single-target adaptive experiment
│   ├── train_rl.rok_multi.yaml        ← multi-target high-res baseline
│   ├── train_rl.rok_poc_high_res.yaml ← simplified PoC (high-res, no SENSE)
│   ├── train_quick.yaml               ← 20-episode smoke test
│   └── sweep/                         ← generated grid configs
├── scripts/
│   ├── train.py                       ← cluster entry point, full training loop
│   ├── run_pipeline.py                ← unified launcher (smoke mode, agent override)
│   ├── make_sweep.py                  ← generate sweep grid configs
│   └── run_izar_smoke.sbatch          ← SLURM script for headless GPU cluster
├── src/
│   ├── agents/
│   │   ├── ppo_agent.py               ← PPOAgent: act / store / update_from_rollouts
│   │   └── baselines/
│   │       ├── high_res.py            ← fixed high-resolution (no SENSE)
│   │       ├── low_res.py             ← always worst resolution
│   │       ├── random_sensing.py      ← SENSE with fixed probability
│   │       └── fixed_schedule.py      ← SENSE every N steps
│   ├── models/
│   │   ├── lstm.py                    ← PolicyLSTM: recurrent policy + value head
│   │   └── agent_policy.py            ← AgentPolicy: DinoV2 + target embedding + LSTM
│   ├── simulation/
│   │   └── thor_env.py                ← ThorEnv: Gymnasium wrapper, dynamic resolution
│   └── utils/
│       ├── image_resolution.py        ← degrade_resolution()
│       └── trajectory_logger.py       ← per-step CSV + scene layout JSON
└── tests/
    ├── conftest.py
    ├── test_thor_env_unit.py
    ├── test_ppo_agent_unit.py
    ├── test_models_architecture_unit.py
    ├── test_train_script_unit.py
    └── test_pipeline_end_to_end.py
```

---

## Data flow

```
cfgs/train_rl.yaml
       │
       ▼
scripts/train.py  (run_training_pipeline)
       │
       ├── N × ThorEnv (parallel slots)
       │
       └── AgentPolicy(DinoV2 + target_embedding + PolicyLSTM)
                  │
                  │  per slot, per step
                  │
                  │  agent.act(image, aux_features, hidden)
                  │──────────────────────► PolicyLSTM.forward()
                  │◄────── (action_idx, log_prob, value, hidden)
                  │
                  │  env.step(action_idx)
                  │──────────────────────► ThorEnv
                  │◄────── (obs, reward, terminated, truncated, info)
                  │
                  │  agent.store(image, aux, action, log_prob, reward, value, done)
                  │
                  │  [all slots done → build rollouts list]
                  │
                  │  agent.update_from_rollouts(rollouts)
                  │    GAE per rollout
                  │    normalize advantages
                  │    PPO epochs → Adam step
                  │
                  └── log to WandB, save checkpoint, optional HF push
```

---

## Components

### `src/simulation/thor_env.py`

Gymnasium-style wrapper around AI2-THOR with dynamic resolution.

**Actions**: `["MoveAhead", "RotateRight", "RotateLeft", "SENSE", "DONE"]`
(configurable via `action_set: full | minimal | navigation`)

**Observation**: RGB frame `(3, H, W)` float in `[0, 1]`, degraded based on current resolution level.

**Aux features** (`get_aux_features()`):
```
GPS            2   normalized agent position (x, z)
compass        4   sin/cos of yaw and horizon
prev_action    n   one-hot of last action
res_level      1   current_downgrade / base_downgrade
budget         1   remaining_budget / max_budget
target_idx     1   index into target vocabulary
```

**Episode state** (reset each `reset()`):
```
_step_count              int    steps taken this episode
_current_downgrade       int    resolution level [0=full … base_downgrade=worst]
_remaining_sensing_budget int   SENSE uses left
_last_sense_was_valid    bool   was the last SENSE in-budget and not at full res?
_closest_distance        float  best distance to target seen so far
```

#### Resolution mechanism

```
base_downgrade = floor(log2(min(H, W)))   e.g. 7 for 224×224

_current_downgrade starts at base_downgrade  (worst: single-pixel blocks)

_compute_obs():
    k = _current_downgrade
    avg_pool2d(kernel=2^k, stride=2^k)   ← downsample
    interpolate(size=(H,W), nearest)      ← stretch back

SENSE action:
    _last_sense_was_valid = (downgrade > 0 AND budget > 0)
    obs returned at OLD downgrade   ← improvement not yet visible this step
    _current_downgrade -= 1         ← visible from next step
    _remaining_sensing_budget -= 1

Move action:
    _current_downgrade = base_downgrade   ← reset to blurry immediately
```

Timeline: `reset → blurry(7) → SENSE → blurry(7) → SENSE → clearer(6) → MoveAhead → blurry(7) → …`

#### Reward structure

| Situation | Reward |
|-----------|--------|
| Got closer to target | `+distance_scale × Δdist` |
| Valid SENSE | `−sense_penalty` |
| Invalid SENSE (budget=0 or already full res) | `−sense_penalty − oversensing_penalty` |
| DONE + success (target visible, within distance) | `+success_reward` |
| DONE + fail | `−wrong_done_penalty × distance_ratio` |
| Move blocked (wall / invalid) | `−bump_penalty` |
| Normal move / rotate | `−step_penalty` |
| Episode truncated (max_steps) | `−timeout_penalty` |
| First time target is visible | `+first_visibility_bonus` |

---

### `src/models/agent_policy.py` + `src/models/lstm.py`

**`AgentPolicy`** (outer wrapper):
- Frozen DinoV2 visual encoder (`dinov2_vitb14` → 768-dim CLS token by default)
- Learnable target embedding table (semantic structure across object types)
- Concatenates `[vis_features | aux_features | target_embedding]` → passes to PolicyLSTM

**`PolicyLSTM`** (recurrent core):
```
Input: [vis_features (vis_dim) | gps (2) | compass (4) | prev_action (n_actions) |
         res_level (1) | budget (1) | target_embedding (target_embed_dim)]
       → LSTM(hidden_dim, lstm_layers)
       → last timestep
       → policy_head  → logits (n_actions)
       → value_head   → scalar
```

`forward(obs, hidden=None)` accepts:
- `(B, input_dim)` — single step
- `(B, T, input_dim)` — sequence, uses last timestep output

Returns `(logits, value, hidden)`.

---

### `src/agents/ppo_agent.py`

**`PPOAgent`** — standard PPO with GAE and multi-rollout batching.

| Method | What it does |
|--------|-------------|
| `act(image, aux, hidden)` | No-grad forward, sample action → `(action_idx, log_prob, value, hidden)` |
| `store(image, aux, action, log_prob, reward, value, done)` | Append transition to buffer |
| `update_from_rollouts(rollouts)` | GAE → normalize → PPO epochs → Adam step |

**PPO update per epoch**:
```
1. forward(obs, hidden=initial_per_rollout)  ← LSTM state preserved within rollout
2. clipped surrogate loss  min(ratio×A, clip(ratio)×A)
3. value MSE (optionally clipped)
4. entropy bonus  −entropy_coef × H(π)
5. total = policy + value_coef×value − entropy_coef×entropy
6. zero_grad → backward → clip_grad_norm(0.5) → Adam step
```

**Baselines** (`src/agents/baselines/`):
- `HighResBaseline`: `fixed_high_res=True`, SENSE masked out
- `LowResBaseline`: always worst resolution, SENSE masked out
- `RandomSensingBaseline`: SENSE with fixed `sense_prob`
- `FixedScheduleBaseline`: SENSE every `sense_interval` steps (if budget remains)

---

## Training loop (`scripts/train.py`)

```python
cfg = yaml.safe_load(open("cfgs/train_rl.yaml"))

# Build N parallel environment slots
envs = [ThorEnv(**cfg["env"]) for _ in range(num_parallel_envs)]
policy = AgentPolicy(**cfg["model"], **cfg["visual_encoder"])
agent = PPOAgent(policy, **cfg["agent"])

for episode in range(num_episodes):
    # Reset all slots
    for slot in slots:
        obs, aux = env.reset(scene, episode)
        slot.hidden = None

    # Collect rollout from each slot
    while any_slot_active:
        for slot in active_slots:
            action, log_prob, value, hidden = agent.act(obs, aux, hidden)
            next_obs, reward, term, trunc, info = env.step(action)
            agent.store(obs, aux, action, log_prob, reward, value, done)

    # Single batched PPO update across all slots
    rollouts = [build_rollout(slot) for slot in slots]
    losses = agent.update_from_rollouts(rollouts)

    wandb.log({...})
    if checkpoint_due:
        torch.save(checkpoint, path)
```

---

## Config reference (`cfgs/train_rl.yaml`)

```yaml
training:
  scenes: [FloorPlan1..6]     # AI2-THOR floor plans to cycle through
  num_episodes: 10000
  num_parallel_envs: 4        # concurrent rollout slots per PPO update
  checkpoint_every: 250
  output_dir: output
  checkpoint_dir: checkpoints
  resume_from: null           # path to .pt to resume, or null
  target:
    mode: random              # random | fixed | cycle
    candidates: [Mug, Bowl]   # object vocabulary

env:
  base_resolution: [224, 224]
  max_steps: 100
  max_sensing_budget: 1000000
  fixed_high_res: false       # disable resolution downgrade entirely
  action_set: full            # full | minimal | navigation
  move_magnitude: 0.25
  rotate_degrees: 45.0
  success_distance: 1.5
  reward_cfg:
    step_penalty: 0.01
    sense_penalty: 0.01
    oversensing_penalty: 0.0
    bump_penalty: 0.05
    timeout_penalty: 0.5
    wrong_done_penalty: 2.0
    success_reward: 10.0
    distance_scale: 0.5

agent:
  lr: 1.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.1
  value_coef: 0.5
  entropy_coef: 0.05
  epochs: 8

model:
  hidden_dim: 512
  lstm_layers: 1

visual_encoder:
  model_name: dinov2_vitb14   # vits14 (384) | vitb14 (768) | vitl14 (1024) | vitg14 (1536)

wandb:
  enabled: true
  project: ai2thor-rl
  entity: r0kasb-epfl
  run_name: exp2

trajectory_logging:
  enabled: false              # per-step CSV + scene layout JSON

huggingface:
  repo_id: R0kasB/resaction-nav
  push: false
  push_every: 100
```
