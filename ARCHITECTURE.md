# Codebase Architecture

OBSOLETE

## What this project does

A reinforcement learning agent that navigates AI2-THOR indoor environments to find a target object.
The core research idea is **dynamic resolution**: the agent always starts with a heavily downgraded
(blurry) camera view and can spend a limited budget of SENSE actions to sharpen it — at a cost —
before any movement resets the view back to blurry.

---

## File map

```
resaction-nav/
├── cfgs/
│   └── train_rl.yaml           ← single source of truth for all hyperparameters
├── scripts/
│   └── train.py                ← cluster entry point, loads YAML, runs training loop
├── src/
│   ├── agents/
│   │   └── ppo_agent.py        ← PPOAgent: act / store / update
│   ├── models/
│   │   └── lstm.py             ← PolicyLSTM: recurrent policy + value head
│   ├── simulation/
│   │   ├── thor_env.py         ← ThorEnv: Gymnasium-style AI2-THOR wrapper  ← main env
│   │   └── thor_camera.py      ← standalone camera utility (not used in training)
│   └── utils/
│       └── image_resolution.py ← degrade_resolution() used by thor_camera only
└── notebooks/
    ├── controller_test.ipynb
    └── test_codebase.ipynb     ← function-by-function tests
```

---

## Data flow

```
cfgs/train_rl.yaml
       │  yaml.safe_load()
       ▼
scripts/train.py
       │
       ├── builds ThorEnv  ──────────────────────────────────────────────────┐
       │                                                                      │
       └── builds PPOAgent(PolicyLSTM)                                        │
                  │                                                           │
                  │  each step                                                │
                  │                                                           │
                  │  agent.act(obs, hidden)                                   │
                  │──────────────────────────► PolicyLSTM.forward()          │
                  │◄────── (action_idx, log_prob, value, hidden) ────────────│
                  │                                                           │
                  │  env.step(action_idx) ◄───────────────────────────────── │
                  │                       ──► (obs, reward, term, trunc, info)
                  │                                                           │
                  │  agent.store(obs, action, log_prob, reward, value, done)  │
                  │                                                           │
                  │  [episode end]                                            │
                  │                                                           │
                  │  agent.update()                                           │
                  │    reads self.buffer                                      │
                  │    computes returns + advantages                          │
                  │    PPO epochs → Adam step                                 │
                  │    clears buffer                                          │
                  │    returns loss dict                                      │
                  │                                                           │
                  └── log to WandB, save checkpoint, push to HuggingFace ────┘
```

---

## Training loop (scripts/train.py)

```python
cfg = yaml.safe_load(open("cfgs/train_rl.yaml"))

env   = ThorEnv(**cfg["env"])
model = PolicyLSTM(**cfg["model"])
agent = PPOAgent(model, **cfg["agent"])

for episode in range(num_episodes):
    obs    = env.reset(scene)          # (3, H, W) float tensor
    hidden = None                      # LSTM state, reset each episode

    while not done:
        action, log_prob, value, hidden = agent.act(obs, hidden)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.store(obs, action, log_prob, reward, value, done)

    losses = agent.update()            # PPO weight update, clears buffer
    wandb.log({...})
    if checkpoint_due:
        torch.save(...)
```

---

## Module reference

### `src/simulation/thor_env.py`

**`RewardConfig`** — frozen dataclass. All reward shaping scalars in one place.
Passed to `ThorEnv.__init__` as `reward_cfg=RewardConfig(...)`.

**`ThorEnv`** — the environment.

| Method | What it does |
|--------|-------------|
| `reset(scene, target_obj_type)` | Loads scene, random-spawns objects, picks target, returns first obs |
| `step(action_idx)` | Runs one action. Returns `(obs, reward, terminated, truncated, info)` |
| `close()` | Stops the AI2-THOR controller |
| `update_seed(seed)` | Changes seed for the next `reset()` |
| `_compute_obs()` | Grabs current frame, applies resolution downgrading → `(3, H, W)` float in `[0,1]` |
| `_define_target()` | Picks a random pickupable object as the episode target |
| `_get_min_distance_to_object(obj_type)` | Euclidean distance: agent → nearest instance of obj_type |
| `_get_distance_to_position(obj_pos)` | Euclidean distance: agent → specific position dict |
| `_compute_reward(truncated)` | Progress shaping + action penalty + success/fail bonus |
| `_fail_checker()` | `True` when step_count >= max_steps |
| `_check_success()` | `True` when any target instance is visible and within success_distance |

**episode state (reset each `reset()`):**
```
_step_count              int   steps taken this episode
_current_action          str   last action name
_current_downgrade       int   resolution level [0=full … base_downgrade=worst]
_remaining_sensing_budget int  SENSE uses left
_last_sense_was_valid    bool  was the last SENSE within budget & not at full res?
_closest_distance        float best distance to target seen so far
_done                    bool  episode over?
```

---

### Resolution mechanism

```
base_downgrade = floor(log2(min(H, W)))   e.g. 7 for 224×224

_current_downgrade starts at base_downgrade  (worst: 128×128 pixel blocks)

_compute_obs():
    k = _current_downgrade
    avg_pool2d(kernel=2^k, stride=2^k)   ← downsample
    interpolate(size=(H,W), nearest)      ← stretch back to original size

SENSE action:
    1. flag  _last_sense_was_valid = (downgrade > 0 AND budget > 0)
    2. obs   computed at OLD downgrade            ← improvement not yet visible
    3. then  _current_downgrade -= 1              ← visible from NEXT step
             _remaining_sensing_budget -= 1

Move action:
    _current_downgrade = base_downgrade           ← immediate reset (new position)
```

Visual: start blurry, sense to sharpen, move to blur again.

```
episode:  reset → blurry(7) → SENSE → blurry(7) → SENSE → clearer(6) → MoveAhead → blurry(7) → …
                                     obs=blurry(7)       obs=clearer(6)
```

---

### Reward structure

| Situation | Reward |
|-----------|--------|
| Got closer to target | `+distance_scale × Δdist` |
| Valid SENSE | `−sense_penalty` |
| Invalid SENSE (budget=0 or already full res) | `−sense_penalty − oversensing_penalty` |
| DONE + success | `+success_reward` |
| DONE + fail | `−fail_penalty` |
| Move blocked (wall / invalid) | `−bump_penalty` |
| Normal move/rotate/look | `−step_penalty` |
| Episode truncated (max_steps) | `−fail_penalty` |

---

### `src/models/lstm.py`

**`PolicyLSTM`** — recurrent policy + value network.

Input vector (flat, concatenated at each step):
```
vis_features   vis_dim      raw frame CNN / DinoV2 CLS token  (TODO: encoder)
GPS            3            agent position x, y, z
compass        2            yaw/360, camera_horizon/360
action         n_actions    one-hot of previous action
res_level      1            current_downgrade / base_downgrade  (0=full, 1=worst)
budget         1            remaining_budget / max_budget
─────────────────────────────────────────────
total          vis_dim + 7 + n_actions
```

Architecture:
```
input → LSTM(hidden_dim, lstm_layers) → last_timestep → ┬→ Linear → action_logits (n_actions)
                                                         └→ Linear → value (1)
```

`forward(obs, hidden=None)` accepts:
- `(B, input_dim)` — single step (unsqueezed internally to `(B, 1, input_dim)`)
- `(B, T, input_dim)` — sequence, uses last timestep output

Returns `(logits, value, hidden)`.

---

### `src/agents/ppo_agent.py`

**`PPOAgent`** — owns its rollout buffer. Standard `store() / update()` interface.

| Method | What it does |
|--------|-------------|
| `act(obs, hidden)` | No-grad forward pass, samples action → `(action_idx, log_prob, value, hidden)` |
| `store(obs, action, log_prob, reward, value, done)` | Appends transition to `self.buffer` |
| `update()` | Computes returns + advantages, runs PPO epochs, clears buffer → loss dict |
| `_compute_returns(rewards, dones, last_value)` | Reverse-accumulated discounted returns |

PPO update (per epoch):
```
1. forward(obs, hidden=None)          ← LSTM hidden reset per epoch (BPTT is TODO)
2. clipped surrogate loss             ← min(ratio*A, clip(ratio)*A)
3. value MSE loss                     ← (V - returns)²
4. entropy bonus                      ← −entropy_coef × H(π)
5. total = policy + value_coef*value − entropy_coef*entropy
6. zero_grad → backward → clip_grad_norm(0.5) → Adam step
```

---

## Config reference (`cfgs/train_rl.yaml`)

```yaml
training:
  scenes: [...]           # which AI2-THOR floor plans to cycle through
  num_episodes: 500       # total training episodes
  print_every: 10         # console log frequency
  checkpoint_every: 50    # save .pt every N episodes
  output_dir: output      # training_log.txt goes here
  checkpoint_dir: checkpoints
  resume_from: null       # path to .pt to resume, or null

env:                      # passed directly to ThorEnv(**cfg["env"])
  base_resolution: [224, 224]
  max_steps: 200
  max_sensing_budget: 5
  seed: 42
  reward_cfg: { ... }     # all RewardConfig fields

agent:                    # passed to PPOAgent(**cfg["agent"])
  lr: 3.0e-4
  gamma: 0.99
  clip_eps: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  epochs: 4

model:                    # passed to PolicyLSTM(**cfg["model"])
  hidden_dim: 512
  lstm_layers: 1

wandb:
  enabled: true / false
  project: ...
  entity: ...
  run_name: ...

huggingface:
  repo_id: ...
  push: true / false
  push_every: 100
```

---

## Current TODOs

| What | Where | Priority |
|------|-------|----------|
| Replace raw frame with DinoV2 encoder | `thor_env._compute_obs()` | High |
| Add GPS + compass + action + metadata to obs | `thor_env._compute_obs()` | High |
| Full BPTT (store initial hidden per episode) | `ppo_agent.update()` | Medium |
| Wire `cfgs/train_rl.yaml` entity/repo_id to real values | `cfgs/train_rl.yaml` | Before first run |
