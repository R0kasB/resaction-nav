# resaction-nav

## Overview

This project explores **dynamic resolution for efficient visual navigation** in AI2-THOR.

---

## Structure

```
resaction-nav/
├── cfgs/                 # Config files (training, env, logging)
├── scripts/              # Entry points (run, debug, experiments)
├── src/
│   ├── agents/           # RL agents (policies, algorithms)
│   ├── models/           # Neural networks (vision encoder, policy/value)
│   ├── simulation/       # AI2-THOR interface (camera, controller)
│   └── utils/            # Helper functions (e.g. image resolution)
├── pyproject.toml        # Dependencies (uv)
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
uv run python scripts/<your_script>.py
```

---

## Notes
