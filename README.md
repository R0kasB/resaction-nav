# resaction-nav

## Overview

This project explores **dynamic resolution for efficient visual navigation** in AI2-THOR.
The project details and explaination can be found in `webpage.html`

---

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
sbatch scripts/run_izar_smoke.sbatch
```

---
