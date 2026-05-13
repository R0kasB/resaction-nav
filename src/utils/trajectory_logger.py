"""Per-step trajectory logger. Writes a CSV of step metrics and JSON layouts per scene."""

from __future__ import annotations

import csv
import json
from pathlib import Path


# Columns always present regardless of config
_BASE_COLS = ["episode", "step", "scene", "target"]

# Optional column groups and their CSV header names
_OPT_GROUPS: dict[str, list[str]] = {
    "position":       ["x", "y", "z"],
    "rotation":       ["rotation_y", "camera_horizon"],
    "action":         ["action"],
    "downgrade":      ["downgrade"],
    "sensing_budget": ["sensing_budget"],
    "reward":         ["reward"],
}


class TrajectoryLogger:
    """
    Usage:
        logger = TrajectoryLogger.from_cfg(output_dir, cfg)
        logger.save_layout_once(scene, env)   # call after env.reset()
        logger.log_step(episode, step, scene, target, env, action_name, reward, info)
        logger.close()
    """

    def __init__(self, output_dir: str, fields: dict[str, bool]):
        self._out = Path(output_dir)
        self._layouts_dir = self._out / "layouts"
        self._layouts_dir.mkdir(parents=True, exist_ok=True)
        self._saved_layouts: set[str] = set()

        cols = list(_BASE_COLS)
        for group, group_cols in _OPT_GROUPS.items():
            if fields.get(group, True):
                cols.extend(group_cols)
        self._fields = fields
        self._cols = cols

        self._file = open(self._out / "trajectories.csv", "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._cols, extrasaction="ignore")
        self._writer.writeheader()

    @classmethod
    def from_cfg(cls, output_dir: str, traj_cfg: dict) -> "TrajectoryLogger | None":
        """Return a logger if enabled in config, else None."""
        if not traj_cfg.get("enabled", False):
            return None
        fields = traj_cfg.get("fields", {})
        return cls(output_dir, fields)

    def save_layout_once(self, scene: str, env) -> None:
        """Save reachable positions for a scene. No-op if already saved."""
        if scene in self._saved_layouts:
            return
        layout_path = self._layouts_dir / f"{scene}.json"
        if layout_path.exists():
            self._saved_layouts.add(scene)
            return
        event = env.controller.step(action="GetReachablePositions")
        positions = event.metadata.get("actionReturn", [])
        layout_path.write_text(
            json.dumps({"scene": scene, "reachable_positions": positions}, indent=2),
            encoding="utf-8",
        )
        self._saved_layouts.add(scene)

    def log_step(
        self,
        episode: int,
        step: int,
        scene: str,
        target: str,
        env,
        action_name: str,
        reward: float,
        info: dict,
    ) -> None:
        agent_meta = env.current_event.metadata["agent"]
        pos = agent_meta["position"]
        rot = agent_meta["rotation"]

        row: dict = {
            "episode": episode,
            "step":    step,
            "scene":   scene,
            "target":  target,
        }

        if self._fields.get("position", True):
            row.update({"x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4)})
        if self._fields.get("rotation", True):
            row.update({
                "rotation_y":     round(rot["y"], 2),
                "camera_horizon": round(agent_meta.get("cameraHorizon", 0.0), 2),
            })
        if self._fields.get("action", True):
            row["action"] = action_name
        if self._fields.get("downgrade", True):
            row["downgrade"] = info["downgrade"]
        if self._fields.get("sensing_budget", True):
            row["sensing_budget"] = info["sensing_budget"]
        if self._fields.get("reward", True):
            row["reward"] = round(reward, 5)

        self._writer.writerow(row)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
