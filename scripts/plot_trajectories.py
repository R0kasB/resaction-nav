"""
Plot trajectory data produced by TrajectoryLogger.

Usage:
    uv run python scripts/plot_trajectories.py --output-dir output/test_logging/adaptive

Outputs (saved to <output-dir>/plots/):
    topdown.png   — top-down agent paths overlaid on reachable positions
    reward.png    — per-step reward lines + total reward per episode
    actions.png   — action frequency bar chart
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def load_csv(output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / "trajectories.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"trajectories.csv not found in {output_dir}")
    return pd.read_csv(csv_path)


def load_layout(output_dir: Path, scene: str) -> list[dict] | None:
    path = output_dir / "layouts" / f"{scene}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["reachable_positions"]


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_topdown(df: pd.DataFrame, output_dir: Path, plots_dir: Path) -> None:
    if "x" not in df.columns or "z" not in df.columns:
        print("No position columns (x, z) — skipping top-down plot.")
        return

    scenes = df["scene"].unique()
    fig, axes = plt.subplots(1, len(scenes), figsize=(6 * len(scenes), 6), squeeze=False)

    for ax, scene in zip(axes[0], scenes):
        positions = load_layout(output_dir, scene)
        if positions:
            rx = [p["x"] for p in positions]
            rz = [p["z"] for p in positions]
            ax.scatter(rx, rz, s=4, c="#cccccc", zorder=1, label="reachable")

        episodes = df[df["scene"] == scene]["episode"].unique()
        palette = cm.tab10(np.linspace(0, 1, max(len(episodes), 1)))

        for i, ep in enumerate(episodes):
            ep_df = df[(df["scene"] == scene) & (df["episode"] == ep)].sort_values("step")
            x, z = ep_df["x"].values, ep_df["z"].values
            color = palette[i % len(palette)]
            ax.plot(x, z, "-", color=color, alpha=0.75, linewidth=1.2, zorder=2, label=f"ep {ep}")
            ax.scatter(x[0],  z[0],  marker="o", color=color, s=50, zorder=3)  # start
            ax.scatter(x[-1], z[-1], marker="X", color=color, s=50, zorder=3)  # end

        ax.set_title(scene)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
        if len(episodes) <= 10:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Agent Trajectories (top-down view)\ncircle = start   ✕ = end", fontsize=11)
    fig.tight_layout()
    _save(fig, plots_dir / "topdown.png")


def plot_reward(df: pd.DataFrame, plots_dir: Path) -> None:
    if "reward" not in df.columns:
        print("No reward column — skipping reward plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # left: per-step reward curves + mean
    for ep, ep_df in df.groupby("episode"):
        ep_df = ep_df.sort_values("step")
        ax1.plot(ep_df["step"], ep_df["reward"],
                 alpha=0.25, linewidth=0.9, color="steelblue")

    mean_by_step = df.groupby("step")["reward"].mean()
    ax1.plot(mean_by_step.index, mean_by_step.values,
             color="darkblue", linewidth=2, label="mean")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title("Per-step Reward")
    ax1.set_xlabel("step")
    ax1.set_ylabel("reward")
    ax1.legend()

    # right: total reward per episode
    ep_totals = df.groupby("episode")["reward"].sum()
    ax2.bar(ep_totals.index, ep_totals.values, color="steelblue", edgecolor="white")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_title("Total Reward per Episode")
    ax2.set_xlabel("episode")
    ax2.set_ylabel("total reward")
    ax2.set_xticks(ep_totals.index)

    fig.tight_layout()
    _save(fig, plots_dir / "reward.png")


def plot_actions(df: pd.DataFrame, plots_dir: Path) -> None:
    if "action" not in df.columns:
        print("No action column — skipping action plot.")
        return

    counts = df["action"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Action Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, plots_dir / "actions.png")


def plot_sensing_budget(df: pd.DataFrame, plots_dir: Path) -> None:
    if "sensing_budget" not in df.columns:
        print("No sensing_budget column — skipping sensing-budget plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for ep, ep_df in df.groupby("episode"):
        ep_df = ep_df.sort_values("step")
        ax.plot(ep_df["step"], ep_df["sensing_budget"],
                alpha=0.5, linewidth=1.0, drawstyle="steps-post")

    ax.set_title("Sensing Budget over Steps (per episode)")
    ax.set_xlabel("step")
    ax.set_ylabel("remaining budget")
    ax.set_yticks(range(int(df["sensing_budget"].max()) + 1))
    fig.tight_layout()
    _save(fig, plots_dir / "sensing_budget.png")


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TrajectoryLogger output.")
    parser.add_argument(
        "--output-dir", required=True,
        help="Agent output directory containing trajectories.csv and layouts/",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(output_dir)
    print(
        f"Loaded {len(df):,} rows | "
        f"{df['episode'].nunique()} episodes | "
        f"{df['scene'].nunique()} scene(s): {list(df['scene'].unique())}"
    )

    plot_topdown(df, output_dir, plots_dir)
    plot_reward(df, plots_dir)
    plot_actions(df, plots_dir)
    plot_sensing_budget(df, plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
