import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
import yaml


def _import_train_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != "ai2thor":
            raise
        ai2thor = types.ModuleType("ai2thor")
        controller_mod = types.ModuleType("ai2thor.controller")

        class _DummyController:
            def __init__(self, *args, **kwargs):
                pass

        controller_mod.Controller = _DummyController
        ai2thor.controller = controller_mod
        sys.modules["ai2thor"] = ai2thor
        sys.modules["ai2thor.controller"] = controller_mod
        spec.loader.exec_module(module)
    return module


def test_load_cfg_reads_yaml(tmp_path):
    train_module = _import_train_module()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_data = {"training": {"num_episodes": 3}, "env": {"max_steps": 5}}
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    loaded = train_module.load_cfg(str(cfg_path))
    assert loaded == cfg_data


def test_save_checkpoint_writes_expected_keys(tmp_path):
    train_module = _import_train_module()

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt.pt"

    train_module.save_checkpoint(model, optimizer, 7, str(ckpt_path))
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    assert payload["episode"] == 7
    assert "policy" in payload
    assert "optimizer" in payload


def test_output_data_creates_metrics_log(tmp_path):
    train_module = _import_train_module()
    out_path = tmp_path / "training_log.txt"

    train_module.output_data(
        rewards=[1.0, 2.0, 3.0],
        successes=[True, False, True],
        episode_lengths=[10, 12, 14],
        run_type="ppo_dynamic_resolution",
        params={"lr": 3e-4},
        filename=str(out_path),
    )

    content = out_path.read_text(encoding="utf-8")
    assert "Type: ppo_dynamic_resolution" in content
    assert "Episodes: 3" in content
    assert "Mean Reward" in content


def test_resolve_target_setup_fixed_mode_sets_env_candidates():
    train_module = _import_train_module()
    env_cfg = {"target_object_types": None}
    fixed, cycle = train_module._resolve_target_setup(
        training_cfg={
            "target": {
                "mode": "fixed",
                "object_type": "Mug",
                "candidates": ["Mug", "Apple"],
            }
        },
        env_cfg=env_cfg,
    )
    assert fixed == "Mug"
    assert cycle is None
    assert env_cfg["target_object_types"] == ["Mug", "Apple"]


def test_resolve_target_setup_cycle_requires_non_empty_cycle():
    train_module = _import_train_module()
    with pytest.raises(ValueError, match="non-empty list"):
        train_module._resolve_target_setup(
            training_cfg={
                "target": {
                    "mode": "cycle",
                    "cycle": [],
                }
            },
            env_cfg={},
        )


def test_run_agent_executes_parallel_env_batches():
    train_module = _import_train_module()

    class _FakeEnv:
        def __init__(self, name):
            self.name = name
            self.action_list = ["DONE"]
            self.reset_calls = []
            self._step = 0

        def reset(self, scene, target_obj_type=None):
            self._step = 0
            self.reset_calls.append((scene, target_obj_type))
            return torch.zeros(3, 8, 8)

        def get_aux_features(self, prev_action_idx=None):
            return torch.zeros(4)

        def step(self, action_idx):
            self._step += 1
            return (
                torch.zeros(3, 8, 8),
                1.0,
                True,
                False,
                {
                    "step": self._step,
                    "downgrade": 0,
                    "sensing_budget": 0,
                    "success": True,
                },
            )

    class _FakeAgent:
        def __init__(self):
            self.policy = torch.nn.Linear(1, 1)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
            self.store_calls = 0
            self.update_calls = 0

        def act(self, image, aux_features, hidden=None):
            return 0, torch.tensor(0.0), torch.tensor(0.0), None

        def store_initial_hidden(self, hidden):
            return None

        def store(self, **kwargs):
            self.store_calls += 1

        def update(self, next_image, next_aux_features, final_hidden=None):
            self.update_calls += 1
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
            }

    env0 = _FakeEnv("env0")
    env1 = _FakeEnv("env1")
    agent = _FakeAgent()
    cfg = {
        "training": {
            "scenes": ["FloorPlan1", "FloorPlan2"],
            "num_episodes": 4,
            "num_parallel_envs": 2,
            "print_every": 100,
            "checkpoint_every": 100,
            "checkpoint_dir": "unused",
        },
        "wandb": {"enabled": False},
        "huggingface": {"push": False},
    }

    rewards, episode_lengths, successes = train_module.run_agent(
        envs=[env0, env1],
        agent=agent,
        cfg=cfg,
        device=torch.device("cpu"),
        target_object_cycle=["Mug", "Apple"],
    )

    assert rewards == [1.0, 1.0, 1.0, 1.0]
    assert episode_lengths == [1, 1, 1, 1]
    assert successes == [True, True, True, True]
    assert env0.reset_calls == [("FloorPlan1", "Mug"), ("FloorPlan1", "Mug")]
    assert env1.reset_calls == [("FloorPlan2", "Apple"), ("FloorPlan2", "Apple")]
    assert agent.store_calls == 4
    assert agent.update_calls == 2


def test_build_cluster_run_id_prefers_explicit_value(monkeypatch):
    train_module = _import_train_module()
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_PROCID", "2")

    run_id = train_module._build_cluster_run_id(
        {"run_id": "manual-id", "auto_cluster_run_id": True}
    )
    assert run_id == "manual-id"


def test_apply_run_id_to_training_dirs_uses_slurm_env(monkeypatch):
    train_module = _import_train_module()
    monkeypatch.setenv("SLURM_JOB_ID", "4321")
    monkeypatch.setenv("SLURM_PROCID", "7")

    training_cfg = {
        "output_dir": "output",
        "checkpoint_dir": "checkpoints",
        "auto_cluster_run_id": True,
    }
    run_id = train_module._apply_run_id_to_training_dirs(training_cfg)

    assert run_id == "job-4321-rank-7"
    assert training_cfg["output_dir"].endswith("output/job-4321-rank-7")
    assert training_cfg["checkpoint_dir"].endswith("checkpoints/job-4321-rank-7")
