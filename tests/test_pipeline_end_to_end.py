import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def _import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _import_train_module():
    try:
        return _import_module_from_path(
            Path(__file__).resolve().parents[1] / "scripts" / "train.py",
            "train_script_for_pipeline_test",
        )
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
        return _import_module_from_path(
            Path(__file__).resolve().parents[1] / "scripts" / "train.py",
            "train_script_for_pipeline_test",
        )


class _FakeEnv:
    reset_targets = []

    def __init__(self, *args, **kwargs):
        self.action_list = ["MoveAhead", "SENSE", "DONE"]
        self.action2id = {name: idx for idx, name in enumerate(self.action_list)}
        self._step = 0
        self.target_object_embed_dim = kwargs.get("target_object_embed_dim", 8)

    def reset(self, scene, target_obj_type=None):
        self._step = 0
        _FakeEnv.reset_targets.append(target_obj_type)
        return torch.rand(3, 8, 8)

    def get_aux_features(self, prev_action_idx=None):
        return torch.zeros(3 + 2 + len(self.action_list) + 1 + 1 + self.target_object_embed_dim)

    def step(self, action_idx):
        self._step += 1
        obs = torch.rand(3, 8, 8)
        reward = 1.0 if action_idx == self.action2id["DONE"] else 0.1
        terminated = action_idx == self.action2id["DONE"] or self._step >= 3
        truncated = False
        info = {
            "step": self._step,
            "downgrade": 0,
            "sensing_budget": 1,
            "success": bool(action_idx == self.action2id["DONE"]),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        return None


class _FakePolicy(nn.Module):
    last_target_object_embed_dim = None

    def __init__(self, n_actions: int, target_object_embed_dim: int = 8, **kwargs):
        super().__init__()
        self.n_actions = n_actions
        _FakePolicy.last_target_object_embed_dim = target_object_embed_dim
        self.backbone = nn.Sequential(
            nn.Linear(3 * 8 * 8 + (3 + 2 + n_actions + 1 + 1 + target_object_embed_dim), 32),
            nn.Tanh(),
        )
        self.pi = nn.Linear(32, n_actions)
        self.v = nn.Linear(32, 1)

    def forward(self, image, aux_features, hidden=None):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if aux_features.dim() == 1:
            aux_features = aux_features.unsqueeze(0)

        x = torch.cat([image.flatten(start_dim=1), aux_features], dim=1)
        h = self.backbone(x)
        return self.pi(h), self.v(h), hidden


def test_run_pipeline_end_to_end_with_smoke_mode(tmp_path, monkeypatch):
    train_module = _import_train_module()
    monkeypatch.setattr(train_module, "ThorEnv", _FakeEnv)
    monkeypatch.setattr(train_module, "AgentPolicy", _FakePolicy)
    _FakeEnv.reset_targets = []
    _FakePolicy.last_target_object_embed_dim = None

    run_pipeline_module = _import_module_from_path(
        Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py",
        "run_pipeline_script_for_pipeline_test",
    )
    monkeypatch.setattr(run_pipeline_module, "_load_train_module", lambda: train_module)

    cfg = {
        "training": {
            "scenes": ["FakeScene"],
            "num_episodes": 5,
            "print_every": 1,
            "checkpoint_every": 1,
            "output_dir": str(tmp_path / "output"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "resume_from": None,
            "target": {
                "mode": "fixed",
                "object_type": "Mug",
                "candidates": ["Mug", "Apple"],
            },
        },
        "env": {
            "base_resolution": [64, 64],
            "max_steps": 10,
            "target_object_embed_dim": 6,
            "reward_cfg": {},
        },
        "agent": {
            "lr": 3e-4,
            "gamma": 0.99,
            "clip_eps": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "epochs": 2,
        },
        "model": {
            "hidden_dim": 32,
            "lstm_layers": 1,
        },
        "visual_encoder": {
            "model_name": "dinov2_vitb14",
        },
        "wandb": {
            "enabled": False,
        },
        "huggingface": {
            "push": False,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    summary = run_pipeline_module.run_pipeline(cfg_path=str(cfg_path), smoke=True)

    assert summary["episodes"] == 2
    assert Path(summary["output_log"]).exists()
    assert list((tmp_path / "checkpoints" / "smoke").glob("*.pt"))
    assert _FakeEnv.reset_targets == ["Mug", "Mug"]
    assert _FakePolicy.last_target_object_embed_dim == 6


def test_apply_smoke_overrides_sets_small_runtime_defaults():
    run_pipeline_module = _import_module_from_path(
        Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py",
        "run_pipeline_script_for_overrides_test",
    )

    cfg = {
        "training": {"num_episodes": 50, "checkpoint_every": 5, "print_every": 10},
        "env": {"max_steps": 200, "base_resolution": [224, 224]},
        "wandb": {"enabled": True},
        "huggingface": {"push": True},
    }
    out = run_pipeline_module.apply_smoke_overrides(cfg)

    assert out["training"]["num_episodes"] == 2
    assert out["training"]["checkpoint_every"] == 1
    assert out["env"]["max_steps"] == 8
    assert out["env"]["base_resolution"] == [64, 64]
    assert out["wandb"]["enabled"] is False
    assert out["huggingface"]["push"] is False
