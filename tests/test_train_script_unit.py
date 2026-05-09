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
