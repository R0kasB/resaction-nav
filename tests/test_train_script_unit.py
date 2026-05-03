import importlib.util
from pathlib import Path

import torch
import yaml


def _import_train_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
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
