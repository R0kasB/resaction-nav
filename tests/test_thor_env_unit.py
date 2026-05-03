import importlib
import sys
import types

import numpy as np
import pytest
import torch


def _import_thor_env():
    try:
        module = importlib.import_module("src.simulation.thor_env")
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
        module = importlib.import_module("src.simulation.thor_env")
    return module.ThorEnv


def _fake_event(frame, *, last_action_success=True):
    return types.SimpleNamespace(
        frame=frame,
        metadata={
            "lastActionSuccess": last_action_success,
            "agent": {
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "rotation": {"y": 90.0},
                "cameraHorizon": 45.0,
            },
            "objects": [],
        },
    )


def test_compute_obs_requires_reset_before_use():
    ThorEnv = _import_thor_env()
    env = ThorEnv.__new__(ThorEnv)
    env.current_event = None
    with pytest.raises(RuntimeError, match="Call reset"):
        env._compute_obs()


def test_compute_obs_rejects_non_rgb_frames():
    ThorEnv = _import_thor_env()
    env = ThorEnv.__new__(ThorEnv)
    env._current_downgrade = 0
    env.current_event = _fake_event(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected RGB frame"):
        env._compute_obs()


def test_compute_obs_clamps_downgrade_and_normalizes():
    ThorEnv = _import_thor_env()
    env = ThorEnv.__new__(ThorEnv)
    env._current_downgrade = 99
    # negative-stride slice to ensure np.ascontiguousarray path is exercised
    frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)[:, ::-1, :]
    env.current_event = _fake_event(frame)

    obs = env._compute_obs()
    assert tuple(obs.shape) == (3, 8, 8)
    assert obs.dtype == torch.float32
    assert float(obs.min()) >= 0.0
    assert float(obs.max()) <= 1.0


def test_get_aux_features_shape_and_action_encoding():
    ThorEnv = _import_thor_env()
    env = ThorEnv.__new__(ThorEnv)
    env.device = torch.device("cpu")
    env.action_list = ["MoveAhead", "RotateRight", "SENSE", "DONE"]
    env.base_downgrade = 7
    env._current_downgrade = 3
    env.max_sensing_budget = 5
    env._remaining_sensing_budget = 2
    env.current_event = _fake_event(np.zeros((8, 8, 3), dtype=np.uint8))

    feat = env.get_aux_features(prev_action_idx=2)
    assert tuple(feat.shape) == (3 + 2 + len(env.action_list) + 1 + 1,)
    # previous action one-hot starts at index 5 (gps + compass)
    assert feat[5 + 2].item() == pytest.approx(1.0)


def test_step_applies_sense_budget_after_reward():
    ThorEnv = _import_thor_env()
    env = ThorEnv.__new__(ThorEnv)
    env.action_list = ["MoveAhead", "SENSE", "DONE"]
    env.action2id = {a: i for i, a in enumerate(env.action_list)}
    env.action_params = {"MoveAhead": {"moveMagnitude": 0.25}}
    env._step_count = 0
    env._current_action = "MoveAhead"
    env.base_downgrade = 7
    env._current_downgrade = 2
    env.max_sensing_budget = 2
    env._remaining_sensing_budget = 1
    env._last_sense_was_valid = False
    env._done = False
    env.controller = types.SimpleNamespace(step=lambda **kwargs: env.current_event)
    env.current_event = _fake_event(np.zeros((8, 8, 3), dtype=np.uint8))
    env._fail_checker = lambda: False
    env._compute_obs = lambda: torch.zeros(3, 8, 8)
    env._compute_reward = lambda truncated: 0.123
    env._check_success = lambda: False

    _, reward, terminated, truncated, info = env.step(env.action2id["SENSE"])
    assert reward == pytest.approx(0.123)
    assert terminated is False
    assert truncated is False
    assert info["downgrade"] == 1
    assert info["sensing_budget"] == 0
