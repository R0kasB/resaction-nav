import pytest


def test_controller_observation_pipeline_smoke():
    pytest.importorskip("ai2thor")

    from src.simulation.thor_env import ThorEnv

    env = None
    try:
        env = ThorEnv(
            seed=0,
            device="cpu",
            base_resolution=(128, 128),
            max_steps=5,
        )
    except OSError as exc:
        pytest.skip(f"Skipping integration smoke (unable to initialize AI2-THOR): {exc}")

    try:
        obs0 = env.reset(scene="FloorPlan1")
        obs1, reward, terminated, truncated, _ = env.step(env.action2id["MoveAhead"])
    finally:
        env.close()

    assert obs0.ndim == 3
    assert obs1.ndim == 3
    assert obs0.shape == obs1.shape
    assert obs0.dtype == obs1.dtype
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
