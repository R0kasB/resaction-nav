.PHONY: test test-e2e test-fast

test:
	uv run pytest -q tests -rs

test-e2e:
	uv run pytest -q tests/test_pipeline_end_to_end.py -rs

test-fast:
	uv run pytest -q tests/test_thor_env_unit.py tests/test_ppo_agent_unit.py tests/test_train_script_unit.py -rs
