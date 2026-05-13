.PHONY: test test-e2e test-fast

test:
	python -m pytest -q tests -rs

test-e2e:
	python -m pytest -q tests/test_pipeline_end_to_end.py -rs

test-fast:
	python -m pytest -q tests/test_thor_env_unit.py tests/test_ppo_agent_unit.py tests/test_train_script_unit.py -rs
