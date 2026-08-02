.PHONY: help setup setup-mac doctor lint test validate data run figures tables paper clean freeze

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## install package + dev + llm extras and pre-commit hooks (Linux/GPU box)
	pip install -e ".[dev,llm]" && pre-commit install

setup-mac:  ## install without the llm extra (macOS, Days 1-5)
	pip install -e ".[dev]" && pre-commit install

doctor:  ## check the environment is sane before starting a day
	python scripts/doctor.py

lint:  ## ruff + black --check + mypy
	ruff check src tests scripts && black --check src tests scripts && mypy src

test:  ## unit tests (fast)
	pytest -m "not slow and not gpu and not data"

validate:  ## GATE 1: simulator physics vs closed-form CIR
	python scripts/validate_channel_physics.py

data:  ## download every registered dataset into data/raw
	python scripts/download_data.py --all

run:  ## run one experiment: make run CFG=configs/experiment/e1_attack_success.yaml
	python scripts/run_experiment.py --config $(CFG)

figures:  ## regenerate all paper figures from results/runs
	python scripts/make_figures.py

tables:  ## regenerate all paper tables from results/runs
	python scripts/make_tables.py

paper: figures tables  ## rebuild figures, tables, then the PDF
	cd paper && latexmk -pdf main.tex

freeze:  ## pin the exact environment for the artifact release
	pip freeze > requirements.lock.txt

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
