# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core environment, metrics, and utilities (e.g., `src/env/`, `src/utils/`, `src/tinker_env.py`).
- `scripts/`: runnable entry points for smoke tests, dataset splits, training, evaluation, and plotting.
- `tests/`: pytest suite (`test_*.py`, shared fixtures in `conftest.py`).
- `splits/`: deterministic train/eval splits; `artifacts/` and `runs/` store manifests, plots, and telemetry outputs.
- `docs/`: runbook and memo notes.
- `vendor/KernelBench` and `vendor/tinker-cookbook`: vendored dependencies/submodules.

## Build, Test, and Development Commands
- `uv sync --extra dev`: create/refresh the virtual env with dev deps.
- `uv run python -m scripts.dataset_smoke`: validate KernelBench L1 data access.
- `uv run python -m scripts.make_split --level 1 --seed 42`: generate deterministic split JSON.
- `uv run python -m scripts.kernelbench_smoke`: sanity check KernelBench integration.
- `uv run python -m pytest -q`: run the test suite (quiet mode).
- `uv run python -m scripts.train_smoke --max_batches 3`: RLVR smoke training (requires GPU + `TINKER_API_KEY`).

## Coding Style & Naming Conventions
- Python 3.11+, PEP 8 with 4-space indentation.
- Use type hints and dataclasses where helpful (see `src/`).
- Modules, functions, and variables use `snake_case`; classes use `CapWords`.
- No formatter/linter is configured in this repo; keep changes consistent with surrounding code.

## Testing Guidelines
- Pytest configuration lives in `pyproject.toml` (`testpaths = ["tests"]`).
- New tests should live in `tests/` and be named `test_*.py`.
- Prefer small, deterministic unit tests; use script-based smoke tests for end-to-end checks.

## Commit & Pull Request Guidelines
- PRs should include a short summary, key commands run (with outputs or notes), and any required credentials or hardware (GPU, `TINKER_API_KEY`).

## Environment & Secrets
- Required: `TINKER_API_KEY` (set in `.env` or shell).
- Optional: `HF_TOKEN` for dataset access, `WANDB_API_KEY` for logging.
