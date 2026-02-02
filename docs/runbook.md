# Runbook: KernelBench RL Environment

## Prerequisites

- Python 3.11+
- CUDA-capable GPU for KernelBench evaluation
- `TINKER_API_KEY` in `.env` or exported in shell
- Optional: `HF_TOKEN` for dataset access

## Setup

```bash
uv sync --extra dev
```

## Verify Installation

```bash
# Check dataset access
uv run python -m scripts.dataset_smoke

# Check KernelBench integration
uv run python -m scripts.kernelbench_smoke

# Run test suite
uv run python -m pytest -q
```

## Create Data Splits

```bash
uv run python -m scripts.make_split --level 1 --seed 42
```

## RLVR Training

```bash
# Smoke test (3 batches)
uv run python -m scripts.train_smoke --max_batches 3

# Full training with fast-proxy evaluation
uv run python -m scripts.train_smoke \
  --eval_mode fast \
  --normalize_reward \
  --reward_baseline_window 32
```

## Test-Time Evaluation

```bash
# Best-of-N baseline
uv run python -m scripts.best_of_n --max_tasks 5 --k 64

# Batch TTT with BoA selection
uv run python -m scripts.batch_ttt --max_tasks 5 --k 32 --steps 5

# SDPO self-distillation
uv run python -m scripts.sdpo_train --max_tasks 5 --k 32 --steps 1
```
