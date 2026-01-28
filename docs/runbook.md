# Runbook: KernelBench RL Environment

## Prereqs

- Python 3.11+
- CUDA-capable GPU for KernelBench evaluation
- `TINKER_API_KEY` in `.env` or exported in shell
- Optional: `HF_TOKEN` for dataset access, `WANDB_API_KEY` for logging

## Setup

```bash
uv sync --extra dev
```

## Data + manifest

```bash
uv run python -m scripts.l1_smoke
uv run python -m scripts.make_split --level 1 --seed 42
uv run python -m scripts.write_manifest --split splits/l1_seed42.json
uv run python -m scripts.print_manifest
```

## Environment sanity

```bash
uv run python -m scripts.kb_smoke
uv run python -m pytest -q
```

## RLVR smoke training

```bash
uv run python -m scripts.train_smoke --max_batches 3
```

## Fast-proxy training (recommended for scale)

Use a fast-proxy evaluator during training (fewer trials) while keeping full KernelBench evaluation for checkpoints.

```bash
uv run python -m scripts.train_smoke \
  --eval_mode fast --num_correct_trials 1 --num_perf_trials 5 \
  --normalize_reward --reward_baseline_window 32 --correctness_bonus 0.01
```

After each checkpoint, run a full KernelBench eval on the same tasks:

```bash
uv run python -m scripts.best_of_n --eval_mode full --max_tasks 10 --k 64
```

## Test-time self-learning

```bash
uv run python -m scripts.best_of_n --max_tasks 1 --k 32
uv run python -m scripts.inner_loop_smoke --max_tasks 1 --k 32 --steps 1
uv run python -m scripts.compare_inner_loop
```

## Artifacts + memo

```bash
uv run python -m scripts.make_plots
uv run python -m scripts.write_memo
```

## One-command repro

```bash
uv run python -m scripts.repro_small
```
