# KernelBench RL Environment

Self-learning RL environment for kernel optimization on KernelBench L1, with GRPO/RLVR training and test-time LoRA adaptation.

## Quickstart

```bash
# Create venv + install deps
uv sync --extra dev

# Smoke: print KernelBench L1 row count
uv run python -m scripts.l1_smoke

# Create deterministic split
uv run python -m scripts.make_split --level 1 --seed 42

# Write and print manifest
uv run python -m scripts.write_manifest --split splits/l1_seed42.json
uv run python -m scripts.print_manifest

# KernelBench submodule check
uv run python -m scripts.kb_smoke

# Tests
uv run python -m pytest -q
```

For a full runbook, see `docs/runbook.md`.

## Single-sample run

```bash
# Dry run (no GPU required)
uv run python -m scripts.run_one \
  --problem_id 100 \
  --kernel_path path/to/kernel.py \
  --run_name demo \
  --dry_run

# Evaluate run telemetry
uv run python -m scripts.eval_run --run demo
```

## RLVR smoke training (requires Tinker API key + GPU for KernelBench eval)

```bash
export TINKER_API_KEY=...  # set in .env or shell
uv run python -m scripts.train_smoke --max_batches 3
```

## Test-time adaptation + baseline

```bash
# Best-of-N baseline
uv run python -m scripts.best_of_n --max_tasks 1 --k 32

# Inner-loop LoRA adaptation
uv run python -m scripts.inner_loop_smoke --max_tasks 1 --k 32 --steps 1

# Compare results
uv run python -m scripts.compare_inner_loop
```

## Plots + memo

```bash
uv run python -m scripts.make_plots
uv run python -m scripts.write_memo
```

## One-command repro (small budget)

```bash
uv run python -m scripts.repro_small
```

## Repository layout

- `scripts/`: dataset utilities, evaluation, telemetry, training
- `splits/`: deterministic train/eval splits
- `runs/`: run outputs and telemetry
- `artifacts/`: manifests and plots
- `docs/`: memo and notes
- `vendor/KernelBench`: KernelBench submodule
- `vendor/tinker-cookbook`: Tinker cookbook submodule
