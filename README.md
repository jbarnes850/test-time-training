# Surprisal-Guided Selection

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2602.07670-red)](http://arxiv.org/abs/2602.07670)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Jarrodbarnes/KernelBench-RLVR-120b)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

> Compute-Optimal Test-Time Strategies for Execution-Grounded Code Generation

This repository contains the code for reproducing the experiments in our paper on compute-optimal test-time strategies for verifiable execution-grounded (VEG) tasks. We study GPU kernel optimization using KernelBench as our testbed and a 120B-parameter model (GPT-OSS-120B with LoRA adaptation).

![Environment](artifacts/fig_environment.png)

## Key Findings

| Finding | Result | Evidence |
|---------|--------|----------|
| **Search outperforms minimal adaptation** | Best-of-N K=64: 90% task success (18/20 L1 tasks); TTT BoA: 30.6% (3-seed mean) | TTT equivalent K < 1, worse than single-sample inference |
| **Surprisal-guided selection** | 80% success vs 50% for confidence-guided | Highest-surprisal correct samples yield best kernels at zero cost |
| **Over-sharpening mechanism** | TTT collapses diversity toward mediocre solutions | Gradient updates concentrate probability on early successes, missing the distribution tail |

*fast_1: fraction of samples that are both correct and achieve speedup > 1x over the reference implementation.*

![Adaptation Trajectory](artifacts/fig2_trajectory.png)

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/jbarnes850/test-time-training.git
cd test-time-training

# Install dependencies
uv sync --extra dev

# Verify installation
uv run python -m scripts.dataset_smoke
uv run python -m scripts.kernelbench_smoke
uv run python -m pytest -q
```

## Reproducing Paper Results

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (16GB+ VRAM)
- `TINKER_API_KEY` for training (contact [Thinking Machines Lab](https://tinker-docs.thinkingmachines.ai/))

```bash
# Set environment variables
export TINKER_API_KEY=your_key_here
export HF_TOKEN=your_token_here  # Optional, for dataset access
```

### Best-of-N Baseline

```bash
uv run python -m scripts.best_of_n \
  --split splits/l1_seed42.json \
  --subset eval \
  --k 64 \
  --max_tasks 5
```

### Batch TTT with BoA Selection

```bash
uv run python -m scripts.batch_ttt \
  --split splits/l1_seed42.json \
  --subset eval \
  --k 32 \
  --steps 5
```

### SDPO Self-Distillation

```bash
# Prompt-only (recommended)
uv run python -m scripts.sdpo_train \
  --split splits/l1_seed42.json \
  --k 32 \
  --steps 1 \
  --prompt_only

# With execution feedback
uv run python -m scripts.sdpo_train \
  --split splits/l1_seed42.json \
  --k 32 \
  --steps 1
```

## Project Structure

```
kernelbench-rl-env/
├── src/                    # Core environment and utilities
│   ├── env/                # Evaluator, telemetry, tasking
│   ├── utils/              # Dataset, code, checkpoint utilities
│   ├── tinker_env.py       # Tinker RL environment
│   └── metrics.py          # Evaluation metrics
├── scripts/                # Entry points
│   ├── best_of_n.py        # Best-of-N baseline
│   ├── batch_ttt.py        # Batch TTT with BoA
│   ├── sdpo_train.py       # SDPO self-distillation
│   ├── eval_task.py        # Single task evaluation
│   ├── train_smoke.py      # RLVR training
│   └── make_split.py       # Create train/eval splits
├── splits/                 # Deterministic data splits
├── artifacts/              # Paper figures and generation scripts
├── docs/                   # Paper source (paper.md) and runbook
├── tests/                  # Test suite
└── vendor/                 # Submodules
    ├── KernelBench/        # Evaluation framework
    └── tinker-cookbook/     # Training library
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TINKER_API_KEY` | Yes | - | API access for Tinker training infrastructure |
| `HF_TOKEN` | No | - | HuggingFace token for dataset access |
| `KERNELBENCH_EVAL_MODE` | No | `full` | `fast` (5 trials) or `full` (50 trials) |

## Hardware Requirements

- **Evaluation**: CUDA 11.8+, 16GB+ VRAM
- **Training**: Requires Tinker API (cloud infrastructure)
- **Model inference**: 8x A100 80GB for full 120B model

## Using the Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Jarrodbarnes/KernelBench-RLVR-120b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Jarrodbarnes/KernelBench-RLVR-120b")
```

## Citation

```bibtex
@article{barnes2026surprisal,
  title={Surprisal-Guided Selection: Compute-Optimal Test-Time Strategies for Execution-Grounded Code Generation},
  author={Barnes, Jarrod},
  journal={arXiv preprint arXiv:2602.07670},
  year={2026},
  url={http://arxiv.org/abs/2602.07670}
}
```

## Acknowledgments

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (Ouyang et al., 2025)
- [Tinker](https://tinker-docs.thinkingmachines.ai/) (Thinking Machines Lab)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
