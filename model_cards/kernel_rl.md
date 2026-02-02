---
license: apache-2.0
base_model: openai/gpt-oss-120b
tags:
  - gpu-kernel
  - cuda
  - code-generation
  - reinforcement-learning
  - grpo
  - kernelbench
datasets:
  - ScalingIntelligence/KernelBench
language:
  - en
pipeline_tag: text-generation
model-index:
  - name: KernelBench-RLVR-120b
    results:
      - task:
          type: text-generation
          name: GPU Kernel Generation
        dataset:
          name: KernelBench L1
          type: ScalingIntelligence/KernelBench
        metrics:
          - name: fast_1
            type: custom
            value: 30.6
          - name: correctness
            type: accuracy
            value: 91.5
---

# KernelBench-RLVR-120b

A 120B parameter LLM fine-tuned with GRPO (Group Relative Policy Optimization) for GPU kernel generation, evaluated on KernelBench L1.

## Quick Start

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

## Model Description

This model was trained using an execution-grounded RL framework where:

1. **Environment**: KernelBench provides deterministic execution feedback via CUDA compiler and GPU hardware
2. **Reward**: Raw speedup (correctness-gated) normalized by running baseline
3. **Algorithm**: GRPO with group-relative advantages
4. **Evaluation**: Same evaluator as training (no reward hacking possible)

| Parameter | Value |
|-----------|-------|
| Base Model | openai/gpt-oss-120b |
| Algorithm | GRPO (Group Relative Policy Optimization) |
| LoRA Rank | 16 |
| Training Steps | 40 |
| Learning Rate | 1e-5 |
| Temperature | 0.25 |
| Max Tokens | 1024 |
| Training Tasks | 80 (KernelBench L1 train split) |

## Evaluation Results

**Training Checkpoint (Step 40):**
- Correctness: 98.4%
- Mean Speedup: 0.87x on training distribution

**Test-Time Evaluation (3 seeds, 10 tasks):**

| Method | fast_1 | std | Correctness | Rollouts |
|--------|--------|-----|-------------|----------|
| **Batch-TTT BoA** | **30.6%** | 11.3% | 91.5% | 960 |
| **SDPO Prompt-Only** | **30.4%** | 7.6% | 91.9% | 320 |
| Best-of-N (K=64) | 30.9% | - | 87.2% | 320 |

**Note:** fast_1 = fraction of samples that are both correct AND achieve speedup > 1x.

## Key Findings

This model was developed as part of research on test-time training efficiency for verifiable execution-grounded tasks. Three key findings emerged:

1. **Efficiency Frontier**: Performance peaks after 1-2 adaptation steps then regresses, indicating that checkpoint selection (Best-of-Adaptation) rather than extended training drives the benefit.

2. **Feedback Redundancy**: Rich tokenized execution feedback provides no lift over prompt-only self-distillation (+4.2% advantage for prompt-only across 3 seeds). When the world provides dense continuous rewards, teacher-based interpretation becomes redundant.

3. **Regime Dependence**: Adaptation helps when base policy achieves >30% coverage, but hurts performance below that threshold. Search maintains a diversity advantage in hard regimes.

## Hardware Requirements

- **GPU Memory**: ~240GB for bf16 inference (e.g., 8x A100 40GB, 4x A100 80GB, or 3x H100)
- **Disk Space**: ~240GB for model weights
- **Recommended**: Use `device_map="auto"` for automatic multi-GPU distribution

For single-GPU inference, consider using quantization:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "Jarrodbarnes/KernelBench-RLVR-120b",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Intended Use

This model is designed for GPU kernel optimization research. Given a PyTorch reference implementation, it generates optimized CUDA kernel code.

**Input format:**
```
Given the following PyTorch reference implementation:

```python
[reference code]
```

Write an optimized CUDA kernel that computes the same result.
```

## Limitations

- Evaluated on KernelBench L1 only (250 ML workloads)
- Hardware-specific optimizations (A100)
- Extended test-time adaptation may cause regression (use BoA selection with early stopping)
- Single model size evaluated (120B)

## Citation

```bibtex
@article{barnes2026ttt,
  title={When Does Test-Time Training Saturate? Evidence from Verifiable Execution-Grounded Tasks},
  author={Barnes, Jarrod},
  journal={arXiv preprint},
  year={2026}
}
```

## Related Work

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) - Ouyang et al., 2025
- [TTT-Discover](https://arxiv.org/abs/2601.16175) - Yuksekgonul et al., 2026
- [SDPO](https://arxiv.org/abs/2601.20802) - Zeng et al., 2026
- [Scalable Power Sampling](https://arxiv.org/abs/2601.21590) - Ji et al., 2026
