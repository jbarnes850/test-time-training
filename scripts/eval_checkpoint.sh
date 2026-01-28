#!/bin/bash
# Evaluation script for step-10 checkpoint
# Usage: bash scripts/eval_checkpoint.sh [fast|full]

set -e

EVAL_MODE="${1:-fast}"
CHECKPOINT_STEP="${2:-10}"

# Using previous successful run (medium_reasoning, high correctness)
RUN_DIR="runs/rlvr_prod_120b_ep4_medium_scaffold_fs5"
CHECKPOINT_PATH="${RUN_DIR}/checkpoints.jsonl"

echo "Using checkpoint from: ${RUN_DIR}"
echo "Config: renderer=gpt_oss_medium_reasoning, temp=0.25, max_tokens=1024"

echo "=== Evaluation Configuration ==="
echo "Mode: ${EVAL_MODE}"
echo "Checkpoint: step-${CHECKPOINT_STEP}"
echo "Run directory: ${RUN_DIR}"
echo ""

# Set environment based on eval mode
if [ "$EVAL_MODE" = "fast" ]; then
    export KERNELBENCH_EVAL_MODE=fast
    export KERNELBENCH_NUM_CORRECT_TRIALS=1
    export KERNELBENCH_NUM_PERF_TRIALS=5
    echo "Fast eval: 1 correct trial, 5 perf trials"
else
    export KERNELBENCH_EVAL_MODE=full
    export KERNELBENCH_NUM_CORRECT_TRIALS=5
    export KERNELBENCH_NUM_PERF_TRIALS=50
    echo "Full eval: 5 correct trials, 50 perf trials"
fi
echo ""

# Phase 1: Best-of-N evaluation (K=64)
echo "=== Phase 1: Best-of-N (K=64) ==="
BEST_OF_N_DIR="runs/best_of_n_step${CHECKPOINT_STEP}_k64_${EVAL_MODE}"

PYTHONPATH=/home/ubuntu/kernelbench-rl-env ./.venv/bin/python scripts/best_of_n.py \
    --model openai/gpt-oss-120b \
    --renderer_name gpt_oss_medium_reasoning \
    --checkpoint_jsonl "${CHECKPOINT_PATH}" \
    --k 64 \
    --split splits/l1_seed42.json \
    --log_path "${BEST_OF_N_DIR}" \
    --temperature 0.25 \
    --max_tokens 1024

echo "Best-of-N results: ${BEST_OF_N_DIR}/best_of_n_summary.json"
echo ""

# Phase 2: Inner-Loop TTT evaluation (K=64, steps=15)
echo "=== Phase 2: Inner-Loop TTT (K=64, steps=15) ==="
INNER_LOOP_DIR="runs/inner_loop_step${CHECKPOINT_STEP}_k64_${EVAL_MODE}"

PYTHONPATH=/home/ubuntu/kernelbench-rl-env ./.venv/bin/python scripts/inner_loop_smoke.py \
    --model openai/gpt-oss-120b \
    --renderer_name gpt_oss_medium_reasoning \
    --checkpoint_jsonl "${CHECKPOINT_PATH}" \
    --k 64 \
    --steps 15 \
    --split splits/l1_seed42.json \
    --log_path "${INNER_LOOP_DIR}" \
    --temperature 0.25 \
    --max_tokens 1024 \
    --lora_rank 16

echo "Inner-loop results: ${INNER_LOOP_DIR}/inner_loop_summary.json"
echo ""

# Phase 3: Comparison
echo "=== Phase 3: Generate Comparison ==="
COMPARE_PATH="runs/compare_step${CHECKPOINT_STEP}_${EVAL_MODE}.json"

PYTHONPATH=/home/ubuntu/kernelbench-rl-env ./.venv/bin/python scripts/compare_inner_loop.py \
    --inner_loop_path "${INNER_LOOP_DIR}/inner_loop_summary.json" \
    --best_of_n_path "${BEST_OF_N_DIR}/best_of_n_summary.json" \
    --output_path "${COMPARE_PATH}"

echo ""
echo "=== Evaluation Complete ==="
echo "Comparison saved to: ${COMPARE_PATH}"
echo ""
echo "To view results:"
echo "  cat ${COMPARE_PATH} | jq ."
