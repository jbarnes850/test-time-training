"""Batch Test-Time Training (TTRL-style) with Best-of-Adaptation (BoA) selection.

This script implements TTRL-style batch adaptation to fix the oscillation
problem observed in single-task TTT. Instead of adapting on 1 task at a time,
we update on N tasks per step with K samples each.

Key insight from TTRL (arXiv:2504.16084): Batch adaptation across multiple
problems reduces gradient variance and enables stable improvement.

Best-of-Adaptation (BoA): After running the adaptation trajectory, this script
automatically selects the checkpoint with the highest aggregate fast_1 score.
This makes BoA a concrete, executable algorithm rather than post-hoc analysis.

Outputs:
  - batch_ttt_summary.json: Full run summary with all step metrics
  - boa_selected.json: Selected checkpoint with BoA selection metadata
  - metrics.jsonl: Per-step metrics log
  - actions.jsonl: Per-sample action log

Single-task TTT (broken):
  Task 1: K=128 samples -> high variance gradient -> oscillation

Batch TTT (this script):
  Task 1: K=32 samples -|
  Task 2: K=32 samples -|-> joint gradient (low variance)
  Task N: K=32 samples -|
"""

import argparse
import json
import os
import random
import statistics
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

import tinker

from src.env.evaluator import evaluate_kernel
from src.env.tasking import load_task, build_messages
from src.metrics import fast_1
from src.utils.code_utils import extract_python_code, assemble_modelnew_code
from src.utils.env_utils import load_dotenv
from src.utils.checkpoint_utils import load_latest_checkpoint
from src.utils.path_utils import repo_root
from src.utils.tinker_utils import ensure_tinker_cookbook_on_path
from src.rlvr_utils import build_datums_from_group


# Baseline history for reward normalization (per problem_id)
_BASELINE_HISTORY: Dict[int, deque] = {}


def compute_normalized_reward(
    speedup: float,
    correctness: bool,
    problem_id: int,
    baseline_window: int = 32,
    correct_bonus: float = 0.0,
    epsilon: float = 1e-6,
) -> tuple[float, float]:
    """Compute normalized reward using running baseline median.

    Matches RLVR training: reward = max(0, speedup/baseline) + bonus

    Returns (reward, baseline_median) tuple.
    """
    # Get or create baseline history for this problem
    if problem_id not in _BASELINE_HISTORY:
        _BASELINE_HISTORY[problem_id] = deque(maxlen=baseline_window)

    history = _BASELINE_HISTORY[problem_id]
    baseline_median = float(statistics.median(history)) if history else 1.0

    if not correctness:
        return 0.0, baseline_median

    # Normalize speedup by baseline (matches RLVR)
    norm_speedup = speedup / max(baseline_median, epsilon)
    reward = max(0.0, norm_speedup) + correct_bonus

    # Update baseline history on correct samples only (matches RLVR)
    if correctness and speedup > 0:
        history.append(speedup)

    return reward, baseline_median

load_dotenv()
ensure_tinker_cookbook_on_path()

from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer


@dataclass
class TaskSamples:
    """Holds samples and evaluation results for a single task."""
    problem_id: int
    prompt: Any  # tinker.ModelInput
    sampled_tokens: List[List[int]] = field(default_factory=list)
    logprobs: List[List[float]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    correct: List[bool] = field(default_factory=list)
    speedups: List[float] = field(default_factory=list)


def _load_problem_ids(split_path: str, max_tasks: int | None, problem_ids: str | None, task_offset: int = 0) -> list[int]:
    """Load problem IDs from split file or explicit list."""
    if problem_ids:
        return [int(x.strip()) for x in problem_ids.split(",") if x.strip()]
    split = json.loads(Path(split_path).read_text())
    ids = list(split["problem_ids"]["eval"])
    if task_offset > 0:
        ids = ids[task_offset:]
    if max_tasks is not None:
        ids = ids[:max_tasks]
    return ids


def _sample_task(
    sampling_client,
    renderer,
    prompt,
    k: int,
    max_tokens: int,
    temperature: float,
) -> tuple[List[List[int]], List[List[float]], List[Any]]:
    """Sample K completions for a single task."""
    future = sampling_client.sample(
        prompt=prompt,
        num_samples=k,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            stop=renderer.get_stop_sequences(),
            temperature=temperature,
        ),
    )
    result = future.result()
    sampled_tokens = []
    logprobs = []
    messages = []
    for seq in result.sequences:
        sampled_tokens.append(seq.tokens)
        logprobs.append(seq.logprobs)
        msg, _ = renderer.parse_response(seq.tokens)
        messages.append(msg)
    return sampled_tokens, logprobs, messages


def _evaluate_task_samples(
    problem_id: int,
    ref_code: str,
    messages: List[Any],
    action_log_path: Path | None,
    step_idx: int,
    baseline_window: int = 32,
    correct_bonus: float = 0.0,
) -> tuple[List[float], List[bool], List[float]]:
    """Evaluate all samples for a single task with normalized rewards."""
    rewards = []
    correct = []
    speedups = []
    for idx, msg in enumerate(messages):
        raw_action = extract_python_code(get_text_content(msg))
        kernel_code = assemble_modelnew_code(raw_action, ref_code)
        if action_log_path:
            with action_log_path.open("a") as handle:
                handle.write(
                    json.dumps({
                        "problem_id": problem_id,
                        "step_idx": step_idx,
                        "sample_idx": idx,
                        "raw_action": raw_action,
                        "assembled_code": kernel_code,
                    }) + "\n"
                )
        result = evaluate_kernel(problem_id, kernel_code)
        # Use normalized reward with running baseline
        reward, _ = compute_normalized_reward(
            result.speedup,
            result.correctness,
            problem_id,
            baseline_window=baseline_window,
            correct_bonus=correct_bonus,
        )
        rewards.append(reward)
        correct.append(result.correctness)
        speedups.append(result.speedup)
    return rewards, correct, speedups


def compute_global_advantages(all_rewards: List[float]) -> List[float]:
    """Compute normalized advantages across ALL samples from ALL tasks.

    This is the key TTRL insight: by computing advantages globally,
    we reduce gradient variance compared to per-task normalization.
    """
    if len(all_rewards) == 0:
        return []
    mean_r = statistics.mean(all_rewards)
    std_r = statistics.stdev(all_rewards) if len(all_rewards) > 1 else 1.0
    std_r = max(std_r, 1e-6)  # Prevent division by zero
    return [(r - mean_r) / std_r for r in all_rewards]


def batch_ttt_step(
    task_samples_list: List[TaskSamples],
    training_client,
    learning_rate: float,
    shuffle_rewards: bool = False,
) -> None:
    """Perform a single TTRL-style gradient update across all tasks.

    Key difference from single-task TTT:
    - Advantages are computed across ALL samples from ALL tasks
    - Single gradient update on the diverse batch
    - Reduces variance, enables stable improvement
    """
    # Collect all rewards globally
    all_rewards = []
    for ts in task_samples_list:
        all_rewards.extend(ts.rewards)

    if shuffle_rewards:
        random.shuffle(all_rewards)

    # Compute global advantages (TTRL-style)
    all_advantages = compute_global_advantages(all_rewards)

    # Build datums for each task, using global advantages
    all_datums = []
    adv_offset = 0
    for ts in task_samples_list:
        n_samples = len(ts.rewards)
        task_advantages = all_advantages[adv_offset:adv_offset + n_samples]
        adv_offset += n_samples

        datums = build_datums_from_group(
            ts.prompt,
            ts.sampled_tokens,
            ts.logprobs,
            task_advantages,
        )
        all_datums.extend(datums)

    # Single gradient update on the combined batch
    fwd_bwd = training_client.forward_backward(all_datums, loss_fn="importance_sampling")
    optim = training_client.optim_step(tinker.AdamParams(learning_rate=learning_rate))
    _ = fwd_bwd.result()
    _ = optim.result()


def log_step_metrics(
    metrics_log_path: Path,
    step_idx: int,
    task_samples_list: List[TaskSamples],
) -> Dict[str, Any]:
    """Log per-task and aggregate metrics for a step."""
    per_task_fast1 = {}
    per_task_correct = {}
    all_correct = []
    all_speedups = []
    all_rewards = []

    for ts in task_samples_list:
        task_fast1 = fast_1(ts.correct, ts.speedups)
        task_correct_rate = sum(1 for c in ts.correct if c) / len(ts.correct) if ts.correct else 0.0
        per_task_fast1[ts.problem_id] = task_fast1
        per_task_correct[ts.problem_id] = task_correct_rate
        all_correct.extend(ts.correct)
        all_speedups.extend(ts.speedups)
        all_rewards.extend(ts.rewards)

    agg_fast1 = fast_1(all_correct, all_speedups)
    agg_correct = sum(1 for c in all_correct if c) / len(all_correct) if all_correct else 0.0
    agg_reward = statistics.mean(all_rewards) if all_rewards else 0.0
    correct_speedups = [s for s, c in zip(all_speedups, all_correct) if c]
    agg_speedup = statistics.mean(correct_speedups) if correct_speedups else 0.0

    # Log aggregate metrics
    agg_record = {
        "step_idx": step_idx,
        "aggregate_fast_1": agg_fast1,
        "aggregate_correct": agg_correct,
        "aggregate_speedup": agg_speedup,
        "aggregate_reward": agg_reward,
        "per_task_fast_1": per_task_fast1,
        "per_task_correct": per_task_correct,
        "n_samples": len(all_rewards),
    }
    with metrics_log_path.open("a") as handle:
        handle.write(json.dumps(agg_record) + "\n")

    return agg_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch TTT (TTRL-style) for kernel optimization")
    parser.add_argument("--split", type=str, default="splits/l1_seed42.json")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--renderer_name", type=str, default="gpt_oss_no_sysprompt")
    parser.add_argument("--checkpoint_jsonl", type=str, default="runs/rlvr_normalized_optimized_prompt/checkpoints.jsonl")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of tasks per gradient step")
    parser.add_argument("--k", type=int, default=32, help="Samples per task per step")
    parser.add_argument("--steps", type=int, default=8, help="Number of TTT steps")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_tasks", type=int, default=5, help="Limit to first N eval tasks")
    parser.add_argument("--task_offset", type=int, default=0, help="Skip first N eval tasks")
    parser.add_argument("--problem_ids", type=str, default="", help="Explicit comma-separated problem IDs")
    parser.add_argument("--log_path", type=str, default="runs/batch_ttt")
    parser.add_argument("--eval_mode", type=str, default="fast", choices=["full", "fast"])
    parser.add_argument("--num_correct_trials", type=int, default=1)
    parser.add_argument("--num_perf_trials", type=int, default=5)
    parser.add_argument("--baseline_window", type=int, default=32, help="Window size for reward normalization")
    parser.add_argument("--correct_bonus", type=float, default=0.0, help="Bonus reward for correct samples")
    parser.add_argument("--shuffle_rewards", action="store_true", help="Shuffle rewards across samples before GRPO (noise control)")
    args = parser.parse_args()

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set.")

    os.environ["KERNELBENCH_EVAL_MODE"] = args.eval_mode
    if args.num_correct_trials is not None:
        os.environ["KERNELBENCH_NUM_CORRECT_TRIALS"] = str(args.num_correct_trials)
    if args.num_perf_trials is not None:
        os.environ["KERNELBENCH_NUM_PERF_TRIALS"] = str(args.num_perf_trials)

    root = repo_root()
    log_dir = root / args.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    action_log_path = log_dir / "actions.jsonl"
    metrics_log_path = log_dir / "metrics.jsonl"
    summary_path = log_dir / "batch_ttt_summary.json"
    compute_path = log_dir / "batch_ttt_compute.json"

    problem_ids = _load_problem_ids(args.split, args.max_tasks, args.problem_ids or None, args.task_offset)

    # Limit to batch_size tasks
    if len(problem_ids) > args.batch_size:
        problem_ids = problem_ids[:args.batch_size]

    print(f"Batch TTT with {len(problem_ids)} tasks: {problem_ids}")
    print(f"K={args.k} samples/task, {args.steps} steps")
    print(f"Total rollouts per step: {len(problem_ids) * args.k}")
    print(f"Eval mode: {args.eval_mode} (correct_trials={args.num_correct_trials}, perf_trials={args.num_perf_trials})")
    print(f"Reward normalization: baseline_window={args.baseline_window}, correct_bonus={args.correct_bonus}")

    compute = {
        "total_rollouts": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "wall_clock_s": 0.0,
    }
    run_start = time.time()

    renderer_name = args.renderer_name or get_recommended_renderer_name(args.model)
    tokenizer = get_tokenizer(args.model)
    renderer = get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient()

    # Load checkpoint from RLVR training
    base_sampler_path = ""
    state_path = ""
    ckpt = load_latest_checkpoint(args.checkpoint_jsonl)
    if ckpt:
        base_sampler_path = ckpt.get("sampler_path", "")
        state_path = ckpt.get("state_path", "")
        print(f"Loaded checkpoint: sampler={base_sampler_path}")

    # Initialize training client
    if state_path:
        training_client = service_client.create_training_client_from_state(state_path)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=args.model, rank=args.lora_rank
        )

    if base_sampler_path:
        sampling_path = base_sampler_path
    else:
        sampling_path = training_client.save_weights_for_sampler(name="base").result().path

    sampling_client = service_client.create_sampling_client(model_path=sampling_path)

    # Preload tasks and build prompts
    tasks = []
    prompts = []
    for pid in problem_ids:
        task = load_task(pid)
        tasks.append(task)
        messages = build_messages(task)
        prompt = renderer.build_generation_prompt(messages)
        prompts.append(prompt)

    # Track best results per task (for oracle comparison)
    best_per_task: Dict[int, Dict[str, Any]] = {pid: {"fast_1": 0.0, "step": -1} for pid in problem_ids}
    step_summaries = []
    checkpoint_paths: Dict[int, str] = {0: sampling_path}  # Track checkpoint path per step

    # Step 0: Baseline evaluation
    print("Step 0: Baseline evaluation...")
    task_samples_list = []
    for pid, task, prompt in zip(problem_ids, tasks, prompts):
        tokens, lps, messages = _sample_task(
            sampling_client, renderer, prompt, args.k, args.max_tokens, args.temperature
        )
        compute["total_rollouts"] += len(tokens)
        compute["total_prompt_tokens"] += prompt.length * len(tokens)
        compute["total_completion_tokens"] += sum(len(t) for t in tokens)
        rewards, correct, speedups = _evaluate_task_samples(
            pid, task.reference_code, messages, action_log_path, step_idx=0,
            baseline_window=args.baseline_window, correct_bonus=args.correct_bonus
        )
        ts = TaskSamples(
            problem_id=pid,
            prompt=prompt,
            sampled_tokens=tokens,
            logprobs=lps,
            rewards=rewards,
            correct=correct,
            speedups=speedups,
        )
        task_samples_list.append(ts)

        # Track best
        task_f1 = fast_1(correct, speedups)
        if task_f1 > best_per_task[pid]["fast_1"]:
            best_per_task[pid] = {"fast_1": task_f1, "step": 0}

    step_metrics = log_step_metrics(metrics_log_path, 0, task_samples_list)
    step_metrics["checkpoint_path"] = checkpoint_paths[0]
    step_summaries.append(step_metrics)
    print(f"  Step 0: agg_fast_1={step_metrics['aggregate_fast_1']:.3f}, "
          f"per_task={step_metrics['per_task_fast_1']}")

    # TTT steps
    for step in range(args.steps):
        print(f"Step {step + 1}: Gradient update + re-sample...")

        # Gradient update using current samples
        batch_ttt_step(task_samples_list, training_client, args.learning_rate, args.shuffle_rewards)

        # Save new weights and create new sampling client
        sampling_path = training_client.save_weights_for_sampler(name=f"step_{step + 1}").result().path
        checkpoint_paths[step + 1] = sampling_path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        # Re-sample all tasks with updated model
        task_samples_list = []
        for pid, task, prompt in zip(problem_ids, tasks, prompts):
            tokens, lps, messages = _sample_task(
                sampling_client, renderer, prompt, args.k, args.max_tokens, args.temperature
            )
            compute["total_rollouts"] += len(tokens)
            compute["total_prompt_tokens"] += prompt.length * len(tokens)
            compute["total_completion_tokens"] += sum(len(t) for t in tokens)
            rewards, correct, speedups = _evaluate_task_samples(
                pid, task.reference_code, messages, action_log_path, step_idx=step + 1,
                baseline_window=args.baseline_window, correct_bonus=args.correct_bonus
            )
            ts = TaskSamples(
                problem_id=pid,
                prompt=prompt,
                sampled_tokens=tokens,
                logprobs=lps,
                rewards=rewards,
                correct=correct,
                speedups=speedups,
            )
            task_samples_list.append(ts)

            # Track best
            task_f1 = fast_1(correct, speedups)
            if task_f1 > best_per_task[pid]["fast_1"]:
                best_per_task[pid] = {"fast_1": task_f1, "step": step + 1}

        step_metrics = log_step_metrics(metrics_log_path, step + 1, task_samples_list)
        step_metrics["checkpoint_path"] = checkpoint_paths[step + 1]
        step_summaries.append(step_metrics)
        print(f"  Step {step + 1}: agg_fast_1={step_metrics['aggregate_fast_1']:.3f}, "
              f"per_task={step_metrics['per_task_fast_1']}")

    # Compute oracle (best across all steps)
    oracle_fast1_per_task = {pid: info["fast_1"] for pid, info in best_per_task.items()}
    oracle_agg_fast1 = statistics.mean(oracle_fast1_per_task.values()) if oracle_fast1_per_task else 0.0

    # BoA Selection: select checkpoint with highest aggregate fast_1
    boa_selected_step = max(range(len(step_summaries)), key=lambda i: step_summaries[i]["aggregate_fast_1"])
    boa_selected_metrics = step_summaries[boa_selected_step]
    boa_selected = {
        "algorithm": "Best-of-Adaptation (BoA)",
        "selection_criterion": "argmax(aggregate_fast_1)",
        "selected_step": boa_selected_step,
        "selected_checkpoint": checkpoint_paths[boa_selected_step],
        "selected_metrics": {
            "aggregate_fast_1": boa_selected_metrics["aggregate_fast_1"],
            "aggregate_correct": boa_selected_metrics["aggregate_correct"],
            "per_task_fast_1": boa_selected_metrics["per_task_fast_1"],
        },
        "all_steps_fast_1": [s["aggregate_fast_1"] for s in step_summaries],
        "config": {
            "model": args.model,
            "renderer_name": renderer_name,
            "tasks": problem_ids,
            "k": args.k,
            "steps": args.steps,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "learning_rate": args.learning_rate,
            "base_checkpoint": args.checkpoint_jsonl,
        },
    }
    boa_selected_path = log_dir / "boa_selected.json"
    boa_selected_path.write_text(json.dumps(boa_selected, indent=2))
    print(f"\nBoA Selection: step {boa_selected_step} with fast_1={boa_selected_metrics['aggregate_fast_1']:.3f}")
    print(f"  Checkpoint: {checkpoint_paths[boa_selected_step]}")
    print(f"  Wrote BoA selection to {boa_selected_path}")

    # Final summary
    summary = {
        "config": {
            "model": args.model,
            "renderer_name": renderer_name,
            "batch_size": len(problem_ids),
            "k": args.k,
            "steps": args.steps,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "lora_rank": args.lora_rank,
            "learning_rate": args.learning_rate,
            "checkpoint": args.checkpoint_jsonl,
        },
        "problem_ids": problem_ids,
        "step_summaries": step_summaries,
        "checkpoint_paths": {str(k): v for k, v in checkpoint_paths.items()},
        "boa_selected": {
            "step": boa_selected_step,
            "checkpoint": checkpoint_paths[boa_selected_step],
            "fast_1": boa_selected_metrics["aggregate_fast_1"],
        },
        "oracle": {
            "per_task_fast_1": oracle_fast1_per_task,
            "aggregate_fast_1": oracle_agg_fast1,
            "best_steps": {pid: info["step"] for pid, info in best_per_task.items()},
        },
        "final_checkpoint": sampling_path,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    compute["wall_clock_s"] = time.time() - run_start
    compute_path.write_text(json.dumps(compute, indent=2))
    print(f"\nWrote summary to {summary_path}")
    print(f"Wrote compute stats to {compute_path}")
    print(f"Final step fast_1: {step_summaries[-1]['aggregate_fast_1']:.3f}")
    print(f"Oracle fast_1: {oracle_agg_fast1:.3f}")
    print(f"BoA selected fast_1: {boa_selected_metrics['aggregate_fast_1']:.3f} (step {boa_selected_step})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
