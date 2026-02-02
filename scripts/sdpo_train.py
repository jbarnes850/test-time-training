"""SDPO-style batch test-time training with execution-grounded feedback.

This script mirrors scripts/batch_ttt.py but replaces scalar reward advantages with
SDPO-style token-level advantages derived from a self-teacher conditioned on
execution feedback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import tinker
import torch

from src.env.evaluator import evaluate_kernel
from src.env.tasking import load_task, build_messages, PromptConfig
from src.metrics import fast_1
from src.utils.code_utils import extract_python_code, assemble_modelnew_code
from src.utils.env_utils import load_dotenv
from src.utils.checkpoint_utils import load_latest_checkpoint
from src.utils.path_utils import repo_root
from src.utils.tinker_utils import ensure_tinker_cookbook_on_path
from src.utils.feedback_utils import (
    extract_error_info,
    build_execution_feedback,
    build_teacher_context,
)


load_dotenv()
ensure_tinker_cookbook_on_path()

from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer


_BASELINE_HISTORY: Dict[int, deque] = {}


def compute_normalized_reward(
    speedup: float,
    correctness: bool,
    problem_id: int,
    baseline_window: int = 32,
    correct_bonus: float = 0.0,
    epsilon: float = 1e-6,
) -> tuple[float, float]:
    if problem_id not in _BASELINE_HISTORY:
        _BASELINE_HISTORY[problem_id] = deque(maxlen=baseline_window)

    history = _BASELINE_HISTORY[problem_id]
    baseline_median = float(statistics.median(history)) if history else 1.0

    if not correctness:
        return 0.0, baseline_median

    norm_speedup = speedup / max(baseline_median, epsilon)
    reward = max(0.0, norm_speedup) + correct_bonus
    if correctness and speedup > 0:
        history.append(speedup)

    return reward, baseline_median


def compute_sdpo_advantages(
    sampled_logprobs: List[List[float]],
    teacher_logprobs: List[List[float | None]],
    sdpo_coef: float,
) -> List[List[float]]:
    advantages: List[List[float]] = []
    for s_lps, t_lps in zip(sampled_logprobs, teacher_logprobs):
        seq_adv: List[float] = []
        for s_lp, t_lp in zip(s_lps, t_lps):
            if t_lp is None:
                seq_adv.append(0.0)
            else:
                seq_adv.append(float(sdpo_coef * (t_lp - s_lp)))
        if len(seq_adv) < len(s_lps):
            seq_adv.extend([0.0] * (len(s_lps) - len(seq_adv)))
        advantages.append(seq_adv)
    return advantages


def build_datums_from_sdpo(
    prompt: tinker.ModelInput,
    sampled_tokens_G_T: List[List[int]],
    sampled_logprobs_G_T: List[List[float]],
    token_advantages_G_T: List[List[float]],
) -> List[tinker.Datum]:
    datums: list[tinker.Datum] = []
    ob_len = prompt.length - 1
    for sampled_tokens, logprobs, advs in zip(
        sampled_tokens_G_T, sampled_logprobs_G_T, token_advantages_G_T
    ):
        model_input = prompt.append(tinker.EncodedTextChunk(tokens=sampled_tokens[:-1]))
        target_tokens = [0] * ob_len + sampled_tokens
        padded_logprobs = [0.0] * ob_len + logprobs
        padded_advantages = [0.0] * ob_len + advs
        datum = tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(target_tokens)
                ),
                "logprobs": tinker.TensorData.from_torch(
                    torch.tensor(padded_logprobs)
                ),
                "advantages": tinker.TensorData.from_torch(
                    torch.tensor(padded_advantages)
                ),
            },
        )
        datums.append(datum)
    return datums


@dataclass
class TaskSamples:
    problem_id: int
    prompt: Any
    sampled_tokens: List[List[int]] = field(default_factory=list)
    logprobs: List[List[float]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    correct: List[bool] = field(default_factory=list)
    speedups: List[float] = field(default_factory=list)
    teacher_prompts: List[tinker.ModelInput] = field(default_factory=list)


def _load_problem_ids(split_path: str, max_tasks: int | None, problem_ids: str | None, task_offset: int = 0) -> list[int]:
    if problem_ids:
        return [int(x.strip()) for x in problem_ids.split(",") if x.strip()]
    split = json.loads(Path(split_path).read_text())
    ids = list(split["problem_ids"]["eval"])
    if task_offset > 0:
        ids = ids[task_offset:]
    if max_tasks is not None:
        ids = ids[:max_tasks]
    return ids


def _sample_task(sampling_client, renderer, prompt, k, max_tokens, temperature):
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
    task_prompt: str,
    messages: List[Any],
    teacher_renderer,
    prompt_cfg: PromptConfig,
    action_log_path: Path | None,
    step_idx: int,
    baseline_window: int = 32,
    correct_bonus: float = 0.0,
    include_feedback: bool = True,
    include_success_solution: bool = False,
) -> tuple[List[float], List[bool], List[float], List[tinker.ModelInput]]:
    rewards = []
    correct = []
    speedups = []
    teacher_inputs: List[tinker.ModelInput] = []
    successful_solution: str | None = None

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

        if include_success_solution and result.correctness and successful_solution is None:
            successful_solution = raw_action

        error_message, error_trace = extract_error_info(result.metadata)
        feedback = build_execution_feedback(
            compiled=result.compiled,
            correctness=result.correctness,
            speedup=result.speedup,
            runtime_us=result.runtime_us,
            ref_runtime_us=result.ref_runtime_us,
            error_message=error_message,
            error_trace=error_trace,
        )

        teacher_context = build_teacher_context(
            task_prompt,
            raw_action,
            feedback if include_feedback else "",
            successful_solution=successful_solution if include_success_solution else None,
        )

        teacher_prompt = teacher_renderer.build_generation_prompt(
            [
                {"role": "system", "content": prompt_cfg.system_prompt},
                {"role": "user", "content": teacher_context},
            ]
        )
        teacher_inputs.append(teacher_prompt)

    return rewards, correct, speedups, teacher_inputs


def _compute_teacher_logprobs(
    teacher_client: tinker.SamplingClient,
    teacher_prompts: List[tinker.ModelInput],
    sampled_tokens_G_T: List[List[int]],
) -> List[List[float | None]]:
    full_sequence_inputs = []
    for prompt, tokens in zip(teacher_prompts, sampled_tokens_G_T, strict=True):
        full_sequence_inputs.append(prompt.append(tinker.EncodedTextChunk(tokens=tokens)))

    async def _gather():
        return await asyncio.gather(
            *[teacher_client.compute_logprobs_async(seq) for seq in full_sequence_inputs]
        )

    full_logprobs = asyncio.run(_gather())
    trimmed: List[List[float | None]] = []
    for lps, tokens in zip(full_logprobs, sampled_tokens_G_T, strict=True):
        if not tokens:
            trimmed.append([])
            continue
        tail = lps[-len(tokens):]
        trimmed.append(tail)
    return trimmed


def log_step_metrics(metrics_log_path: Path, step_idx: int, task_samples_list: List[TaskSamples]) -> Dict[str, Any]:
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

    record = {
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
        handle.write(json.dumps(record) + "\n")

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="SDPO Batch TTT for kernel optimization")
    parser.add_argument("--split", type=str, default="splits/l1_seed42.json")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--renderer_name", type=str, default="gpt_oss_no_sysprompt")
    parser.add_argument("--checkpoint_jsonl", type=str, default="runs/rlvr_normalized_optimized_prompt/checkpoints.jsonl")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_tasks", type=int, default=5)
    parser.add_argument("--task_offset", type=int, default=0)
    parser.add_argument("--problem_ids", type=str, default="")
    parser.add_argument("--log_path", type=str, default="runs/sdpo_batch_ttt")
    parser.add_argument("--eval_mode", type=str, default="fast", choices=["full", "fast"])
    parser.add_argument("--num_correct_trials", type=int, default=1)
    parser.add_argument("--num_perf_trials", type=int, default=5)
    parser.add_argument("--baseline_window", type=int, default=32)
    parser.add_argument("--correct_bonus", type=float, default=0.0)
    parser.add_argument("--no_feedback", action="store_true")
    parser.add_argument("--include_success_solution", action="store_true")
    parser.add_argument("--sdpo_coef", type=float, default=1.0)
    parser.add_argument("--teacher_mode", type=str, default="student", choices=["student", "base"])
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

    problem_ids = _load_problem_ids(args.split, args.max_tasks, args.problem_ids or None, args.task_offset)
    if len(problem_ids) > args.batch_size:
        problem_ids = problem_ids[:args.batch_size]

    renderer_name = args.renderer_name or get_recommended_renderer_name(args.model)
    tokenizer = get_tokenizer(args.model)
    renderer = get_renderer(renderer_name, tokenizer)

    teacher_renderer = renderer

    service_client = tinker.ServiceClient()

    base_sampler_path = ""
    state_path = ""
    ckpt = load_latest_checkpoint(args.checkpoint_jsonl)
    if ckpt:
        base_sampler_path = ckpt.get("sampler_path", "")
        state_path = ckpt.get("state_path", "")

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
    base_sampling_client = sampling_client
    teacher_client = sampling_client if args.teacher_mode == "student" else base_sampling_client

    tasks = []
    prompts = []
    task_prompts = []
    prompt_cfg = PromptConfig()
    for pid in problem_ids:
        task = load_task(pid)
        tasks.append(task)
        messages = build_messages(task)
        prompt = renderer.build_generation_prompt(messages)
        prompts.append(prompt)
        task_prompts.append(messages[-1]["content"])

    step_summaries = []
    checkpoint_paths: Dict[int, str] = {0: sampling_path}
    compute = {
        "total_rollouts": 0,
        "student_prompt_tokens": 0,
        "student_completion_tokens": 0,
        "teacher_prompt_tokens": 0,
        "teacher_completion_tokens": 0,
        "teacher_logprob_calls": 0,
        "teacher_logprob_s": 0.0,
        "wall_clock_s": 0.0,
    }
    run_start = time.time()

    print(f"SDPO Batch TTT with {len(problem_ids)} tasks: {problem_ids}")
    print(f"K={args.k} samples/task, {args.steps} steps")

    for step in range(args.steps + 1):
        if step == 0:
            print("Step 0: Baseline evaluation...")
        else:
            print(f"Step {step}: SDPO update + re-sample...")

        task_samples_list: List[TaskSamples] = []
        for pid, task, prompt, task_prompt in zip(problem_ids, tasks, prompts, task_prompts):
            tokens, lps, messages = _sample_task(
                sampling_client, renderer, prompt, args.k, args.max_tokens, args.temperature
            )
            compute["total_rollouts"] += len(tokens)
            compute["student_prompt_tokens"] += prompt.length * len(tokens)
            compute["student_completion_tokens"] += sum(len(t) for t in tokens)
            rewards, correct, speedups, teacher_prompts = _evaluate_task_samples(
                pid,
                task.reference_code,
                task_prompt,
                messages,
                teacher_renderer,
                prompt_cfg,
                action_log_path,
                step_idx=step,
                baseline_window=args.baseline_window,
                correct_bonus=args.correct_bonus,
                include_feedback=not args.no_feedback,
                include_success_solution=args.include_success_solution,
            )
            compute["teacher_prompt_tokens"] += sum(tp.length for tp in teacher_prompts)
            compute["teacher_completion_tokens"] += sum(len(t) for t in tokens)
            ts = TaskSamples(
                problem_id=pid,
                prompt=prompt,
                sampled_tokens=tokens,
                logprobs=lps,
                rewards=rewards,
                correct=correct,
                speedups=speedups,
                teacher_prompts=teacher_prompts,
            )
            task_samples_list.append(ts)

        step_metrics = log_step_metrics(metrics_log_path, step, task_samples_list)
        step_metrics["checkpoint_path"] = sampling_path
        step_summaries.append(step_metrics)

        if step == args.steps:
            break

        # SDPO update
        all_datums: List[tinker.Datum] = []
        for ts in task_samples_list:
            start = time.time()
            teacher_logprobs = _compute_teacher_logprobs(
                teacher_client, ts.teacher_prompts, ts.sampled_tokens
            )
            compute["teacher_logprob_s"] += time.time() - start
            compute["teacher_logprob_calls"] += len(ts.teacher_prompts)
            token_adv = compute_sdpo_advantages(ts.logprobs, teacher_logprobs, args.sdpo_coef)
            all_datums.extend(
                build_datums_from_sdpo(ts.prompt, ts.sampled_tokens, ts.logprobs, token_adv)
            )

        fwd_bwd = training_client.forward_backward(all_datums, loss_fn="importance_sampling")
        optim = training_client.optim_step(tinker.AdamParams(learning_rate=args.learning_rate))
        _ = fwd_bwd.result()
        _ = optim.result()

        sampling_path = training_client.save_weights_for_sampler(name=f"step_{step}").result().path
        checkpoint_paths[step] = sampling_path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
        teacher_client = sampling_client if args.teacher_mode == "student" else base_sampling_client

    compute["wall_clock_s"] = time.time() - run_start

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
            "include_feedback": not args.no_feedback,
            "include_success_solution": args.include_success_solution,
            "sdpo_coef": args.sdpo_coef,
            "teacher_mode": args.teacher_mode,
        },
        "problem_ids": problem_ids,
        "step_summaries": step_summaries,
        "checkpoint_paths": {str(k): v for k, v in checkpoint_paths.items()},
        "final_checkpoint": sampling_path,
    }
    summary_path = log_dir / "sdpo_batch_ttt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    compute_path = log_dir / "sdpo_batch_ttt_compute.json"
    compute_path.write_text(json.dumps(compute, indent=2))
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote compute stats to {compute_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
