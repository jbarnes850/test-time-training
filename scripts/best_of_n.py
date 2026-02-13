"""Best-of-N with pipelined sampling and parallel evaluation.

Optimizations:
1. Prefetch sampling requests (pipeline)
2. Parallel kernel evaluation using ThreadPoolExecutor
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import statistics
import time
from typing import List, Tuple, Any

import tinker

from src.env.evaluator import evaluate_kernel, compute_reward
from src.env.tasking import load_task, build_messages
from src.metrics import fast_1
from src.utils.checkpoint_utils import load_latest_checkpoint
from src.utils.code_utils import extract_python_code, assemble_modelnew_code
from src.utils.env_utils import load_dotenv
from src.utils.path_utils import repo_root
from src.utils.tinker_utils import ensure_tinker_cookbook_on_path

load_dotenv()
ensure_tinker_cookbook_on_path()

from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _load_problem_ids(split_path: str, max_tasks: int | None, task_offset: int = 0, problem_ids: str | None = None) -> list[int]:
    if problem_ids:
        return [int(x.strip()) for x in problem_ids.split(",") if x.strip()]
    split = json.loads(Path(split_path).read_text())
    ids = list(split["problem_ids"]["eval"])
    if task_offset > 0:
        ids = ids[task_offset:]
    if max_tasks is not None:
        ids = ids[:max_tasks]
    return ids


def _submit_sample_request(sampling_client, renderer, prompt, k, max_tokens, temperature: float):
    """Submit sampling request and return future (non-blocking)."""
    return sampling_client.sample(
        prompt=prompt,
        num_samples=k,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            stop=renderer.get_stop_sequences(),
            temperature=temperature,
        ),
    )


def _process_sample_result(result, renderer):
    """Process completed sampling result."""
    messages_G = []
    token_counts = []
    for seq in result.sequences:
        msg, _ = renderer.parse_response(seq.tokens)
        messages_G.append(msg)
        token_counts.append(len(seq.tokens))
    return messages_G, token_counts


def _eval_single(args: Tuple) -> Tuple[int, float, bool, float, float]:
    """Evaluate single kernel (for parallel execution)."""
    problem_id, idx, raw_action, kernel_code, action_log_path, level = args
    if action_log_path:
        with action_log_path.open("a") as handle:
            handle.write(json.dumps({
                "problem_id": problem_id,
                "sample_idx": idx,
                "raw_action": raw_action,
                "assembled_code": kernel_code,
            }) + "\n")
    result = evaluate_kernel(problem_id, kernel_code, level=level)
    reward = compute_reward(result.speedup, result.correctness)
    runtime = result.runtime_us if result.runtime_us > 0 else 0.0
    return idx, reward, result.correctness, result.speedup, runtime


def _evaluate_messages(problem_id, ref_code, messages, action_log_path, max_workers: int = 4, level: int = 1):
    """Evaluate messages in parallel."""
    eval_tasks = []
    for idx, msg in enumerate(messages):
        raw_action = extract_python_code(get_text_content(msg))
        kernel_code = assemble_modelnew_code(raw_action, ref_code)
        eval_tasks.append((problem_id, idx, raw_action, kernel_code, action_log_path, level))

    results = [None] * len(messages)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_eval_single, t): t[1] for t in eval_tasks}
        for future in as_completed(futures):
            idx, reward, corr, spd, rt = future.result()
            results[idx] = (reward, corr, spd, rt)

    return ([r[0] for r in results], [r[1] for r in results],
            [r[2] for r in results], [r[3] for r in results])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="splits/l1_seed42.json")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_tasks", type=int, default=1)
    parser.add_argument("--task_offset", type=int, default=0, help="Skip first N eval tasks")
    parser.add_argument("--problem_ids", type=str, default="", help="Explicit comma-separated problem IDs")
    parser.add_argument("--log_path", type=str, default="runs/best_of_n")
    parser.add_argument("--checkpoint_jsonl", type=str, default="runs/rlvr_normalized_optimized_prompt/checkpoints.jsonl")
    parser.add_argument("--sampler_path", type=str, default="")
    parser.add_argument("--renderer_name", type=str, default="gpt_oss_no_sysprompt")
    parser.add_argument("--eval_mode", type=str, default="full", choices=["full", "fast"])
    parser.add_argument("--num_correct_trials", type=int, default=None)
    parser.add_argument("--num_perf_trials", type=int, default=None)
    parser.add_argument("--level", type=int, default=1, help="KernelBench level (1, 2, or 3)")
    parser.add_argument("--eval_workers", type=int, default=4, help="Parallel kernel eval workers")
    parser.add_argument("--prefetch", type=int, default=2, help="Tasks to prefetch samples for")
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

    problem_ids = _load_problem_ids(args.split, args.max_tasks, args.task_offset, args.problem_ids or None)

    renderer_name = args.renderer_name or get_recommended_renderer_name(args.model)
    tokenizer = get_tokenizer(args.model)
    renderer = get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient()
    sampling_path = args.sampler_path
    if not sampling_path:
        ckpt = load_latest_checkpoint(args.checkpoint_jsonl)
        if ckpt:
            sampling_path = ckpt.get("sampler_path", "")
    if sampling_path:
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
    else:
        training_client = service_client.create_lora_training_client(base_model=args.model, rank=8)
        sampling_path = training_client.save_weights_for_sampler(name="base").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

    # Prepare all tasks
    tasks_data = []
    for pid in problem_ids:
        task = load_task(pid, level=args.level)
        msgs = build_messages(task)
        prompt = renderer.build_generation_prompt(msgs)
        tasks_data.append({"problem_id": pid, "task": task, "prompt": prompt})

    summary = []
    compute = {
        "total_rollouts": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "wall_clock_s": 0.0,
    }
    run_start = time.time()

    # Prefetch sampling requests
    pending = {}
    for td in tasks_data[:args.prefetch]:
        pending[td["problem_id"]] = _submit_sample_request(
            sampling_client, renderer, td["prompt"],
            args.k, args.max_tokens, args.temperature
        )

    for i, td in enumerate(tasks_data):
        problem_id = td["problem_id"]

        # Prefetch next task
        next_idx = i + args.prefetch
        if next_idx < len(tasks_data):
            next_td = tasks_data[next_idx]
            pending[next_td["problem_id"]] = _submit_sample_request(
                sampling_client, renderer, next_td["prompt"],
                args.k, args.max_tokens, args.temperature
            )

        # Wait for current samples
        if problem_id in pending:
            result = pending.pop(problem_id).result()
        else:
            result = _submit_sample_request(
                sampling_client, renderer, td["prompt"],
                args.k, args.max_tokens, args.temperature
            ).result()

        messages_G, token_counts = _process_sample_result(result, renderer)
        rewards, correct, speedups, runtimes = _evaluate_messages(
            problem_id, td["task"].reference_code, messages_G, action_log_path,
            max_workers=args.eval_workers, level=args.level
        )
        sample_fast1 = fast_1(correct, speedups)
        best_speedup = 0.0
        best_correct = False
        for is_correct, sp in zip(correct, speedups):
            if is_correct and sp > best_speedup:
                best_speedup = sp
                best_correct = True
        selected_fast1 = 1.0 if best_correct and best_speedup > 1.0 else 0.0
        correct_rate = sum(1 for c in correct if c) / len(correct)
        latency = statistics.mean(runtimes) if runtimes else 0.0

        compute["total_rollouts"] += len(messages_G)
        compute["total_prompt_tokens"] += td["prompt"].length * len(messages_G)
        compute["total_completion_tokens"] += sum(token_counts)

        summary.append({
            "problem_id": problem_id,
            "fast_1": selected_fast1,
            "sample_fast_1": sample_fast1,
            "selected_speedup": best_speedup,
            "selected_correct": best_correct,
            "correct": correct_rate,
            "latency": latency,
        })
        print(f"[{i+1}/{len(tasks_data)}] Task {problem_id}: fast_1={selected_fast1:.0f}, speedup={best_speedup:.2f}x")

    compute["wall_clock_s"] = time.time() - run_start

    out_path = log_dir / "best_of_n_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    compute_path = log_dir / "best_of_n_compute.json"
    compute_path.write_text(json.dumps(compute, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {compute_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
