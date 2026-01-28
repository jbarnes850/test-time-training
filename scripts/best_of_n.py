import argparse
import json
import os
from pathlib import Path
import statistics

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


def _load_problem_ids(split_path: str, max_tasks: int | None) -> list[int]:
    split = json.loads(Path(split_path).read_text())
    ids = list(split["problem_ids"]["eval"])
    if max_tasks is not None:
        ids = ids[:max_tasks]
    return ids


def _sample_group(sampling_client, renderer, prompt, k, max_tokens, temperature: float):
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
    messages_G = []
    for seq in result.sequences:
        msg, _ = renderer.parse_response(seq.tokens)
        messages_G.append(msg)
    return messages_G


def _evaluate_messages(problem_id, ref_code, messages, action_log_path):
    rewards = []
    correct = []
    speedups = []
    runtimes = []
    for idx, msg in enumerate(messages):
        raw_action = extract_python_code(get_text_content(msg))
        kernel_code = assemble_modelnew_code(raw_action, ref_code)
        if action_log_path:
            with action_log_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "problem_id": problem_id,
                            "sample_idx": idx,
                            "raw_action": raw_action,
                            "assembled_code": kernel_code,
                        }
                    )
                    + "\n"
                )
        result = evaluate_kernel(problem_id, kernel_code)
        reward = compute_reward(result.speedup, result.correctness)
        rewards.append(reward)
        correct.append(result.correctness)
        speedups.append(result.speedup)
        runtimes.append(result.runtime_us if result.runtime_us > 0 else 0.0)
    return rewards, correct, speedups, runtimes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="splits/l1_seed42.json")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_tasks", type=int, default=1)
    parser.add_argument("--log_path", type=str, default="runs/best_of_n")
    parser.add_argument("--checkpoint_jsonl", type=str, default="runs/rlvr_smoke/checkpoints.jsonl")
    parser.add_argument("--sampler_path", type=str, default="")
    parser.add_argument("--renderer_name", type=str, default="gpt_oss_no_sysprompt")
    parser.add_argument("--eval_mode", type=str, default="full", choices=["full", "fast"])
    parser.add_argument("--num_correct_trials", type=int, default=None)
    parser.add_argument("--num_perf_trials", type=int, default=None)
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

    problem_ids = _load_problem_ids(args.split, args.max_tasks)

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

    summary = []

    for problem_id in problem_ids:
        task = load_task(problem_id)
        messages = build_messages(task)
        prompt = renderer.build_generation_prompt(messages)

        messages_G = _sample_group(
            sampling_client,
            renderer,
            prompt,
            args.k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        rewards, correct, speedups, runtimes = _evaluate_messages(
            problem_id, task.reference_code, messages_G, action_log_path
        )
        fast1 = fast_1(correct, speedups)
        correct_rate = sum(1 for c in correct if c) / len(correct)
        latency = statistics.mean(runtimes) if runtimes else 0.0

        summary.append({
            "problem_id": problem_id,
            "fast_1": fast1,
            "correct": correct_rate,
            "latency": latency,
        })

    out_path = log_dir / "best_of_n_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
