import argparse
import json
import os
from pathlib import Path
import statistics

import tinker

from src.env.evaluator import evaluate_kernel, compute_reward
from src.env.tasking import load_task, build_messages
from src.guardrails import GuardrailConfig, should_rollback
from src.metrics import fast_1
from src.utils.code_utils import extract_python_code, assemble_modelnew_code
from src.utils.env_utils import load_dotenv
from src.utils.checkpoint_utils import load_latest_checkpoint
from src.utils.path_utils import repo_root
from src.utils.tinker_utils import ensure_tinker_cookbook_on_path
from src.rlvr_utils import build_datums_from_group

load_dotenv()
ensure_tinker_cookbook_on_path()

from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _parse_problem_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _load_problem_ids(split_path: str, max_tasks: int | None, problem_ids: str | None) -> list[int]:
    explicit = _parse_problem_ids(problem_ids)
    if explicit:
        ids = explicit
    else:
        split = json.loads(Path(split_path).read_text())
        ids = list(split["problem_ids"]["eval"])
    if max_tasks is not None:
        ids = ids[:max_tasks]
    return ids


def _sample_group(sampling_client, renderer, prompt, k, max_tokens, temperature):
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
    sampled_tokens_G_T = []
    logprobs_G_T = []
    messages_G = []
    for seq in result.sequences:
        sampled_tokens_G_T.append(seq.tokens)
        logprobs_G_T.append(seq.logprobs)
        msg, _ = renderer.parse_response(seq.tokens)
        messages_G.append(msg)
    return sampled_tokens_G_T, logprobs_G_T, messages_G


def _evaluate_messages(problem_id, ref_code, messages, action_log_path, step_idx):
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
                            "step_idx": step_idx,
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
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_tasks", type=int, default=1)
    parser.add_argument("--problem_ids", type=str, default="")
    parser.add_argument("--log_path", type=str, default="runs/inner_loop")
    parser.add_argument("--checkpoint_jsonl", type=str, default="runs/rlvr_smoke/checkpoints.jsonl")
    parser.add_argument("--sampler_path", type=str, default="")
    parser.add_argument("--renderer_name", type=str, default="gpt_oss_no_sysprompt")
    parser.add_argument("--eval_mode", type=str, default="full", choices=["full", "fast"])
    parser.add_argument("--num_correct_trials", type=int, default=None)
    parser.add_argument("--num_perf_trials", type=int, default=None)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_after", type=int, default=3)
    parser.add_argument("--early_stop_min_reward_delta", type=float, default=0.0)
    parser.add_argument("--early_stop_min_fast1_delta", type=float, default=0.0)
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

    problem_ids = _load_problem_ids(args.split, args.max_tasks, args.problem_ids)

    renderer_name = args.renderer_name or get_recommended_renderer_name(args.model)
    tokenizer = get_tokenizer(args.model)
    renderer = get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient()

    base_sampler_path = args.sampler_path
    state_path = ""
    if not base_sampler_path:
        ckpt = load_latest_checkpoint(args.checkpoint_jsonl)
        if ckpt:
            base_sampler_path = ckpt.get("sampler_path", "")
            state_path = ckpt.get("state_path", "")

    summary = []

    for problem_id in problem_ids:
        task = load_task(problem_id)
        messages = build_messages(task)
        prompt = renderer.build_generation_prompt(messages)

        if state_path:
            training_client = service_client.create_training_client_from_state(state_path)
        else:
            training_client = service_client.create_lora_training_client(
                base_model=args.model, rank=args.lora_rank
            )

        if base_sampler_path:
            base_sampling_path = base_sampler_path
        else:
            base_sampling_path = training_client.save_weights_for_sampler(name="base").result().path
        sampling_path = base_sampling_path
        sampling_client = service_client.create_sampling_client(model_path=base_sampling_path)

        sampled_tokens_G_T, logprobs_G_T, messages_G = _sample_group(
            sampling_client,
            renderer,
            prompt,
            args.k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        rewards, correct, speedups, runtimes = _evaluate_messages(
            problem_id, task.reference_code, messages_G, action_log_path, step_idx=0
        )
        base_fast1 = fast_1(correct, speedups)
        base_correct = sum(1 for c in correct if c) / len(correct)
        base_latency = statistics.mean(runtimes) if runtimes else 0.0
        base_reward = statistics.mean(rewards) if rewards else 0.0
        with metrics_log_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "step_idx": 0,
                        "fast_1": base_fast1,
                        "correct": base_correct,
                        "reward": base_reward,
                        "latency": base_latency,
                    }
                )
                + "\n"
            )

        # Inner-loop updates
        reward_trace = []
        early_stop_step = None
        for step in range(args.steps):
            mean_reward = sum(rewards) / len(rewards)
            advantages = [r - mean_reward for r in rewards]
            datums = build_datums_from_group(prompt, sampled_tokens_G_T, logprobs_G_T, advantages)

            fwd_bwd = training_client.forward_backward(datums, loss_fn="importance_sampling")
            optim = training_client.optim_step(tinker.AdamParams(learning_rate=1e-5))
            _ = fwd_bwd.result()
            _ = optim.result()

            sampling_path = training_client.save_weights_for_sampler(name=f"step_{step}").result().path
            sampling_client = service_client.create_sampling_client(model_path=sampling_path)

            sampled_tokens_G_T, logprobs_G_T, messages_G = _sample_group(
                sampling_client,
                renderer,
                prompt,
                args.k,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            rewards, correct, speedups, runtimes = _evaluate_messages(
                problem_id, task.reference_code, messages_G, action_log_path, step_idx=step + 1
            )
            step_fast1 = fast_1(correct, speedups)
            step_correct = sum(1 for c in correct if c) / len(correct)
            step_latency = statistics.mean(runtimes) if runtimes else 0.0
            step_reward = statistics.mean(rewards) if rewards else 0.0
            reward_trace.append(step_reward)
            with metrics_log_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "problem_id": problem_id,
                            "step_idx": step + 1,
                            "fast_1": step_fast1,
                            "correct": step_correct,
                            "reward": step_reward,
                            "latency": step_latency,
                        }
                    )
                    + "\n"
                )
            if args.early_stop and (step + 1) >= args.early_stop_after:
                reward_ok = step_reward > base_reward + args.early_stop_min_reward_delta
                fast1_ok = step_fast1 > base_fast1 + args.early_stop_min_fast1_delta
                if not (reward_ok or fast1_ok):
                    early_stop_step = step + 1
                    with metrics_log_path.open("a") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "problem_id": problem_id,
                                    "step_idx": step + 1,
                                    "early_stop": True,
                                    "reason": "no_improvement",
                                    "base_reward": base_reward,
                                    "base_fast_1": base_fast1,
                                    "reward": step_reward,
                                    "fast_1": step_fast1,
                                }
                            )
                            + "\n"
                        )
                    break

        adapted_fast1 = fast_1(correct, speedups)
        adapted_correct = sum(1 for c in correct if c) / len(correct)
        adapted_latency = statistics.mean(runtimes) if runtimes else 0.0
        adapted_reward = statistics.mean(rewards) if rewards else 0.0

        cfg = GuardrailConfig(correctness_floor=0.0, latency_cap_multiplier=1.5)
        rollback = should_rollback(base_correct, base_latency, adapted_correct, adapted_latency, cfg)
        effective_fast1 = base_fast1 if rollback else adapted_fast1
        effective_correct = base_correct if rollback else adapted_correct

        summary.append({
            "problem_id": problem_id,
            "base_fast_1": base_fast1,
            "adapted_fast_1": adapted_fast1,
            "effective_fast_1": effective_fast1,
            "base_correct": base_correct,
            "adapted_correct": adapted_correct,
            "effective_correct": effective_correct,
            "base_reward": base_reward,
            "adapted_reward": adapted_reward,
            "reward_delta": adapted_reward - base_reward,
            "reward_trace": reward_trace,
            "early_stop_step": early_stop_step,
            "base_checkpoint": base_sampling_path,
            "adapted_checkpoint": sampling_path,
            "rollback": rollback,
        })

    out_path = log_dir / "inner_loop_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
