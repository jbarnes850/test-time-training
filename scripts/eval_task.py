import argparse
import json
from pathlib import Path

from src.env.evaluator import evaluate_kernel, compute_reward, EvalConfig
from src.env.schema import EvalResult
from src.env.tasking import load_task, make_observation
from src.env.telemetry import build_record, validate_telemetry
from src.utils.feedback_utils import extract_error_info
from src.utils.path_utils import repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_id", type=int, required=True)
    parser.add_argument("--kernel_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="demo")
    parser.add_argument("--model_id", type=str, default="gpt-oss-20b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    run_dir = root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "kernels").mkdir(exist_ok=True)
    (run_dir / "prompts").mkdir(exist_ok=True)

    task = load_task(args.problem_id, level=args.level)
    obs = make_observation(task)

    kernel_code = Path(args.kernel_path).read_text()

    config = EvalConfig()
    if args.dry_run:
        eval_result = EvalResult(
            compiled=False,
            correctness=False,
            runtime_us=-1.0,
            ref_runtime_us=-1.0,
            speedup=0.0,
            metadata={"dry_run": True},
        )
    else:
        eval_result = evaluate_kernel(args.problem_id, kernel_code, level=args.level, config=config)

    reward = compute_reward(eval_result.speedup, eval_result.correctness)

    error_message, error_trace = extract_error_info(eval_result.metadata)
    record = build_record(
        run_id=args.run_name,
        seed=args.seed,
        model_id=args.model_id,
        problem_id=args.problem_id,
        level=args.level,
        prompt=obs.prompt,
        kernel_code=kernel_code,
        compiled=eval_result.compiled,
        correctness=eval_result.correctness,
        runtime_us=eval_result.runtime_us,
        ref_runtime_us=eval_result.ref_runtime_us,
        speedup=eval_result.speedup,
        reward=reward,
        timing_method=config.timing_method,
        backend=config.backend,
        precision=config.precision,
        eval_metadata=eval_result.metadata,
        error_message=error_message,
        error_trace=error_trace,
    )
    record_dict = record.to_dict()
    validate_telemetry(record_dict)

    telemetry_path = run_dir / "telemetry.jsonl"
    with telemetry_path.open("a") as f:
        f.write(json.dumps(record_dict) + "\n")

    kernel_out = run_dir / "kernels" / f"problem_{args.problem_id}.py"
    kernel_out.write_text(kernel_code)

    prompt_out = run_dir / "prompts" / f"problem_{args.problem_id}.txt"
    prompt_out.write_text(obs.prompt)

    print(f"Wrote telemetry: {telemetry_path}")
    print(f"Saved kernel: {kernel_out}")
    print(f"Saved prompt: {prompt_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
