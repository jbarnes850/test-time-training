from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from src.env.mutator import (
    KernelMutator,
    MUTATION_SYSTEM_PROMPT,
    TinkerMutatorBackend,
    build_mutation_prompt,
    parse_mutation_response,
)
from src.env.replay_buffer import ReplayBuffer
from src.env.schema import KernelTask
from src.env.tasking import load_task
from src.env.teacher import category_id, infer_task_categories
from src.utils.env_utils import load_dotenv
from src.utils.path_utils import repo_root

load_dotenv()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = []
        curr: ast.AST | None = node
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
        return ".".join(reversed(parts))
    return ""


def _extract_call_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                names.add(name)
    return names


def _ast_node_count(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    return sum(1 for _ in ast.walk(tree))


def _line_count(code: str) -> int:
    return len([ln for ln in code.splitlines() if ln.strip()])


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _write_case_files(
    run_dir: Path,
    *,
    problem_id: int,
    seed_code: str,
    mutated_code: str,
    raw_response: str,
) -> dict[str, str]:
    case_dir = run_dir / "cases" / f"problem_{problem_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    seed_path = case_dir / "seed.py"
    mut_path = case_dir / "mutated.py"
    raw_path = case_dir / "raw_response.txt"
    seed_path.write_text(seed_code)
    mut_path.write_text(mutated_code)
    raw_path.write_text(raw_response)
    return {
        "seed_code_path": str(seed_path.relative_to(run_dir)),
        "mutated_code_path": str(mut_path.relative_to(run_dir)),
        "raw_response_path": str(raw_path.relative_to(run_dir)),
    }


def _load_problem_ids(
    split_path: str,
    subset: str,
    max_tasks: int | None,
    problem_ids: str | None,
) -> list[int]:
    if problem_ids:
        return [int(x.strip()) for x in problem_ids.split(",") if x.strip()]
    split = json.loads(Path(split_path).read_text())
    ids = list(split["problem_ids"][subset])
    if max_tasks is not None:
        ids = ids[:max_tasks]
    return ids


def _resolve_log_dir(log_path: str) -> Path:
    path = Path(log_path)
    if path.is_absolute():
        return path
    return repo_root() / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="splits/l1_seed42.json")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--problem_ids", type=str, default="")
    parser.add_argument("--max_tasks", type=int, default=5)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--mutator_model_path", type=str, default="moonshotai/Kimi-K2.5")
    parser.add_argument("--renderer_name", type=str, default="")
    parser.add_argument("--request_timeout_s", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--semantic_check", action="store_true")
    parser.add_argument("--semantic_correct_trials", type=int, default=1)
    parser.add_argument("--semantic_perf_trials", type=int, default=1)
    parser.add_argument("--log_path", type=str, default="runs/mutator_quality_eval")
    args = parser.parse_args()

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set.")

    run_dir = _resolve_log_dir(args.log_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    per_task_path = run_dir / "mutator_outputs.jsonl"
    summary_path = run_dir / "mutator_quality_summary.json"
    run_config_path = run_dir / "run_config.json"
    run_config_path.write_text(json.dumps(vars(args), indent=2))

    ids = _load_problem_ids(args.split, args.subset, args.max_tasks, args.problem_ids or None)
    tasks: list[KernelTask] = [load_task(pid, level=args.level) for pid in ids]
    print(
        f"[mutator_eval] tasks={len(tasks)} model={args.mutator_model_path} level={args.level}",
        flush=True,
    )

    replay_buffer = ReplayBuffer(run_dir / "scratch_replay.jsonl")
    print("[mutator_eval] initializing backend...", flush=True)
    backend = TinkerMutatorBackend(
        model_id=args.mutator_model_path,
        renderer_name=args.renderer_name or None,
        request_timeout_s=args.request_timeout_s,
    )
    print(
        f"[mutator_eval] backend={backend.backend_name} resolved_path={backend.resolved_model_path}",
        flush=True,
    )
    mutator = KernelMutator(
        backend=backend,
        replay_buffer=replay_buffer,
        max_retries=1,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        semantic_filter="fast" if args.semantic_check else "off",
        semantic_correct_trials=args.semantic_correct_trials,
        semantic_perf_trials=args.semantic_perf_trials,
    )

    stage_counts: Counter[str] = Counter()
    mutation_type_counts: Counter[str] = Counter()
    category_shift = 0
    latencies_s: list[float] = []
    op_jaccard_scores: list[float] = []
    ast_ratio_scores: list[float] = []
    accepted = 0

    for task_idx, task in enumerate(tasks):
        print(
            f"[mutator_eval] task {task_idx + 1}/{len(tasks)} pid={task.problem_id} start",
            flush=True,
        )
        seed_tags = sorted(infer_task_categories(task.reference_code))
        seed_category = category_id(set(seed_tags))
        seed_calls = _extract_call_names(task.reference_code)
        seed_lines = _line_count(task.reference_code)
        seed_ast_nodes = _ast_node_count(task.reference_code)
        prompt = build_mutation_prompt(task, target_category=seed_category)
        start = time.time()
        row: dict[str, object] = {
            "problem_id": task.problem_id,
            "task_name": task.name,
            "seed_category": seed_category,
            "seed_tags": seed_tags,
            "seed_line_count": seed_lines,
            "seed_ast_nodes": seed_ast_nodes,
            "seed_call_names": sorted(seed_calls),
        }
        try:
            raw_response = backend.generate_mutation(
                task,
                prompt,
                system_prompt=MUTATION_SYSTEM_PROMPT,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            stage_counts["api"] += 1
            error_text = f"{type(exc).__name__}: {exc}"
            row.update(
                {
                    "accepted": False,
                    "stage": "api",
                    "error": error_text,
                    "latency_s": time.time() - start,
                }
            )
            with per_task_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"[mutator_eval] pid={task.problem_id} stage=api error={error_text[:120]}",
                flush=True,
            )
            continue

        proposal = parse_mutation_response(raw_response)
        if proposal is None:
            stage_counts["format"] += 1
            row.update(
                {
                    "accepted": False,
                    "stage": "format",
                    "raw_response_preview": raw_response[:1200],
                    "latency_s": time.time() - start,
                }
            )
            with per_task_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"[mutator_eval] pid={task.problem_id} stage=format", flush=True)
            continue

        validation = mutator.validate_mutation(task, proposal.code)
        if not validation.valid:
            stage_counts[validation.stage] += 1
            row.update(
                {
                    "accepted": False,
                    "stage": validation.stage,
                    "message": validation.message,
                    "mutation_type": proposal.mutation_type,
                    "optimization_prompt": proposal.optimization_prompt,
                    "code_preview": proposal.code[:1200],
                    "latency_s": time.time() - start,
                }
            )
            with per_task_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"[mutator_eval] pid={task.problem_id} stage={validation.stage}", flush=True)
            continue

        if args.semantic_check:
            semantic_validation = mutator.semantic_check(task, proposal.code, level=args.level)
            if not semantic_validation.valid:
                stage_counts[semantic_validation.stage] += 1
                row.update(
                    {
                        "accepted": False,
                        "stage": semantic_validation.stage,
                        "message": semantic_validation.message,
                        "mutation_type": proposal.mutation_type,
                        "optimization_prompt": proposal.optimization_prompt,
                        "code_preview": proposal.code[:1200],
                        "latency_s": time.time() - start,
                    }
                )
                with per_task_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(
                    f"[mutator_eval] pid={task.problem_id} stage={semantic_validation.stage}",
                    flush=True,
                )
                continue

        mut_tags = sorted(infer_task_categories(proposal.code))
        mut_category = category_id(set(mut_tags))
        mut_calls = _extract_call_names(proposal.code)
        mut_lines = _line_count(proposal.code)
        mut_ast_nodes = _ast_node_count(proposal.code)
        op_jaccard = _jaccard(seed_calls, mut_calls)
        ast_ratio = (mut_ast_nodes / seed_ast_nodes) if seed_ast_nodes > 0 else 0.0
        op_jaccard_scores.append(op_jaccard)
        ast_ratio_scores.append(ast_ratio)

        accepted += 1
        mutation_type_counts[proposal.mutation_type] += 1
        stage_counts["accepted"] += 1
        if mut_category != seed_category:
            category_shift += 1
        latency = time.time() - start
        latencies_s.append(latency)
        case_paths = _write_case_files(
            run_dir,
            problem_id=task.problem_id,
            seed_code=task.reference_code,
            mutated_code=proposal.code,
            raw_response=proposal.raw_response,
        )
        row.update(
            {
                "accepted": True,
                "stage": "accepted",
                "mutation_type": proposal.mutation_type,
                "optimization_prompt": proposal.optimization_prompt,
                "mutated_tags": mut_tags,
                "mutated_category": mut_category,
                "category_shift": mut_category != seed_category,
                "interface_signature_hash": validation.interface_signature_hash,
                "novelty_hash": validation.novelty_hash,
                "mutated_line_count": mut_lines,
                "mutated_ast_nodes": mut_ast_nodes,
                "mutated_call_names": sorted(mut_calls),
                "new_call_names": sorted(mut_calls - seed_calls),
                "removed_call_names": sorted(seed_calls - mut_calls),
                "op_jaccard": op_jaccard,
                "ast_node_ratio": ast_ratio,
                "code_preview": proposal.code[:1200],
                "raw_response_preview": proposal.raw_response[:1200],
                "latency_s": latency,
                **case_paths,
            }
        )
        with per_task_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"[mutator_eval] pid={task.problem_id} stage=accepted category={mut_category} type={proposal.mutation_type}",
            flush=True,
        )

    n = len(tasks)
    summary = {
        "num_tasks": n,
        "accepted": accepted,
        "acceptance_rate": (accepted / n) if n else 0.0,
        "stage_counts": dict(stage_counts),
        "mutation_type_counts": dict(mutation_type_counts),
        "category_shift_count": category_shift,
        "category_shift_rate": (category_shift / accepted) if accepted else 0.0,
        "mean_latency_s": statistics.mean(latencies_s) if latencies_s else 0.0,
        "median_latency_s": statistics.median(latencies_s) if latencies_s else 0.0,
        "mean_op_jaccard": statistics.mean(op_jaccard_scores) if op_jaccard_scores else 0.0,
        "median_op_jaccard": statistics.median(op_jaccard_scores) if op_jaccard_scores else 0.0,
        "mean_ast_node_ratio": statistics.mean(ast_ratio_scores) if ast_ratio_scores else 0.0,
        "median_ast_node_ratio": statistics.median(ast_ratio_scores) if ast_ratio_scores else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
