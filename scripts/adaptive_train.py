from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.env.evaluator import EvalResult
from src.env.mutator import ApiMutatorBackend, KernelMutator, TinkerMutatorBackend
from src.env.replay_buffer import ReplayBuffer
from src.env.schema import CapabilityProfile, KernelTask, ReplayEntry
from src.env.solver import (
    DryRunSolverBackend,
    SolveOutcome,
    SolverBackend,
    SolverBackendConfig,
    TinkerSolverBackend,
)
from src.env.tasking import load_task
from src.env.teacher import CurriculumTeacher, category_id, infer_task_categories
from src.utils.checkpoint_utils import load_latest_checkpoint
from src.utils.env_utils import load_dotenv
from src.utils.path_utils import repo_root

load_dotenv()


BUDGET_TIER_CAPS = {
    "smoke": 50.0,
    "signal": 150.0,
    "full": 500.0,
}


@dataclass
class CostTracker:
    gpu_hour_rate: float = 1.20
    total_gpu_hours: float = 0.0
    total_api_usd: float = 0.0
    total_usd: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    epoch_costs: list[dict[str, Any]] = field(default_factory=list)

    def add_cost(
        self,
        component: str,
        *,
        gpu_hours: float = 0.0,
        api_usd: float = 0.0,
        epoch: int | None = None,
    ) -> None:
        gpu_hours = max(0.0, float(gpu_hours))
        api_usd = max(0.0, float(api_usd))
        gpu_cost = gpu_hours * self.gpu_hour_rate
        self.total_gpu_hours += gpu_hours
        self.total_api_usd += api_usd
        self.total_usd += gpu_cost + api_usd
        self.events.append(
            {
                "timestamp": time.time(),
                "epoch": epoch,
                "component": component,
                "gpu_hours": gpu_hours,
                "gpu_cost_usd": gpu_cost,
                "api_cost_usd": api_usd,
                "running_total_usd": self.total_usd,
            }
        )

    def finalize_epoch(self, epoch: int, start_total_usd: float) -> float:
        epoch_cost = max(0.0, self.total_usd - float(start_total_usd))
        self.epoch_costs.append({"epoch": epoch, "epoch_cost_usd": epoch_cost})
        return epoch_cost

    def projected_total(self, remaining_epochs: int, fallback_epoch_cost_usd: float) -> float:
        remaining = max(0, int(remaining_epochs))
        if self.epoch_costs:
            per_epoch = statistics.mean(e["epoch_cost_usd"] for e in self.epoch_costs)
        else:
            per_epoch = float(fallback_epoch_cost_usd)
        return self.total_usd + (remaining * per_epoch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_hour_rate": self.gpu_hour_rate,
            "total_gpu_hours": self.total_gpu_hours,
            "total_api_usd": self.total_api_usd,
            "total_usd": self.total_usd,
            "events": self.events,
            "epoch_costs": self.epoch_costs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostTracker":
        return cls(
            gpu_hour_rate=float(data.get("gpu_hour_rate", 1.20)),
            total_gpu_hours=float(data.get("total_gpu_hours", 0.0)),
            total_api_usd=float(data.get("total_api_usd", 0.0)),
            total_usd=float(data.get("total_usd", 0.0)),
            events=list(data.get("events", [])),
            epoch_costs=list(data.get("epoch_costs", [])),
        )


@dataclass(frozen=True)
class TaskHandle:
    task: KernelTask
    category_tags: tuple[str, ...]
    category_id: str


class DryRunMutatorBackend:
    @property
    def backend_name(self) -> str:
        return "dry_run"

    @property
    def model_id(self) -> str:
        return "dry_run/mutator"

    def generate_mutation(
        self,
        seed_task: KernelTask,
        prompt: str,
        *,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        return seed_task.reference_code


def _load_problem_ids(
    split_path: str,
    split_name: str,
    max_tasks: int | None,
    problem_ids: str | None,
) -> list[int]:
    if problem_ids:
        return [int(x.strip()) for x in problem_ids.split(",") if x.strip()]
    split = json.loads(Path(split_path).read_text())
    ids = list(split["problem_ids"][split_name])
    if max_tasks is not None:
        ids = ids[:max_tasks]
    return ids


def _prepare_task_handles(problem_ids: list[int], level: int) -> list[TaskHandle]:
    handles: list[TaskHandle] = []
    for pid in problem_ids:
        task = load_task(pid, level=level)
        tags = tuple(sorted(infer_task_categories(task.reference_code)))
        handles.append(TaskHandle(task=task, category_tags=tags, category_id=category_id(set(tags))))
    return handles


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _load_cost_tracker(path: Path, gpu_hour_rate: float) -> CostTracker:
    if not path.exists():
        return CostTracker(gpu_hour_rate=gpu_hour_rate)
    return CostTracker.from_dict(json.loads(path.read_text()))


def _resolve_log_dir(log_path: str) -> Path:
    path = Path(log_path)
    if path.is_absolute():
        return path
    return repo_root() / path


def _profile_solver(
    solver: SolverBackend,
    eval_tasks: list[TaskHandle],
    *,
    profile_k: int,
    temperature: float,
    max_tokens: int,
    level: int,
    eval_workers: int,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    total_wall = 0.0
    for handle in eval_tasks:
        outcome = solver.solve_task(
            handle.task,
            k=profile_k,
            temperature=temperature,
            max_tokens=max_tokens,
            level=level,
            eval_workers=eval_workers,
        )
        total_wall += outcome.wall_clock_s
        for idx, result in enumerate(outcome.eval_results):
            rows.append(
                {
                    "task_id": f"{handle.task.problem_id}",
                    "sample_idx": idx,
                    "category_id": handle.category_id,
                    "correctness": result.correctness,
                    "speedup": result.speedup,
                    "fast_1": 1.0 if result.correctness and result.speedup > 1.0 else 0.0,
                }
            )
    return rows, total_wall


def _select_seed_task(
    *,
    teacher: CurriculumTeacher,
    replay_buffer: ReplayBuffer,
    ranked_train_tasks: list[TaskHandle],
    profiles: list[CapabilityProfile],
    replay_recency_window: int,
) -> tuple[KernelTask, str, int, str]:
    weak_categories = [
        p.category_id
        for p in sorted(profiles, key=lambda p: p.correctness_rate)
    ]
    for cat in weak_categories:
        replay_seed = replay_buffer.select_seed(
            {
                "category_id": cat,
                "correct_only": True,
                "min_speedup": 1.0,
                "recency_window": replay_recency_window,
            }
        )
        if replay_seed and replay_seed.problem_id is not None:
            seed_task = KernelTask(
                problem_id=int(replay_seed.problem_id),
                name=f"replay_seed_{replay_seed.task_id}",
                reference_code=replay_seed.task_reference_code,
            )
            return seed_task, replay_seed.task_id, int(replay_seed.problem_id), replay_seed.category_id

    if not ranked_train_tasks:
        raise RuntimeError("No training tasks available for seed selection.")
    fallback = ranked_train_tasks[0]
    return (
        fallback.task,
        f"seed_{fallback.task.problem_id}",
        fallback.task.problem_id,
        fallback.category_id,
    )


def _budget_cap(args: argparse.Namespace) -> float:
    if args.budget_cap_usd is not None:
        return float(args.budget_cap_usd)
    return BUDGET_TIER_CAPS[args.budget_tier]


def _load_solver_paths(args: argparse.Namespace) -> tuple[str, str]:
    sampler_path = args.solver_sampler_path or args.sampler_path
    state_path = args.solver_training_state_path or args.training_state_path
    if args.checkpoint_jsonl:
        ckpt = load_latest_checkpoint(args.checkpoint_jsonl)
        if ckpt:
            if not sampler_path:
                sampler_path = ckpt.get("sampler_path", "") or sampler_path
            if not state_path:
                state_path = ckpt.get("state_path", "") or state_path
    return sampler_path, state_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_train", type=str, default="splits/l1_seed42.json")
    parser.add_argument("--split_eval", type=str, default="splits/l2_seed42.json")
    parser.add_argument("--train_subset", type=str, default="train")
    parser.add_argument("--eval_subset", type=str, default="eval")
    parser.add_argument("--train_problem_ids", type=str, default="")
    parser.add_argument("--eval_problem_ids", type=str, default="")
    parser.add_argument("--max_train_tasks", type=int, default=80)
    parser.add_argument("--max_eval_tasks", type=int, default=20)
    parser.add_argument("--tasks_per_epoch", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--profile_k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--level_train", type=int, default=1)
    parser.add_argument("--level_eval", type=int, default=2)
    parser.add_argument("--solver_backend", type=str, default="tinker", choices=["tinker", "dry_run"])
    parser.add_argument("--solver_model_id", type=str, default="")
    parser.add_argument("--solver_sampler_path", type=str, default="")
    parser.add_argument("--solver_renderer_name", type=str, default="")
    parser.add_argument("--solver_training_state_path", type=str, default="")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--sampler_path", type=str, default="")
    parser.add_argument("--checkpoint_jsonl", type=str, default="")
    parser.add_argument("--training_state_path", type=str, default="")
    parser.add_argument("--enable_training", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--renderer_name", type=str, default="gpt_oss_no_sysprompt")
    parser.add_argument("--mutator_backend", type=str, default="tinker", choices=["tinker", "api_stub"])
    parser.add_argument("--mutator_model_path", type=str, default="moonshotai/Kimi-K2.5")
    parser.add_argument("--mutator_renderer_name", type=str, default="")
    parser.add_argument("--mutator_request_timeout_s", type=float, default=180.0)
    parser.add_argument("--mutator_max_retries", type=int, default=3)
    parser.add_argument(
        "--mutator_semantic_filter",
        type=str,
        default="off",
        choices=["off", "fast"],
    )
    parser.add_argument("--mutator_semantic_correct_trials", type=int, default=1)
    parser.add_argument("--mutator_semantic_perf_trials", type=int, default=1)
    parser.add_argument("--teacher_strategy", type=str, default="inverse_correctness")
    parser.add_argument("--replay_recency_window", type=int, default=200)
    parser.add_argument("--log_path", type=str, default="runs/adaptive_phase1")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--budget_tier", type=str, default="smoke", choices=["smoke", "signal", "full"])
    parser.add_argument("--budget_cap_usd", type=float, default=None)
    parser.add_argument("--gpu_hour_rate", type=float, default=1.20)
    parser.add_argument("--default_epoch_cost_usd", type=float, default=40.0)
    parser.add_argument("--profile_api_usd_per_epoch", type=float, default=2.0)
    parser.add_argument("--mutate_api_usd_per_task", type=float, default=0.15)
    parser.add_argument("--solve_api_usd_per_task", type=float, default=0.5)
    parser.add_argument("--train_api_usd_per_epoch", type=float, default=5.0)
    parser.add_argument("--eval_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.eval_workers < 1:
        raise ValueError("--eval_workers must be >= 1")
    solver_backend_name = "dry_run" if args.dry_run else args.solver_backend
    needs_tinker_api = (
        solver_backend_name == "tinker"
        or (not args.dry_run and args.mutator_backend == "tinker")
    )
    if needs_tinker_api and not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set.")

    run_dir = _resolve_log_dir(args.resume_from or args.log_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "run_config.json"
    checkpoint_state_path = run_dir / "checkpoint_state.json"
    cost_tracker_path = run_dir / "cost_tracker.json"
    epoch_summary_path = run_dir / "epoch_summary.jsonl"
    capability_profiles_path = run_dir / "capability_profiles.jsonl"
    replay_path = run_dir / "replay_entries.jsonl"
    mutator_stats_path = run_dir / "mutator_stats.jsonl"

    replay_buffer = ReplayBuffer(replay_path)
    teacher = CurriculumTeacher(seed=args.seed)
    cost_tracker = _load_cost_tracker(cost_tracker_path, args.gpu_hour_rate)

    train_ids = _load_problem_ids(
        args.split_train,
        args.train_subset,
        args.max_train_tasks,
        args.train_problem_ids or None,
    )
    eval_ids = _load_problem_ids(
        args.split_eval,
        args.eval_subset,
        args.max_eval_tasks,
        args.eval_problem_ids or None,
    )
    train_tasks = _prepare_task_handles(train_ids, level=args.level_train)
    eval_tasks = _prepare_task_handles(eval_ids, level=args.level_eval)

    sampler_path, training_state_path = _load_solver_paths(args)
    solver_model_id = args.solver_model_id or args.model
    solver_renderer_name = args.solver_renderer_name or args.renderer_name
    if solver_backend_name == "dry_run":
        solver: SolverBackend = DryRunSolverBackend(sampler_path or "dry_run/sampler")
        mutator_backend = DryRunMutatorBackend()
    else:
        solver_config = SolverBackendConfig(
            backend=solver_backend_name,
            model_id=solver_model_id,
            sampler_path=sampler_path,
            renderer_name=solver_renderer_name,
            training_enabled=args.enable_training,
            training_state_path=training_state_path,
            lora_rank=args.lora_rank,
            learning_rate=args.learning_rate,
        )
        solver = TinkerSolverBackend(config=solver_config)
        if args.mutator_backend == "tinker":
            mutator_backend = TinkerMutatorBackend(
                model_id=args.mutator_model_path,
                renderer_name=args.mutator_renderer_name or None,
                request_timeout_s=args.mutator_request_timeout_s,
            )
        else:
            mutator_backend = ApiMutatorBackend(model_id=args.mutator_model_path)

    run_config = dict(vars(args))
    run_config["resolved_solver_backend"] = solver_backend_name
    run_config["resolved_solver_model_id"] = solver.model_id
    run_config["resolved_solver_sampler_path"] = solver.sampler_path
    run_config["resolved_solver_metadata"] = solver.metadata()
    run_config["resolved_mutator_backend"] = mutator_backend.backend_name
    run_config["resolved_mutator_model_id"] = mutator_backend.model_id
    _write_json(run_config_path, run_config)

    mutator = KernelMutator(
        mutator_backend,
        replay_buffer,
        max_retries=args.mutator_max_retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        semantic_filter=args.mutator_semantic_filter,
        semantic_correct_trials=args.mutator_semantic_correct_trials,
        semantic_perf_trials=args.mutator_semantic_perf_trials,
    )

    next_epoch = 0
    if checkpoint_state_path.exists():
        state = json.loads(checkpoint_state_path.read_text())
        next_epoch = int(state.get("next_epoch", 0))
        if state.get("current_sampler_path"):
            solver.sampler_path = str(state["current_sampler_path"])

    budget_cap = _budget_cap(args)
    start_epoch = next_epoch

    for epoch in range(start_epoch, args.epochs):
        mutator.reset_stats()
        remaining_epochs = args.epochs - epoch
        projected_total = cost_tracker.projected_total(
            remaining_epochs=remaining_epochs,
            fallback_epoch_cost_usd=args.default_epoch_cost_usd,
        )
        if projected_total > budget_cap:
            _append_jsonl(
                epoch_summary_path,
                {
                    "epoch": epoch,
                    "status": "halted_budget_projection",
                    "budget_cap_usd": budget_cap,
                    "projected_total_usd": projected_total,
                    "current_total_usd": cost_tracker.total_usd,
                },
            )
            break

        epoch_start_total = cost_tracker.total_usd
        epoch_records = 0
        successful_records = 0
        profile_rows, profile_wall = _profile_solver(
            solver,
            eval_tasks,
            profile_k=args.profile_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            level=args.level_eval,
            eval_workers=args.eval_workers,
        )
        cost_tracker.add_cost(
            "profile",
            gpu_hours=profile_wall / 3600.0,
            api_usd=0.0 if args.dry_run else args.profile_api_usd_per_epoch,
            epoch=epoch,
        )
        profiles = teacher.update_profile(profile_rows, epoch=epoch, split=args.eval_subset)
        for profile in profiles:
            _append_jsonl(capability_profiles_path, asdict(profile))

        ranked_train_tasks = teacher.rank_tasks(train_tasks, strategy=args.teacher_strategy)
        solve_outcomes: list[SolveOutcome] = []

        for _ in range(min(args.tasks_per_epoch, len(ranked_train_tasks))):
            seed_task, parent_task_id, seed_problem_id, seed_category = _select_seed_task(
                teacher=teacher,
                replay_buffer=replay_buffer,
                ranked_train_tasks=ranked_train_tasks,
                profiles=profiles,
                replay_recency_window=args.replay_recency_window,
            )
            cost_tracker.add_cost(
                "mutate",
                api_usd=0.0 if args.dry_run else args.mutate_api_usd_per_task,
                epoch=epoch,
            )
            mutator_target_category = seed_category
            mutated = mutator.mutate(
                seed_task,
                epoch=epoch,
                seed_problem_id=seed_problem_id,
                target_category=mutator_target_category,
                level=args.level_train,
            )
            if mutated is None:
                bank_eval = EvalResult(
                    compiled=False,
                    correctness=False,
                    runtime_us=-1.0,
                    ref_runtime_us=-1.0,
                    speedup=0.0,
                    metadata={"banked": True, "reason": "mutation_failed"},
                )
                bank_entry = ReplayEntry(
                    entry_id=f"bank_{epoch}_{seed_problem_id}_{time.time_ns()}",
                    task_id=f"bank_{seed_problem_id}_{epoch}",
                    parent_task_id=parent_task_id,
                    problem_id=seed_problem_id,
                    level=args.level_train,
                    category_id=seed_category,
                    task_reference_code=seed_task.reference_code,
                    kernel_code="",
                    eval_result=bank_eval,
                    reward=0.0,
                    sampler_path=solver.sampler_path,
                    backend="bank",
                    timestamp=time.time(),
                    epoch=epoch,
                    is_mutated=False,
                )
                replay_buffer.append(bank_entry)
                continue

            task_for_solver = KernelTask(
                problem_id=seed_problem_id,
                name=mutated.name,
                reference_code=mutated.reference_code,
            )
            solve_outcome = solver.solve_task(
                task_for_solver,
                k=args.k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                level=args.level_train,
                eval_workers=args.eval_workers,
            )
            solve_outcomes.append(solve_outcome)
            cost_tracker.add_cost(
                "solve",
                gpu_hours=solve_outcome.wall_clock_s / 3600.0,
                api_usd=0.0 if args.dry_run else args.solve_api_usd_per_task,
                epoch=epoch,
            )
            for idx, (kernel_code, eval_result, reward) in enumerate(
                zip(solve_outcome.kernel_codes, solve_outcome.eval_results, solve_outcome.rewards)
            ):
                kernel_hash = hashlib.sha256(kernel_code.encode("utf-8")).hexdigest()[:16]
                entry_id = f"{mutated.task_id}_{idx}_{kernel_hash}"
                replay_entry = ReplayEntry(
                    entry_id=entry_id,
                    task_id=mutated.task_id,
                    parent_task_id=parent_task_id,
                    problem_id=seed_problem_id,
                    level=args.level_train,
                    category_id=mutated.category_id,
                    task_reference_code=mutated.reference_code,
                    kernel_code=kernel_code,
                    eval_result=eval_result,
                    reward=reward,
                    sampler_path=solver.sampler_path,
                    backend=solver.backend_name,
                    timestamp=time.time(),
                    epoch=epoch,
                    is_mutated=True,
                )
                replay_buffer.append(replay_entry)
                epoch_records += 1
                if eval_result.correctness and eval_result.speedup > 1.0:
                    successful_records += 1

        if args.enable_training:
            solver.sampler_path = solver.train_on_outcomes(solve_outcomes, epoch=epoch)
            cost_tracker.add_cost(
                "train",
                api_usd=0.0 if args.dry_run else args.train_api_usd_per_epoch,
                epoch=epoch,
            )

        epoch_cost = cost_tracker.finalize_epoch(epoch, epoch_start_total)
        _write_json(cost_tracker_path, cost_tracker.to_dict())
        checkpoint_payload = {
            "next_epoch": epoch + 1,
            "current_sampler_path": solver.sampler_path,
            "total_usd": cost_tracker.total_usd,
            "timestamp": time.time(),
        }
        _write_json(checkpoint_state_path, checkpoint_payload)
        _append_jsonl(
            mutator_stats_path,
            {
                "epoch": epoch,
                **mutator.stats.as_dict(),
            },
        )
        _append_jsonl(
            epoch_summary_path,
            {
                "epoch": epoch,
                "records_added": epoch_records,
                "records_successful": successful_records,
                "success_rate": (successful_records / epoch_records) if epoch_records else 0.0,
                "epoch_cost_usd": epoch_cost,
                "running_total_usd": cost_tracker.total_usd,
                "sampler_path": solver.sampler_path,
                "mutator_stats": mutator.stats.as_dict(),
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
