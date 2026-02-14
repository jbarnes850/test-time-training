from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.env.evaluator import EvalResult
from src.env.mutator import ApiMutatorBackend, KernelMutator, TinkerMutatorBackend
from src.env.replay_buffer import ReplayBuffer
from src.env.schema import CapabilityProfile, KernelTask, ReplayEntry
from src.env.training_replay import TrainingReplayBuffer
from src.env.solver import (
    DryRunSolverBackend,
    SolveOutcome,
    SolverBackend,
    SolverBackendConfig,
    TinkerSolverBackend,
)
from src.env.tasking import load_task
from src.env.teacher import (
    CurriculumTeacher,
    HeuristicTeacherBackend,
    TinkerLLMTeacherBackend,
    ZONE_LEARNING,
    ZONE_MASTERED,
    ZONE_TOO_HARD,
    category_id,
    infer_task_categories,
)
from src.utils.checkpoint_utils import load_latest_checkpoint
from src.utils.dataset_utils import available_kernelbench_levels, load_kernelbench_level
from src.utils.env_utils import load_dotenv
from src.utils.feedback_utils import extract_error_info
from src.utils.path_utils import repo_root

load_dotenv()


PHASE1_CLAIM_SCOPE = {
    "phase": "phase1_inner_loop_only",
    "world_model_enabled": False,
    "in_scope_claim": "adaptive curriculum improves RLVR sample efficiency and cross-level transfer",
    "out_of_scope_claim": "outer-loop world model learns transition dynamics / possibility frontier",
}

EXPERIMENT_ARM_POLICY = {
    "custom": {
        "arm_role": "custom",
        "comparison_scope": "user_defined",
        "primary_matched_compute": False,
    },
    "B0": {
        "arm_role": "matched_compute_curriculum",
        "comparison_scope": "compute_controlled",
        "primary_matched_compute": True,
        "description": "Adaptive curriculum from base model initialization (no RLVR sampler init).",
    },
    "B1": {
        "arm_role": "matched_compute_baseline",
        "comparison_scope": "compute_controlled",
        "primary_matched_compute": True,
        "description": "Random curriculum baseline.",
    },
    "B2": {
        "arm_role": "matched_compute_baseline",
        "comparison_scope": "compute_controlled",
        "primary_matched_compute": True,
        "description": "Static easy-to-hard curriculum baseline.",
    },
    "B3": {
        "arm_role": "status_quo_anchor",
        "comparison_scope": "anchor_only",
        "primary_matched_compute": False,
        "description": "Existing fixed-distribution RLVR checkpoint without additional training.",
    },
    "C": {
        "arm_role": "matched_compute_treatment",
        "comparison_scope": "compute_controlled",
        "primary_matched_compute": True,
        "description": "Adaptive curriculum treatment.",
    },
}

PRIMARY_MATCHED_COMPUTE_ARMS = ("B0", "B1", "B2", "C")
PAPER_BASE_MODEL_ID = "openai/gpt-oss-120b"
TEACHER_DEFAULT_MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507"
BOOTSTRAP_TARGET_SPEEDUP_BAND = (1.1, 1.5)
BOOTSTRAP_GUARD_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("einsum_rewrite", ("torch.einsum(", "einsum(")),
    ("transpose_layout", (".transpose(", "transpose(")),
    ("permute_layout", (".permute(", "permute(")),
    ("movedim_layout", (".movedim(", "movedim(")),
    ("swapaxes_layout", (".swapaxes(", "swapaxes(")),
    ("as_strided_layout", ("as_strided(",)),
    ("explicit_stride", (".stride(", " stride(")),
)


@dataclass(frozen=True)
class TaskHandle:
    task: KernelTask
    level: int
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


def _parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_level_split_map(text: str) -> dict[int, str]:
    result: dict[int, str] = {}
    if not text.strip():
        return result
    for item in text.split(","):
        piece = item.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(
                f"Invalid level split mapping '{piece}'. Expected format '<level>:<path>'."
            )
        raw_level, raw_path = piece.split(":", 1)
        level = int(raw_level.strip())
        path = raw_path.strip()
        if not path:
            raise ValueError(f"Empty split path for level {level}.")
        result[level] = path
    return result


def _default_split_path_for_level(level: int, seed: int) -> Path:
    return repo_root() / "splits" / f"l{level}_seed{seed}.json"


def _resolve_level_problem_ids(
    *,
    level: int,
    subset: str,
    seed: int,
    split_path: str | None,
) -> list[int]:
    if split_path:
        split_file = Path(split_path)
        if not split_file.is_absolute():
            split_file = repo_root() / split_file
        if split_file.exists():
            return _load_problem_ids(
                str(split_file),
                split_name=subset,
                max_tasks=None,
                problem_ids=None,
            )

    default_split = _default_split_path_for_level(level, seed)
    if default_split.exists():
        return _load_problem_ids(
            str(default_split),
            split_name=subset,
            max_tasks=None,
            problem_ids=None,
        )

    ds = load_kernelbench_level(level)
    ids = [int(row["problem_id"]) for row in ds]
    rng = random.Random(seed + (level * 1009))
    rng.shuffle(ids)
    cutoff = int(len(ids) * 0.8)
    if subset == "train":
        return ids[:cutoff]
    if subset == "eval":
        return ids[cutoff:]
    raise ValueError(f"Unknown subset: {subset}")


def _allocate_level_counts(total: int, weights: list[float]) -> list[int]:
    if total <= 0:
        return [0 for _ in weights]
    if not weights:
        raise ValueError("weights must be non-empty")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    s = sum(weights)
    if s <= 0:
        raise ValueError("weights must sum to > 0")
    normalized = [w / s for w in weights]
    raw = [total * w for w in normalized]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    fractional_order = sorted(
        range(len(weights)),
        key=lambda i: (raw[i] - counts[i], -i),
        reverse=True,
    )
    for i in range(remainder):
        counts[fractional_order[i % len(counts)]] += 1
    return counts


def _normalize_zone_quotas(mastered: float, learning: float, too_hard: float) -> dict[str, float]:
    _ = too_hard
    quotas = {
        ZONE_MASTERED: max(0.0, float(mastered)),
        ZONE_LEARNING: max(0.0, float(learning)),
        ZONE_TOO_HARD: 0.0,
    }
    total = sum(quotas.values())
    if total <= 0:
        return {ZONE_MASTERED: 0.20, ZONE_LEARNING: 0.80, ZONE_TOO_HARD: 0.0}
    return {zone: value / total for zone, value in quotas.items()}


def _zone_counts_from_profiles(profiles: list[CapabilityProfile]) -> dict[str, float]:
    counts = {ZONE_MASTERED: 0.0, ZONE_LEARNING: 0.0, ZONE_TOO_HARD: 0.0}
    for profile in profiles:
        counts[ZONE_MASTERED] += profile.n_tasks * profile.mastered_task_rate
        counts[ZONE_LEARNING] += profile.n_tasks * profile.learning_task_rate
        counts[ZONE_TOO_HARD] += profile.n_tasks * profile.too_hard_task_rate
    return counts


def _adaptive_zone_quotas(
    base: dict[str, float],
    zone_counts: dict[str, float],
    *,
    target_learning_share: float,
    max_adjustment: float,
) -> tuple[dict[str, float], dict[str, float]]:
    adjusted = dict(base)
    total = sum(zone_counts.values())
    if total <= 0:
        return adjusted, {"learning_share": 0.0, "delta_learning": 0.0}
    learning_share = zone_counts.get(ZONE_LEARNING, 0.0) / total
    delta = target_learning_share - learning_share
    shift = max(-max_adjustment, min(max_adjustment, delta))
    if abs(shift) < 1e-9:
        return adjusted, {"learning_share": learning_share, "delta_learning": 0.0}

    adjusted[ZONE_LEARNING] = max(0.0, adjusted[ZONE_LEARNING] + shift)
    donor_total = adjusted[ZONE_MASTERED] + adjusted[ZONE_TOO_HARD]
    if donor_total > 0:
        shrink = adjusted[ZONE_LEARNING] - base[ZONE_LEARNING]
        adjusted[ZONE_MASTERED] = max(
            0.0, adjusted[ZONE_MASTERED] - (shrink * (adjusted[ZONE_MASTERED] / donor_total))
        )
        adjusted[ZONE_TOO_HARD] = max(
            0.0, adjusted[ZONE_TOO_HARD] - (shrink * (adjusted[ZONE_TOO_HARD] / donor_total))
        )
    adjusted = _normalize_zone_quotas(
        adjusted[ZONE_MASTERED],
        adjusted[ZONE_LEARNING],
        adjusted[ZONE_TOO_HARD],
    )
    return adjusted, {"learning_share": learning_share, "delta_learning": shift}


def _allocate_zone_counts(total_slots: int, quotas: dict[str, float]) -> dict[str, int]:
    zones = [ZONE_MASTERED, ZONE_LEARNING]
    weights = [quotas.get(zone, 0.0) for zone in zones]
    counts = _allocate_level_counts(total_slots, weights)
    out = {zone: count for zone, count in zip(zones, counts)}
    out[ZONE_TOO_HARD] = 0
    return out


def _build_zone_plan(zone_counts: dict[str, int]) -> list[str]:
    plan: list[str] = []
    for zone in (ZONE_MASTERED, ZONE_LEARNING, ZONE_TOO_HARD):
        plan.extend([zone] * max(0, int(zone_counts.get(zone, 0))))
    return plan


def _pick_category_for_zone(
    profiles_by_zone: dict[str, list[CapabilityProfile]],
    zone: str,
    fallback_category: str,
) -> CapabilityProfile | None:
    candidates = profiles_by_zone.get(zone, [])
    if candidates:
        return candidates[0]
    for alt in (ZONE_LEARNING, ZONE_MASTERED):
        alt_candidates = profiles_by_zone.get(alt, [])
        if alt_candidates:
            return alt_candidates[0]
    return None


def _zone_target_speedup_band(zone: str) -> tuple[float, float]:
    if zone == ZONE_LEARNING:
        return (1.3, 1.8)
    if zone == ZONE_TOO_HARD:
        return (1.2, 1.6)
    if zone == ZONE_MASTERED:
        return (1.8, 2.5)
    return (1.2, 1.8)


def _zone_decision_mode(zone: str) -> str:
    if zone == ZONE_LEARNING:
        return "learning"
    if zone == ZONE_TOO_HARD:
        return "too_hard_decompose"
    if zone == ZONE_MASTERED:
        return "mastered_warmup"
    return "fallback"


def _zone_reason_code(zone: str) -> str:
    if zone == ZONE_LEARNING:
        return "edge_signal"
    if zone == ZONE_TOO_HARD:
        return "decompose"
    if zone == ZONE_MASTERED:
        return "warmup"
    return "fallback"


def _prepare_task_handles(problem_ids: list[int], level: int) -> list[TaskHandle]:
    handles: list[TaskHandle] = []
    for pid in problem_ids:
        task = load_task(pid, level=level)
        tags = tuple(sorted(infer_task_categories(task.reference_code)))
        handles.append(
            TaskHandle(
                task=task,
                level=level,
                category_tags=tags,
                category_id=category_id(set(tags)),
            )
        )
    return handles


def _build_weighted_train_handles(
    args: argparse.Namespace,
    *,
    levels: list[int],
    weights: list[float],
) -> list[TaskHandle]:
    available_levels = set(available_kernelbench_levels())
    requested_pairs = [(lvl, wt) for lvl, wt in zip(levels, weights) if lvl in available_levels]
    if not requested_pairs:
        raise ValueError(
            f"None of requested seed levels are available. "
            f"requested={levels}, available={sorted(available_levels)}"
        )
    levels = [lvl for lvl, _ in requested_pairs]
    weights = [wt for _, wt in requested_pairs]
    counts = _allocate_level_counts(args.max_train_tasks, weights)
    level_split_map = _parse_level_split_map(args.seed_split_paths)

    train_handles: list[TaskHandle] = []
    for level, n_samples in zip(levels, counts):
        if n_samples <= 0:
            continue
        candidate_ids = _resolve_level_problem_ids(
            level=level,
            subset=args.train_subset,
            seed=args.seed,
            split_path=level_split_map.get(level),
        )
        if not candidate_ids:
            continue
        rng = random.Random(args.seed + (level * 9973))
        candidate_ids = list(candidate_ids)
        rng.shuffle(candidate_ids)
        selected_ids = candidate_ids[:n_samples]
        train_handles.extend(_prepare_task_handles(selected_ids, level=level))
    return train_handles


def _build_mixed_train_handles(args: argparse.Namespace) -> list[TaskHandle]:
    levels = _parse_int_csv(args.seed_levels)
    weights = _parse_float_csv(args.seed_mix)
    if len(levels) != len(weights):
        raise ValueError("seed_levels and seed_mix must have the same length.")
    return _build_weighted_train_handles(args, levels=levels, weights=weights)


def _build_uniform_train_handles(args: argparse.Namespace) -> list[TaskHandle]:
    levels = _parse_int_csv(args.seed_levels)
    if not levels:
        raise ValueError("seed_levels must be non-empty for uniform level pool mode.")
    weights = [1.0 for _ in levels]
    return _build_weighted_train_handles(args, levels=levels, weights=weights)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


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
            level=handle.level,
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
                    "runtime_us": result.runtime_us,
                    "fast_1": 1.0 if result.correctness and result.speedup > 1.0 else 0.0,
                }
            )
    return rows, total_wall


def _is_success_result(result: EvalResult, *, speedup_threshold: float) -> bool:
    return bool(result.correctness) and float(result.speedup) > float(speedup_threshold)


def _count_successes_from_eval_results(
    eval_results: list[EvalResult],
    *,
    speedup_threshold: float,
) -> tuple[int, int, float]:
    total = len(eval_results)
    success_count = sum(
        1 for result in eval_results if _is_success_result(result, speedup_threshold=speedup_threshold)
    )
    success_rate = (success_count / total) if total > 0 else 0.0
    return success_count, total, success_rate


def _run_admission_gate(
    verifier: SolverBackend,
    task: KernelTask,
    *,
    k: int,
    temperature: float,
    max_tokens: int,
    level: int,
    eval_workers: int,
    speedup_threshold: float,
) -> dict[str, Any]:
    outcome = verifier.solve_task(
        task,
        k=k,
        temperature=temperature,
        max_tokens=max_tokens,
        level=level,
        eval_workers=eval_workers,
    )
    success_count, total_count, success_rate = _count_successes_from_eval_results(
        outcome.eval_results,
        speedup_threshold=speedup_threshold,
    )
    return {
        "admitted": success_count > 0,
        "success_count": success_count,
        "total_count": total_count,
        "success_rate": success_rate,
        "wall_clock_s": float(outcome.wall_clock_s),
        "eval_results": outcome.eval_results,
    }


def _zone_failure_speedup_ceiling(zone: str) -> float:
    if zone == ZONE_TOO_HARD:
        return 1.10
    if zone == ZONE_LEARNING:
        return 1.80
    if zone == ZONE_MASTERED:
        return 2.20
    return 1.50


def _select_failure_exemplars(
    replay_buffer: ReplayBuffer,
    *,
    category_id: str,
    zone: str,
    recency_window: int,
    limit: int,
) -> list[dict[str, Any]]:
    rows = replay_buffer.query(
        category_id=category_id,
        recency_window=recency_window,
        limit=max(limit * 10, 40),
    )
    if not rows:
        return []

    speed_ceiling = _zone_failure_speedup_ceiling(zone)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for entry in rows:
        correctness = bool(entry.eval_result.correctness)
        compiled = bool(entry.eval_result.compiled)
        speedup = float(entry.eval_result.speedup)
        error_message, _ = extract_error_info(entry.eval_result.metadata)
        is_failure = (not correctness) or (speedup <= speed_ceiling)
        if not is_failure:
            continue
        severity = 0.0
        if not correctness:
            severity += 3.0
        if not compiled:
            severity += 1.0
        severity += max(0.0, speed_ceiling - speedup)
        # Slight recency preference via epoch.
        severity += max(0.0, float(entry.epoch) * 1e-3)
        ranked.append(
            (
                severity,
                {
                    "entry_id": entry.entry_id,
                    "task_id": entry.task_id,
                    "problem_id": entry.problem_id,
                    "category_id": entry.category_id,
                    "zone": zone,
                    "compiled": compiled,
                    "correctness": correctness,
                    "speedup": speedup,
                    "reward": float(entry.reward),
                    "error_message": error_message,
                    "epoch": int(entry.epoch),
                    "timestamp": float(entry.timestamp),
                },
            )
        )
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [payload for _, payload in ranked[: max(0, limit)]]


def _build_teacher_failure_context(
    replay_buffer: ReplayBuffer,
    profiles: list[CapabilityProfile],
    *,
    recency_window: int,
    per_zone_limit: int,
) -> list[dict[str, Any]]:
    by_zone: dict[str, list[CapabilityProfile]] = {
        ZONE_LEARNING: [],
        ZONE_TOO_HARD: [],
        ZONE_MASTERED: [],
    }
    for profile in profiles:
        if profile.zone in by_zone:
            by_zone[profile.zone].append(profile)
    for zone in by_zone:
        by_zone[zone] = sorted(
            by_zone[zone],
            key=lambda p: (-p.normalized_utility, -p.utility_score, p.category_id),
        )

    exemplars: list[dict[str, Any]] = []
    for zone in (ZONE_LEARNING, ZONE_TOO_HARD, ZONE_MASTERED):
        zone_profiles = by_zone.get(zone, [])
        if not zone_profiles:
            continue
        # Pull failures from strongest frontier categories first.
        for profile in zone_profiles[:3]:
            rows = _select_failure_exemplars(
                replay_buffer,
                category_id=profile.category_id,
                zone=zone,
                recency_window=recency_window,
                limit=per_zone_limit,
            )
            exemplars.extend(rows)
            if len([e for e in exemplars if e.get("zone") == zone]) >= per_zone_limit:
                break
    exemplars.sort(
        key=lambda e: (
            str(e.get("zone", "")),
            not bool(e.get("correctness", False)),
            float(e.get("speedup", 0.0)),
            -float(e.get("timestamp", 0.0)),
        )
    )
    return exemplars[: max(0, per_zone_limit * 3)]


def _next_probe_tasks(
    eval_tasks: list[TaskHandle],
    *,
    cursor: int,
    probe_tasks: int,
) -> tuple[list[TaskHandle], int]:
    if not eval_tasks or probe_tasks <= 0:
        return [], cursor
    count = min(len(eval_tasks), probe_tasks)
    picked: list[TaskHandle] = []
    curr = cursor
    for _ in range(count):
        picked.append(eval_tasks[curr % len(eval_tasks)])
        curr += 1
    return picked, curr % len(eval_tasks)


def _gradient_signal_summary(
    outcomes: list[SolveOutcome],
    *,
    std_threshold: float,
) -> dict[str, float]:
    if std_threshold < 0:
        raise ValueError("std_threshold must be >= 0")
    effective = 0
    for outcome in outcomes:
        if not outcome.rewards:
            continue
        reward_std = statistics.pstdev(outcome.rewards) if len(outcome.rewards) > 1 else 0.0
        if reward_std > std_threshold:
            effective += 1
    total = len(outcomes)
    zero_signal = max(0, total - effective)
    return {
        "effective_tasks": float(effective),
        "zero_signal_tasks": float(zero_signal),
        "effective_task_rate": (effective / total) if total > 0 else 0.0,
        "std_threshold": float(std_threshold),
        "task_count": float(total),
    }


def _select_seed_task(
    *,
    replay_buffer: ReplayBuffer,
    ranked_train_tasks: list[TaskHandle],
    profiles: list[CapabilityProfile],
    replay_recency_window: int,
    preferred_category: str,
) -> tuple[KernelTask, str, int, str, int]:
    weak_categories = [preferred_category]
    weak_categories.extend(
        [
            p.category_id
            for p in sorted(profiles, key=lambda p: p.correctness_rate)
            if p.category_id != preferred_category
        ]
    )
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
            replay_level = int(replay_seed.level) if replay_seed.level is not None else 1
            return (
                seed_task,
                replay_seed.task_id,
                int(replay_seed.problem_id),
                replay_seed.category_id,
                replay_level,
            )

    if not ranked_train_tasks:
        raise RuntimeError("No training tasks available for seed selection.")
    fallback = ranked_train_tasks[0]
    return (
        fallback.task,
        f"seed_{fallback.task.problem_id}",
        fallback.task.problem_id,
        fallback.category_id,
        fallback.level,
    )


def _bootstrap_mode_active(profiles_by_zone: dict[str, list[CapabilityProfile]]) -> bool:
    learning = profiles_by_zone.get(ZONE_LEARNING, [])
    mastered = profiles_by_zone.get(ZONE_MASTERED, [])
    return not learning and not mastered


def _compute_forgetting_indicator(
    profiles_by_zone: dict[str, list[CapabilityProfile]],
    prior_mastered: set[str],
) -> dict[str, Any]:
    current_mastered = {p.category_id for p in profiles_by_zone.get(ZONE_MASTERED, [])}
    regressed = sorted(prior_mastered - current_mastered)
    return {"regressed_categories": regressed, "count": len(regressed)}


def _select_bootstrap_seed_task(
    *,
    replay_buffer: ReplayBuffer,
    ranked_train_tasks: list[TaskHandle],
    replay_recency_window: int,
    bootstrap_index: int = 0,
) -> tuple[KernelTask, str, int, str, int, str]:
    replay_seed = replay_buffer.select_seed(
        {
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
        replay_level = int(replay_seed.level) if replay_seed.level is not None else 1
        return (
            seed_task,
            replay_seed.task_id,
            int(replay_seed.problem_id),
            replay_seed.category_id,
            replay_level,
            "replay_positive",
        )

    if not ranked_train_tasks:
        raise RuntimeError("No training tasks available for bootstrap seed selection.")

    l1_candidates = sorted(
        (h for h in ranked_train_tasks if int(h.level) == 1),
        key=lambda h: h.task.problem_id,
    )
    if l1_candidates:
        anchor = l1_candidates[bootstrap_index % len(l1_candidates)]
        source = "l1_anchor"
    else:
        train_candidates = sorted(ranked_train_tasks, key=lambda h: h.task.problem_id)
        anchor = train_candidates[bootstrap_index % len(train_candidates)]
        source = "train_anchor"
    return (
        anchor.task,
        f"seed_{anchor.task.problem_id}",
        anchor.task.problem_id,
        anchor.category_id,
        anchor.level,
        source,
    )


def _passes_bootstrap_mutation_guard(
    seed_reference_code: str,
    mutated_reference_code: str,
) -> tuple[bool, str]:
    seed = seed_reference_code.lower()
    mutated = mutated_reference_code.lower()
    for label, probes in BOOTSTRAP_GUARD_PATTERNS:
        seed_count = sum(seed.count(p) for p in probes)
        mut_count = sum(mutated.count(p) for p in probes)
        if mut_count > seed_count:
            return False, label
    # Guard against introducing explicit transpose shorthand such as B.T
    seed_dot_t = len(re.findall(r"\.\s*t\b", seed))
    mut_dot_t = len(re.findall(r"\.\s*t\b", mutated))
    if mut_dot_t > seed_dot_t:
        return False, "dot_t_layout"
    return True, ""


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


def _normalize_arm_label(label: str) -> str:
    arm = label.strip().upper()
    if not arm:
        return "custom"
    if arm == "CUSTOM":
        return "custom"
    if arm in {"B0", "B1", "B2", "B3", "C"}:
        return arm
    raise ValueError(f"Unsupported experiment arm: {label}")


def _set_arm_default(
    args: argparse.Namespace,
    field_name: str,
    value: Any,
    applied: dict[str, dict[str, Any]],
) -> None:
    current = getattr(args, field_name)
    if current == value:
        return
    applied[field_name] = {"from": current, "to": value}
    setattr(args, field_name, value)


def _apply_experiment_arm_defaults(args: argparse.Namespace) -> tuple[str, dict[str, dict[str, Any]]]:
    arm = _normalize_arm_label(args.experiment_arm)
    applied: dict[str, dict[str, Any]] = {}
    if arm == "custom":
        return arm, applied

    # Enforce baseline purity and explicit arm semantics.
    if arm == "B1":
        _set_arm_default(args, "model", PAPER_BASE_MODEL_ID, applied)
        _set_arm_default(args, "solver_model_id", "", applied)
        _set_arm_default(args, "teacher_model_id", TEACHER_DEFAULT_MODEL_ID, applied)
        _set_arm_default(args, "teacher_strategy", "random", applied)
        _set_arm_default(args, "curriculum_controller", "fixed", applied)
        _set_arm_default(args, "seed_use_mixed_pool", True, applied)
        _set_arm_default(args, "seed_pool_mode", "uniform_levels", applied)
    elif arm == "B2":
        _set_arm_default(args, "model", PAPER_BASE_MODEL_ID, applied)
        _set_arm_default(args, "solver_model_id", "", applied)
        _set_arm_default(args, "teacher_model_id", TEACHER_DEFAULT_MODEL_ID, applied)
        _set_arm_default(args, "teacher_strategy", "easy_to_hard_static", applied)
        _set_arm_default(args, "curriculum_controller", "fixed", applied)
        _set_arm_default(args, "seed_use_mixed_pool", True, applied)
        _set_arm_default(args, "seed_pool_mode", "uniform_levels", applied)
    elif arm == "C":
        _set_arm_default(args, "model", PAPER_BASE_MODEL_ID, applied)
        _set_arm_default(args, "solver_model_id", "", applied)
        _set_arm_default(args, "teacher_model_id", TEACHER_DEFAULT_MODEL_ID, applied)
        _set_arm_default(args, "teacher_strategy", "frontier_band", applied)
        _set_arm_default(args, "curriculum_controller", "adaptive", applied)
        _set_arm_default(args, "seed_use_mixed_pool", True, applied)
        _set_arm_default(args, "seed_pool_mode", "uniform_levels", applied)
    elif arm == "B0":
        _set_arm_default(args, "model", PAPER_BASE_MODEL_ID, applied)
        _set_arm_default(args, "solver_model_id", "", applied)
        _set_arm_default(args, "teacher_model_id", TEACHER_DEFAULT_MODEL_ID, applied)
        _set_arm_default(args, "teacher_strategy", "frontier_band", applied)
        _set_arm_default(args, "curriculum_controller", "adaptive", applied)
        _set_arm_default(args, "seed_use_mixed_pool", True, applied)
        _set_arm_default(args, "seed_pool_mode", "uniform_levels", applied)
        # Force base-model initialization semantics.
        _set_arm_default(args, "solver_sampler_path", "", applied)
        _set_arm_default(args, "sampler_path", "", applied)
        _set_arm_default(args, "checkpoint_jsonl", "", applied)
    elif arm == "B3":
        _set_arm_default(args, "model", PAPER_BASE_MODEL_ID, applied)
        _set_arm_default(args, "solver_model_id", "", applied)
        _set_arm_default(args, "teacher_model_id", TEACHER_DEFAULT_MODEL_ID, applied)
        _set_arm_default(args, "curriculum_controller", "fixed", applied)
        _set_arm_default(args, "enable_training", False, applied)
        _set_arm_default(args, "tasks_per_epoch", 0, applied)
    return arm, applied


def _build_training_batch(
    current_outcomes: list[SolveOutcome],
    current_zones: list[str],
    current_utilities: list[float],
    training_replay: TrainingReplayBuffer,
    *,
    current_epoch: int,
    replay_fraction: float = 0.30,
    recency_epochs: int = 3,
    decay_rate: float = 0.8,
    std_threshold: float = 1e-5,
) -> tuple[list[SolveOutcome], list[float]]:
    learning_outcomes: list[SolveOutcome] = []
    learning_weights: list[float] = []
    for outcome, zone, utility in zip(current_outcomes, current_zones, current_utilities):
        if zone not in {ZONE_LEARNING, ZONE_MASTERED}:
            continue
        if not outcome.rewards:
            continue
        reward_std = statistics.pstdev(outcome.rewards) if len(outcome.rewards) > 1 else 0.0
        variance_factor = min(1.0, reward_std / std_threshold) if std_threshold > 0 else 1.0
        w = utility * 1.0 * variance_factor
        learning_outcomes.append(outcome)
        learning_weights.append(w)

    replay_artifacts = training_replay.sample_replay(
        current_epoch=current_epoch,
        recency_epochs=recency_epochs,
        zone_filter=ZONE_LEARNING,
    )
    n_current = len(learning_outcomes)
    total_target = int(n_current / (1.0 - replay_fraction)) if replay_fraction < 1.0 else n_current * 2
    n_replay_target = max(0, total_target - n_current)
    n_replay = min(n_replay_target, len(replay_artifacts))

    replay_outcomes: list[SolveOutcome] = []
    replay_weights: list[float] = []
    if n_replay > 0:
        grouped: dict[str, list] = {}
        for artifact in replay_artifacts[:n_replay]:
            grouped.setdefault(artifact.outcome_id, []).append(artifact)
        for outcome_id, artifacts in grouped.items():
            age = max(0, current_epoch - artifacts[0].epoch)
            recency_decay = decay_rate ** age
            utility = artifacts[0].utility_score
            rewards = [a.reward for a in artifacts]
            reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            variance_factor = min(1.0, reward_std / std_threshold) if std_threshold > 0 else 1.0
            w = utility * recency_decay * variance_factor
            replay_outcome = SolveOutcome(
                prompt=None,
                sampled_tokens=[a.sampled_tokens for a in artifacts],
                sampled_logprobs=[a.sampled_logprobs for a in artifacts],
                raw_actions=[],
                kernel_codes=[],
                eval_results=[],
                rewards=rewards,
                wall_clock_s=0.0,
                prompt_tokens=artifacts[0].prompt_tokens,
            )
            replay_outcomes.append(replay_outcome)
            replay_weights.append(w)

    mixed_outcomes = learning_outcomes + replay_outcomes
    raw_weights = learning_weights + replay_weights
    if not raw_weights:
        return mixed_outcomes, [1.0] * len(mixed_outcomes)
    clipped = [max(0.01, min(5.0, w)) for w in raw_weights]
    mean_w = statistics.mean(clipped) if clipped else 1.0
    normalized = [w / mean_w if mean_w > 0 else 1.0 for w in clipped]
    return mixed_outcomes, normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_arm",
        type=str,
        default="custom",
        choices=["custom", "B0", "B1", "B2", "B3", "C"],
    )
    parser.add_argument("--split_train", type=str, default="splits/l1_seed42.json")
    parser.add_argument("--split_eval", type=str, default="splits/l2_seed42.json")
    parser.add_argument("--train_subset", type=str, default="train")
    parser.add_argument("--eval_subset", type=str, default="eval")
    parser.add_argument("--seed_use_mixed_pool", action="store_true")
    parser.add_argument("--seed_levels", type=str, default="1,2,3,4,5")
    parser.add_argument("--seed_mix", type=str, default="0.25,0.45,0.20,0.10,0.00")
    parser.add_argument(
        "--seed_pool_mode",
        type=str,
        default="configured_mix",
        choices=["configured_mix", "uniform_levels"],
    )
    parser.add_argument("--seed_split_paths", type=str, default="")
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
    parser.add_argument(
        "--min_discovery_epochs",
        type=int,
        default=0,
        help="Skip training for the first N epochs to accumulate replay diversity",
    )
    parser.add_argument(
        "--train_subsample_per_outcome",
        type=int,
        default=0,
        help="Max rollouts per outcome for training datums (0 = use all k)",
    )
    parser.add_argument(
        "--rollback_threshold",
        type=float,
        default=0.0,
        help="If agg_fast_1 drops by this fraction, skip next train + halve LR (0 = disabled)",
    )
    parser.add_argument("--renderer_name", type=str, default="gpt_oss_no_sysprompt")
    parser.add_argument("--mutator_backend", type=str, default="tinker", choices=["tinker", "api_stub"])
    parser.add_argument("--mutator_model_path", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument(
        "--mutator_frontier_model_path",
        type=str,
        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
    )
    parser.add_argument("--mutator_renderer_name", type=str, default="")
    parser.add_argument("--mutator_request_timeout_s", type=float, default=180.0)
    parser.add_argument("--mutator_primary_retries", type=int, default=1)
    parser.add_argument("--mutator_frontier_retries", type=int, default=2)
    parser.add_argument("--mutator_max_retries", type=int, default=3)  # backward-compatible cap
    parser.add_argument(
        "--mutator_semantic_filter",
        type=str,
        default="off",
        choices=["off", "fast"],
    )
    parser.add_argument("--mutator_semantic_correct_trials", type=int, default=1)
    parser.add_argument("--mutator_semantic_perf_trials", type=int, default=1)
    parser.add_argument(
        "--teacher_strategy",
        type=str,
        default="frontier_band",
        choices=["frontier_band", "inverse_correctness", "loss_proportional", "easy_to_hard_static", "random"],
    )
    parser.add_argument("--teacher_backend", type=str, default="tinker", choices=["heuristic", "tinker"])
    parser.add_argument("--teacher_model_id", type=str, default=TEACHER_DEFAULT_MODEL_ID)
    parser.add_argument("--teacher_renderer_name", type=str, default="")
    parser.add_argument("--teacher_request_timeout_s", type=float, default=45.0)
    parser.add_argument("--teacher_temperature", type=float, default=0.1)
    parser.add_argument("--teacher_max_tokens", type=int, default=384)
    parser.add_argument("--teacher_failure_exemplars_per_zone", type=int, default=3)
    parser.add_argument("--teacher_seed_failure_exemplars", type=int, default=4)
    parser.add_argument("--learnability_low", type=float, default=0.25)
    parser.add_argument("--learnability_high", type=float, default=0.75)
    parser.add_argument("--curriculum_mastered_quota", type=float, default=0.15)
    parser.add_argument("--curriculum_learning_quota", type=float, default=0.60)
    parser.add_argument("--curriculum_too_hard_quota", type=float, default=0.0)
    parser.add_argument("--curriculum_target_learning_share", type=float, default=0.50)
    parser.add_argument("--curriculum_max_adjustment", type=float, default=0.15)
    parser.add_argument(
        "--curriculum_controller",
        type=str,
        default="adaptive",
        choices=["adaptive", "fixed"],
    )
    parser.add_argument("--replay_recency_window", type=int, default=200)
    parser.add_argument("--replay_fraction", type=float, default=0.30)
    parser.add_argument("--replay_recency_epochs", type=int, default=3)
    parser.add_argument("--replay_decay_rate", type=float, default=0.8)
    parser.add_argument("--bootstrap_exit_patience", type=int, default=2)
    parser.add_argument("--log_path", type=str, default="runs/adaptive_phase1")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--budget_tier", type=str, default="smoke", choices=["smoke", "signal", "full"])
    parser.add_argument("--budget_cap_usd", type=float, default=None)
    parser.add_argument("--gpu_hour_rate", type=float, default=1.20)
    parser.add_argument("--default_epoch_cost_usd", type=float, default=40.0)
    parser.add_argument("--profile_api_usd_per_epoch", type=float, default=2.0)
    parser.add_argument("--mini_profile_api_usd_per_probe", type=float, default=0.5)
    parser.add_argument("--mutate_api_usd_per_task", type=float, default=0.15)
    parser.add_argument("--admission_api_usd_per_task", type=float, default=0.08)
    parser.add_argument("--solve_api_usd_per_task", type=float, default=0.5)
    parser.add_argument("--train_api_usd_per_epoch", type=float, default=5.0)
    parser.add_argument("--teacher_api_usd_per_epoch", type=float, default=0.2)
    parser.add_argument("--teacher_api_usd_per_seed_directive", type=float, default=0.03)
    parser.add_argument("--eval_workers", type=int, default=1)
    parser.add_argument("--mini_reprofile_every", type=int, default=4)
    parser.add_argument("--mini_reprofile_probe_tasks", type=int, default=6)
    parser.add_argument("--mini_reprofile_k", type=int, default=1)
    parser.add_argument("--effective_task_std_threshold", type=float, default=1e-5)
    parser.add_argument("--min_effective_tasks", type=int, default=4)
    parser.add_argument("--min_effective_tasks_patience", type=int, default=2)
    parser.add_argument("--enable_admission_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--admission_k", type=int, default=8)
    parser.add_argument("--admission_speedup_threshold", type=float, default=1.5)
    parser.add_argument(
        "--admission_bootstrap_passthrough",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--admission_bootstrap_passthrough_limit", type=int, default=1)
    parser.add_argument(
        "--verifier_backend",
        type=str,
        default="",
        choices=["", "tinker", "dry_run"],
    )
    parser.add_argument(
        "--verifier_model_id",
        type=str,
        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
    )
    parser.add_argument("--verifier_sampler_path", type=str, default="")
    parser.add_argument("--verifier_renderer_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_arm, applied_arm_defaults = _apply_experiment_arm_defaults(args)

    if args.eval_workers < 1:
        raise ValueError("--eval_workers must be >= 1")
    if args.teacher_failure_exemplars_per_zone < 0:
        raise ValueError("--teacher_failure_exemplars_per_zone must be >= 0")
    if args.teacher_seed_failure_exemplars < 0:
        raise ValueError("--teacher_seed_failure_exemplars must be >= 0")
    if args.mini_reprofile_every < 0:
        raise ValueError("--mini_reprofile_every must be >= 0")
    if args.mini_reprofile_probe_tasks < 0:
        raise ValueError("--mini_reprofile_probe_tasks must be >= 0")
    if args.mini_reprofile_k < 1:
        raise ValueError("--mini_reprofile_k must be >= 1")
    if args.min_effective_tasks < 0:
        raise ValueError("--min_effective_tasks must be >= 0")
    if args.min_effective_tasks_patience < 1:
        raise ValueError("--min_effective_tasks_patience must be >= 1")
    if args.effective_task_std_threshold < 0:
        raise ValueError("--effective_task_std_threshold must be >= 0")
    if args.admission_k < 1:
        raise ValueError("--admission_k must be >= 1")
    if args.admission_speedup_threshold <= 0:
        raise ValueError("--admission_speedup_threshold must be > 0")
    if args.admission_bootstrap_passthrough_limit < 0:
        raise ValueError("--admission_bootstrap_passthrough_limit must be >= 0")
    if not (0.0 <= args.learnability_low < args.learnability_high <= 1.0):
        raise ValueError("--learnability_low/--learnability_high must satisfy 0 <= low < high <= 1.")
    if not (0.0 <= args.curriculum_target_learning_share <= 1.0):
        raise ValueError("--curriculum_target_learning_share must be in [0, 1].")
    if not (0.0 <= args.curriculum_max_adjustment <= 1.0):
        raise ValueError("--curriculum_max_adjustment must be in [0, 1].")
    if resolved_arm in {"B1", "B2"} and args.curriculum_controller != "fixed":
        raise ValueError(f"{resolved_arm} must run with curriculum_controller=fixed for baseline purity.")
    if resolved_arm in {"B1", "B2"} and args.seed_pool_mode != "uniform_levels":
        raise ValueError(f"{resolved_arm} must run with seed_pool_mode=uniform_levels.")
    if resolved_arm == "B3" and args.enable_training:
        raise ValueError("B3 is a status-quo anchor and must disable training.")
    if resolved_arm != "custom":
        resolved_solver_model = args.solver_model_id or args.model
        if resolved_solver_model != PAPER_BASE_MODEL_ID:
            raise ValueError(
                f"{resolved_arm} is pinned to paper base model '{PAPER_BASE_MODEL_ID}'. "
                f"Use --experiment_arm custom for non-paper model experiments."
            )

    solver_backend_name = "dry_run" if args.dry_run else args.solver_backend
    teacher_backend_name = "heuristic" if args.dry_run else args.teacher_backend
    verifier_backend_name = (
        "dry_run"
        if args.dry_run
        else (args.verifier_backend or solver_backend_name)
    )
    needs_tinker_api = (
        solver_backend_name == "tinker"
        or (not args.dry_run and args.mutator_backend == "tinker")
        or teacher_backend_name == "tinker"
        or (args.enable_admission_gate and verifier_backend_name == "tinker")
    )
    if needs_tinker_api and not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set.")

    run_dir = _resolve_log_dir(args.resume_from or args.log_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "run_config.json"
    checkpoint_state_path = run_dir / "checkpoint_state.json"
    epoch_summary_path = run_dir / "epoch_summary.jsonl"
    capability_profiles_path = run_dir / "capability_profiles.jsonl"
    replay_path = run_dir / "replay_entries.jsonl"
    mutator_stats_path = run_dir / "mutator_stats.jsonl"
    mutation_events_path = run_dir / "mutation_events.jsonl"
    kpi_dashboard_path = run_dir / "kpi_dashboard.jsonl"

    replay_buffer = ReplayBuffer(replay_path)
    training_artifacts_path = run_dir / "training_artifacts.jsonl"
    training_replay = TrainingReplayBuffer(training_artifacts_path)
    if teacher_backend_name == "tinker":
        teacher_backend = TinkerLLMTeacherBackend(
            model_id=args.teacher_model_id,
            renderer_name=args.teacher_renderer_name or None,
            temperature=args.teacher_temperature,
            max_tokens=args.teacher_max_tokens,
            request_timeout_s=args.teacher_request_timeout_s,
            fallback_backend=HeuristicTeacherBackend(),
        )
    else:
        teacher_backend = HeuristicTeacherBackend()
    teacher = CurriculumTeacher(
        seed=args.seed,
        policy_backend=teacher_backend,
        target_min_completion=args.learnability_low,
        target_max_completion=args.learnability_high,
    )

    if args.seed_use_mixed_pool:
        if args.seed_pool_mode == "uniform_levels":
            train_tasks = _build_uniform_train_handles(args)
        else:
            train_tasks = _build_mixed_train_handles(args)
    else:
        train_ids = _load_problem_ids(
            args.split_train,
            args.train_subset,
            args.max_train_tasks,
            args.train_problem_ids or None,
        )
        train_tasks = _prepare_task_handles(train_ids, level=args.level_train)

    eval_ids = _load_problem_ids(
        args.split_eval,
        args.eval_subset,
        args.max_eval_tasks,
        args.eval_problem_ids or None,
    )
    eval_tasks = _prepare_task_handles(eval_ids, level=args.level_eval)
    if not train_tasks and args.tasks_per_epoch > 0:
        raise RuntimeError("No training tasks resolved. Check seed level mix and split configuration.")

    sampler_path, training_state_path = _load_solver_paths(args)
    solver_model_id = args.solver_model_id or args.model
    solver_renderer_name = args.solver_renderer_name or args.renderer_name
    if solver_backend_name == "dry_run":
        solver: SolverBackend = DryRunSolverBackend(sampler_path or "dry_run/sampler")
        mutator_backend_primary = DryRunMutatorBackend()
        mutator_backend_frontier = DryRunMutatorBackend()
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
            mutator_backend_primary = TinkerMutatorBackend(
                model_id=args.mutator_model_path,
                renderer_name=args.mutator_renderer_name or None,
                request_timeout_s=args.mutator_request_timeout_s,
            )
            mutator_backend_frontier = TinkerMutatorBackend(
                model_id=args.mutator_frontier_model_path or args.mutator_model_path,
                renderer_name=args.mutator_renderer_name or None,
                request_timeout_s=args.mutator_request_timeout_s,
            )
        else:
            mutator_backend_primary = ApiMutatorBackend(model_id=args.mutator_model_path)
            mutator_backend_frontier = ApiMutatorBackend(
                model_id=args.mutator_frontier_model_path or args.mutator_model_path
            )

    verifier: SolverBackend | None = None
    if args.enable_admission_gate:
        resolved_verifier_backend = verifier_backend_name
        verifier_model_id = args.verifier_model_id or solver_model_id
        verifier_renderer_name = args.verifier_renderer_name or solver_renderer_name
        if resolved_verifier_backend == "dry_run":
            verifier = DryRunSolverBackend("dry_run/verifier")
        else:
            verifier_config = SolverBackendConfig(
                backend=resolved_verifier_backend,
                model_id=verifier_model_id,
                sampler_path=args.verifier_sampler_path,
                renderer_name=verifier_renderer_name,
                training_enabled=False,
                training_state_path="",
                lora_rank=args.lora_rank,
                learning_rate=args.learning_rate,
            )
            verifier = TinkerSolverBackend(config=verifier_config)

    run_config = dict(vars(args))
    run_config["resolved_experiment_arm"] = resolved_arm
    run_config["experiment_arm_policy"] = EXPERIMENT_ARM_POLICY[resolved_arm]
    run_config["experiment_arm_defaults_applied"] = applied_arm_defaults
    run_config["primary_matched_compute_arms"] = list(PRIMARY_MATCHED_COMPUTE_ARMS)
    run_config["status_quo_anchor_arm"] = "B3"
    run_config["phase1_claim_scope"] = PHASE1_CLAIM_SCOPE
    run_config["resolved_solver_backend"] = solver_backend_name
    run_config["resolved_solver_model_id"] = solver.model_id
    run_config["resolved_solver_sampler_path"] = solver.sampler_path
    run_config["resolved_solver_metadata"] = solver.metadata()
    run_config["resolved_teacher_backend"] = teacher_backend.backend_name
    run_config["resolved_teacher_model_id"] = teacher_backend.model_id
    run_config["admission_gate_enabled"] = bool(args.enable_admission_gate)
    run_config["admission_k"] = int(args.admission_k)
    run_config["admission_speedup_threshold"] = float(args.admission_speedup_threshold)
    run_config["admission_bootstrap_passthrough"] = bool(args.admission_bootstrap_passthrough)
    run_config["admission_bootstrap_passthrough_limit"] = int(args.admission_bootstrap_passthrough_limit)
    run_config["resolved_verifier_backend"] = (
        verifier.backend_name if verifier is not None else "disabled"
    )
    run_config["resolved_verifier_model_id"] = (
        verifier.model_id if verifier is not None else ""
    )
    run_config["resolved_verifier_metadata"] = (
        verifier.metadata() if verifier is not None else {}
    )
    run_config["resolved_mutator_backend"] = mutator_backend_primary.backend_name
    run_config["resolved_mutator_model_id"] = mutator_backend_primary.model_id
    run_config["resolved_mutator_frontier_model_id"] = mutator_backend_frontier.model_id
    run_config["resolved_train_level_counts"] = {
        str(level): sum(1 for h in train_tasks if h.level == level)
        for level in sorted({h.level for h in train_tasks})
    }
    run_config["resolved_curriculum_base_quotas"] = _normalize_zone_quotas(
        args.curriculum_mastered_quota,
        args.curriculum_learning_quota,
        args.curriculum_too_hard_quota,
    )
    _write_json(run_config_path, run_config)

    mutator_primary = KernelMutator(
        mutator_backend_primary,
        replay_buffer,
        max_retries=max(1, min(args.mutator_primary_retries, args.mutator_max_retries)),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        semantic_filter=args.mutator_semantic_filter,
        semantic_correct_trials=args.mutator_semantic_correct_trials,
        semantic_perf_trials=args.mutator_semantic_perf_trials,
    )
    mutator_frontier = KernelMutator(
        mutator_backend_frontier,
        replay_buffer,
        max_retries=max(1, min(args.mutator_frontier_retries, args.mutator_max_retries)),
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

    start_epoch = next_epoch
    prev_frontier_size: float | None = None
    prev_too_hard_size: float | None = None
    low_signal_streak = 0
    bootstrap_exit_confirmations = 0
    bootstrap_mode = True
    prior_mastered_categories: set[str] = set()
    prev_agg_fast_1: float | None = None
    train_skip_next = False

    for epoch in range(start_epoch, args.epochs):
        mutator_primary.reset_stats()
        mutator_frontier.reset_stats()
        probe_pool = list(eval_tasks)
        random.Random(args.seed + (epoch * 4049)).shuffle(probe_pool)
        probe_cursor = 0
        epoch_records = 0
        successful_records = 0
        profile_rows, profile_wall = _profile_solver(
            solver,
            eval_tasks,
            profile_k=args.profile_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            eval_workers=args.eval_workers,
        )
        profiles = teacher.update_profile(profile_rows, epoch=epoch, split=args.eval_subset)
        for profile in profiles:
            _append_jsonl(capability_profiles_path, asdict(profile))
        teacher_failure_context = _build_teacher_failure_context(
            replay_buffer,
            profiles,
            recency_window=args.replay_recency_window,
            per_zone_limit=args.teacher_failure_exemplars_per_zone,
        )
        teacher_decision = teacher.select_frontier_target(
            target_min_completion=args.learnability_low,
            target_max_completion=args.learnability_high,
            failure_exemplars=teacher_failure_context,
        )

        ranked_train_tasks = teacher.rank_tasks(train_tasks, strategy=args.teacher_strategy)
        profiles_by_zone = teacher.profiles_by_zone()
        base_quotas = _normalize_zone_quotas(
            args.curriculum_mastered_quota,
            args.curriculum_learning_quota,
            args.curriculum_too_hard_quota,
        )
        observed_zone_counts = _zone_counts_from_profiles(profiles)
        if args.curriculum_controller == "adaptive":
            adjusted_quotas, quota_adjustment = _adaptive_zone_quotas(
                base_quotas,
                observed_zone_counts,
                target_learning_share=args.curriculum_target_learning_share,
                max_adjustment=args.curriculum_max_adjustment,
            )
            quota_adjustment["controller"] = "adaptive"
        else:
            adjusted_quotas = dict(base_quotas)
            total = sum(observed_zone_counts.values())
            learning_share = (
                observed_zone_counts.get(ZONE_LEARNING, 0.0) / total if total > 0 else 0.0
            )
            quota_adjustment = {
                "controller": "fixed",
                "learning_share": learning_share,
                "delta_learning": 0.0,
            }
        slot_count = min(args.tasks_per_epoch, len(ranked_train_tasks))
        desired_zone_counts = _allocate_zone_counts(slot_count, adjusted_quotas)
        zone_plan = _build_zone_plan(desired_zone_counts)
        if len(zone_plan) < slot_count:
            zone_plan.extend([ZONE_LEARNING] * (slot_count - len(zone_plan)))
        zone_plan = zone_plan[:slot_count]
        realized_zone_counts = {ZONE_MASTERED: 0, ZONE_LEARNING: 0, ZONE_TOO_HARD: 0}
        if _bootstrap_mode_active(profiles_by_zone):
            bootstrap_mode = True
            bootstrap_exit_confirmations = 0
        else:
            bootstrap_exit_confirmations += 1
            if bootstrap_exit_confirmations >= args.bootstrap_exit_patience:
                bootstrap_mode = False
                bootstrap_exit_confirmations = 0
        bootstrap_mode_entered = bootstrap_mode
        bootstrap_count = 0
        bootstrap_seed_source_counts: dict[str, int] = {
            "replay_positive": 0,
            "l1_anchor": 0,
            "train_anchor": 0,
        }

        solve_outcomes: list[SolveOutcome] = []
        solve_zones: list[str] = []
        solve_utilities: list[float] = []
        slot_idx = 0
        mini_reprofile_events = 0
        admission_attempted_tasks = 0
        admission_admitted_tasks = 0
        admission_rejected_tasks = 0
        admission_success_rate_sum = 0.0
        admission_gpu_hours = 0.0
        solver_rollouts_skipped_by_admission = 0
        admission_bootstrap_passthrough_used = 0
        bootstrap_guard_rejections = 0

        while slot_idx < slot_count:
            slot_zone = zone_plan[slot_idx] if slot_idx < len(zone_plan) else ZONE_LEARNING
            slot_idx += 1
            slot_profile = _pick_category_for_zone(
                profiles_by_zone,
                slot_zone,
                teacher_decision.target_category,
            )
            bootstrap_seed_source = ""
            if bootstrap_mode:
                effective_zone = ZONE_LEARNING
                preferred_category = (
                    slot_profile.category_id
                    if slot_profile is not None
                    else teacher_decision.target_category
                )
                decision_mode = "learning"
                reason_code = "bootstrap"
                target_speedup_band = BOOTSTRAP_TARGET_SPEEDUP_BAND
                mutation_instruction = (
                    "Bootstrap from a solvable anchor with one local structural change only. "
                    "Preserve interface and keep the same solution family. Avoid transpose/layout "
                    "rewrites, matmul<->einsum rewrites, and new strided reshape patterns."
                )
                seed_task, parent_task_id, seed_problem_id, seed_category, seed_level, bootstrap_seed_source = (
                    _select_bootstrap_seed_task(
                        replay_buffer=replay_buffer,
                        ranked_train_tasks=ranked_train_tasks,
                        replay_recency_window=args.replay_recency_window,
                        bootstrap_index=bootstrap_count,
                    )
                )
                bootstrap_count += 1
                bootstrap_seed_source_counts[bootstrap_seed_source] = (
                    bootstrap_seed_source_counts.get(bootstrap_seed_source, 0) + 1
                )
            else:
                if slot_profile is None:
                    _append_jsonl(
                        mutation_events_path,
                        {
                            "epoch": epoch,
                            "status": "no_eligible_seed_zone",
                            "requested_zone": slot_zone,
                            "bootstrap_mode": False,
                        },
                    )
                    continue
                effective_zone = (
                    slot_profile.zone
                    if slot_profile.zone in {ZONE_MASTERED, ZONE_LEARNING}
                    else slot_zone
                )
                if effective_zone not in {ZONE_MASTERED, ZONE_LEARNING}:
                    _append_jsonl(
                        mutation_events_path,
                        {
                            "epoch": epoch,
                            "status": "skip_too_hard_zone",
                            "requested_zone": slot_zone,
                            "zone": effective_zone,
                            "category_id": slot_profile.category_id,
                            "bootstrap_mode": False,
                        },
                    )
                    continue
                preferred_category = slot_profile.category_id
                decision_mode = _zone_decision_mode(effective_zone)
                reason_code = _zone_reason_code(effective_zone)
                target_speedup_band = _zone_target_speedup_band(effective_zone)
                mutation_instruction = "Generate a valid interface-preserving mutation."
                mutation_instruction = (
                    teacher_decision.mutation_instruction
                    if (
                        effective_zone == ZONE_LEARNING
                        and preferred_category == teacher_decision.target_category
                        and teacher_decision.mutation_instruction
                    )
                    else (
                        "Add one compositional operation while preserving interface."
                        if effective_zone == ZONE_MASTERED
                        else "Generate a structurally harder but learnable mutation."
                    )
                )
                if (
                    effective_zone == ZONE_LEARNING
                    and preferred_category == teacher_decision.target_category
                    and teacher_decision.decision_mode
                ):
                    decision_mode = teacher_decision.decision_mode
                    reason_code = teacher_decision.reason_code or reason_code
                    target_speedup_band = teacher_decision.target_speedup_band or target_speedup_band
                    if teacher_decision.mutation_instruction:
                        mutation_instruction = teacher_decision.mutation_instruction

                seed_task, parent_task_id, seed_problem_id, seed_category, seed_level = _select_seed_task(
                    replay_buffer=replay_buffer,
                    ranked_train_tasks=ranked_train_tasks,
                    profiles=profiles,
                    replay_recency_window=args.replay_recency_window,
                    preferred_category=preferred_category,
                )
            preferred_profile = teacher.latest_profile().get(seed_category)
            solver_trace_summary = (
                f"category={seed_category} zone={preferred_profile.zone} "
                f"correctness={preferred_profile.correctness_rate:.3f} "
                f"fast_1={preferred_profile.fast_1_rate:.3f} "
                f"mean_speedup={preferred_profile.mean_speedup:.3f} "
                f"mean_best_speedup={preferred_profile.mean_best_speedup:.3f} "
                f"utility={preferred_profile.utility_score:.4f}"
                if preferred_profile is not None
                else f"category={seed_category} no_profile"
            )
            seed_failure_exemplars = _select_failure_exemplars(
                replay_buffer,
                category_id=seed_category,
                zone=effective_zone,
                recency_window=args.replay_recency_window,
                limit=args.teacher_seed_failure_exemplars,
            )
            seed_plan = teacher.plan_seed_mutation(
                seed_task=seed_task,
                target_category=seed_category,
                zone=effective_zone,
                target_speedup_band=target_speedup_band,
                solver_trace_summary=solver_trace_summary,
                failure_exemplars=seed_failure_exemplars,
            )
            decision_mode = seed_plan.decision_mode or decision_mode
            reason_code = seed_plan.reason_code or reason_code
            target_speedup_band = seed_plan.target_speedup_band or target_speedup_band
            mutation_instruction = seed_plan.mutation_instruction or mutation_instruction
            if bootstrap_mode:
                decision_mode = "learning"
                reason_code = "bootstrap"
                target_speedup_band = BOOTSTRAP_TARGET_SPEEDUP_BAND
                mutation_instruction = (
                    "Bootstrap: apply one local in-family change only; keep interface and semantics "
                    "stable; avoid transpose/layout rewrites, matmul<->einsum rewrites, and new "
                    "strided reshape/batching patterns."
                )
            teacher_seed_rationale = seed_plan.rationale
            mutator_target_category = seed_category
            use_frontier_mutator = decision_mode == "too_hard_decompose" or bootstrap_mode
            if use_frontier_mutator:
                mutated = mutator_frontier.mutate(
                    seed_task,
                    epoch=epoch,
                    seed_problem_id=seed_problem_id,
                    target_category=mutator_target_category,
                    level=seed_level,
                    target_speedup_band=target_speedup_band,
                    failure_exemplars=seed_failure_exemplars,
                    solver_trace_summary=solver_trace_summary,
                    mutation_instruction=mutation_instruction,
                    decision_mode=decision_mode,
                    reason_code=reason_code,
                    teacher_seed_rationale=teacher_seed_rationale,
                )
                if mutated is None:
                    mutated = mutator_primary.mutate(
                        seed_task,
                        epoch=epoch,
                        seed_problem_id=seed_problem_id,
                        target_category=mutator_target_category,
                        level=seed_level,
                        target_speedup_band=target_speedup_band,
                        failure_exemplars=seed_failure_exemplars,
                        solver_trace_summary=solver_trace_summary,
                        mutation_instruction=mutation_instruction,
                        decision_mode=decision_mode,
                        reason_code=reason_code,
                        teacher_seed_rationale=teacher_seed_rationale,
                    )
            else:
                mutated = mutator_primary.mutate(
                    seed_task,
                    epoch=epoch,
                    seed_problem_id=seed_problem_id,
                    target_category=mutator_target_category,
                    level=seed_level,
                    target_speedup_band=target_speedup_band,
                    failure_exemplars=seed_failure_exemplars,
                    solver_trace_summary=solver_trace_summary,
                    mutation_instruction=mutation_instruction,
                    decision_mode=decision_mode,
                    reason_code=reason_code,
                    teacher_seed_rationale=teacher_seed_rationale,
                )
                if mutated is None:
                    mutated = mutator_frontier.mutate(
                        seed_task,
                        epoch=epoch,
                        seed_problem_id=seed_problem_id,
                        target_category=mutator_target_category,
                        level=seed_level,
                        target_speedup_band=target_speedup_band,
                        failure_exemplars=seed_failure_exemplars,
                        solver_trace_summary=solver_trace_summary,
                        mutation_instruction=mutation_instruction,
                        decision_mode=decision_mode,
                        reason_code=reason_code,
                        teacher_seed_rationale=teacher_seed_rationale,
                    )
            if mutated is None:
                _append_jsonl(
                    mutation_events_path,
                    {
                        "epoch": epoch,
                        "status": "mutation_failed",
                        "seed_problem_id": seed_problem_id,
                        "seed_category": seed_category,
                        "requested_zone": slot_zone,
                        "zone": effective_zone,
                        "decision_mode": decision_mode,
                        "reason_code": reason_code,
                        "target_speedup_band": list(target_speedup_band),
                        "mutation_instruction": mutation_instruction,
                        "solver_trace_summary": solver_trace_summary,
                        "teacher_seed_rationale": teacher_seed_rationale,
                        "failure_exemplar_ids": [
                            str(row.get("entry_id", "")) for row in seed_failure_exemplars
                        ],
                        "failure_exemplar_count": len(seed_failure_exemplars),
                        "bootstrap_mode": bootstrap_mode,
                        "bootstrap_seed_source": bootstrap_seed_source,
                    },
                )
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
                    level=seed_level,
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

            if bootstrap_mode:
                guard_ok, guard_reason = _passes_bootstrap_mutation_guard(
                    seed_task.reference_code,
                    mutated.reference_code,
                )
                if not guard_ok:
                    bootstrap_guard_rejections += 1
                    _append_jsonl(
                        mutation_events_path,
                        {
                            "epoch": epoch,
                            "status": "bootstrap_guard_rejected",
                            "task_id": mutated.task_id,
                            "seed_problem_id": seed_problem_id,
                            "seed_category": seed_category,
                            "requested_zone": slot_zone,
                            "zone": effective_zone,
                            "guard_reason": guard_reason,
                            "mutation_type": mutated.mutation_type,
                            "mutation_backend": mutated.mutation_backend,
                            "mutation_model_id": mutated.mutation_model_id,
                            "bootstrap_mode": True,
                            "bootstrap_seed_source": bootstrap_seed_source,
                        },
                    )
                    bank_eval = EvalResult(
                        compiled=False,
                        correctness=False,
                        runtime_us=-1.0,
                        ref_runtime_us=-1.0,
                        speedup=0.0,
                        metadata={"banked": True, "reason": "bootstrap_guard_rejected"},
                    )
                    replay_buffer.append(
                        ReplayEntry(
                            entry_id=f"bootstrap_guard_reject_{mutated.task_id}_{time.time_ns()}",
                            task_id=mutated.task_id,
                            parent_task_id=parent_task_id,
                            problem_id=seed_problem_id,
                            level=seed_level,
                            category_id=mutated.category_id,
                            task_reference_code=mutated.reference_code,
                            kernel_code="",
                            eval_result=bank_eval,
                            reward=0.0,
                            sampler_path=solver.sampler_path,
                            backend="bootstrap_guard",
                            timestamp=time.time(),
                            epoch=epoch,
                            is_mutated=True,
                        )
                    )
                    continue

            realized_zone_counts[effective_zone] = realized_zone_counts.get(effective_zone, 0) + 1
            _append_jsonl(
                mutation_events_path,
                {
                    "epoch": epoch,
                    "status": "mutated",
                    "task_id": mutated.task_id,
                    "seed_problem_id": seed_problem_id,
                    "seed_category": seed_category,
                    "requested_zone": slot_zone,
                    "zone": effective_zone,
                    "decision_mode": mutated.teacher_decision_mode,
                    "reason_code": mutated.teacher_reason_code,
                    "target_speedup_band": list(mutated.teacher_target_speedup_band),
                    "mutation_instruction": mutated.teacher_mutation_instruction,
                    "solver_trace_summary": mutated.solver_trace_summary,
                    "teacher_seed_rationale": mutated.teacher_seed_rationale,
                    "failure_exemplar_ids": list(mutated.teacher_failure_entry_ids),
                    "failure_exemplar_count": len(mutated.teacher_failure_entry_ids),
                    "mutation_type": mutated.mutation_type,
                    "mutation_backend": mutated.mutation_backend,
                    "mutation_model_id": mutated.mutation_model_id,
                    "bootstrap_mode": bootstrap_mode,
                    "bootstrap_seed_source": bootstrap_seed_source,
                },
            )

            task_for_solver = KernelTask(
                problem_id=seed_problem_id,
                name=mutated.name,
                reference_code=mutated.reference_code,
            )
            if args.enable_admission_gate:
                if verifier is None:
                    raise RuntimeError("Admission gate enabled but verifier backend not initialized.")
                admission_result = _run_admission_gate(
                    verifier,
                    task_for_solver,
                    k=args.admission_k,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    level=seed_level,
                    eval_workers=args.eval_workers,
                    speedup_threshold=args.admission_speedup_threshold,
                )
                admission_attempted_tasks += 1
                admission_success_rate_sum += float(admission_result["success_rate"])
                admission_gpu_hours += float(admission_result["wall_clock_s"]) / 3600.0
                if not bool(admission_result["admitted"]):
                    allow_bootstrap_passthrough = (
                        bootstrap_mode
                        and args.admission_bootstrap_passthrough
                        and admission_bootstrap_passthrough_used
                        < args.admission_bootstrap_passthrough_limit
                    )
                    if allow_bootstrap_passthrough:
                        admission_bootstrap_passthrough_used += 1
                        _append_jsonl(
                            mutation_events_path,
                            {
                                "epoch": epoch,
                                "status": "admission_passthrough",
                                "task_id": mutated.task_id,
                                "seed_problem_id": seed_problem_id,
                                "seed_category": seed_category,
                                "zone": effective_zone,
                                "admission_success_count": int(admission_result["success_count"]),
                                "admission_total_count": int(admission_result["total_count"]),
                                "admission_success_rate": float(admission_result["success_rate"]),
                                "admission_speedup_threshold": float(args.admission_speedup_threshold),
                                "bootstrap_mode": bootstrap_mode,
                                "bootstrap_seed_source": bootstrap_seed_source,
                                "passthrough_limit": int(args.admission_bootstrap_passthrough_limit),
                                "passthrough_used": int(admission_bootstrap_passthrough_used),
                            },
                        )
                    else:
                        admission_rejected_tasks += 1
                        solver_rollouts_skipped_by_admission += 1
                        _append_jsonl(
                            mutation_events_path,
                            {
                                "epoch": epoch,
                                "status": "admission_rejected",
                                "task_id": mutated.task_id,
                                "seed_problem_id": seed_problem_id,
                                "seed_category": seed_category,
                                "zone": effective_zone,
                                "admission_success_count": int(admission_result["success_count"]),
                                "admission_total_count": int(admission_result["total_count"]),
                                "admission_success_rate": float(admission_result["success_rate"]),
                                "admission_speedup_threshold": float(args.admission_speedup_threshold),
                                "bootstrap_mode": bootstrap_mode,
                                "bootstrap_seed_source": bootstrap_seed_source,
                            },
                        )
                        bank_eval = EvalResult(
                            compiled=False,
                            correctness=False,
                            runtime_us=-1.0,
                            ref_runtime_us=-1.0,
                            speedup=0.0,
                            metadata={
                                "banked": True,
                                "reason": "admission_rejected",
                                "admission_success_count": int(admission_result["success_count"]),
                                "admission_total_count": int(admission_result["total_count"]),
                                "admission_success_rate": float(admission_result["success_rate"]),
                                "admission_speedup_threshold": float(args.admission_speedup_threshold),
                            },
                        )
                        replay_buffer.append(
                            ReplayEntry(
                                entry_id=f"admission_reject_{mutated.task_id}_{time.time_ns()}",
                                task_id=mutated.task_id,
                                parent_task_id=parent_task_id,
                                problem_id=seed_problem_id,
                                level=seed_level,
                                category_id=mutated.category_id,
                                task_reference_code=mutated.reference_code,
                                kernel_code="",
                                eval_result=bank_eval,
                                reward=0.0,
                                sampler_path=solver.sampler_path,
                                backend="admission_gate",
                                timestamp=time.time(),
                                epoch=epoch,
                                is_mutated=True,
                            )
                        )
                        continue
                admission_admitted_tasks += 1

            solve_outcome = solver.solve_task(
                task_for_solver,
                k=args.k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                level=seed_level,
                eval_workers=args.eval_workers,
            )
            solve_outcomes.append(solve_outcome)
            solve_zones.append(effective_zone)
            solve_utilities.append(
                preferred_profile.utility_score if preferred_profile is not None else 0.0
            )
            outcome_entry_ids: list[str] = []
            for idx, (kernel_code, eval_result, reward) in enumerate(
                zip(solve_outcome.kernel_codes, solve_outcome.eval_results, solve_outcome.rewards)
            ):
                kernel_hash = hashlib.sha256(kernel_code.encode("utf-8")).hexdigest()[:16]
                entry_id = f"{mutated.task_id}_{idx}_{kernel_hash}"
                outcome_entry_ids.append(entry_id)
                replay_entry = ReplayEntry(
                    entry_id=entry_id,
                    task_id=mutated.task_id,
                    parent_task_id=parent_task_id,
                    problem_id=seed_problem_id,
                    level=seed_level,
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
            if args.enable_training and solve_outcome.sampled_tokens:
                training_replay.add_outcome(
                    outcome_id=f"{mutated.task_id}_{epoch}",
                    epoch=epoch,
                    zone=effective_zone,
                    utility_score=(
                        preferred_profile.utility_score if preferred_profile is not None else 0.0
                    ),
                    category_id=mutated.category_id,
                    problem_id=seed_problem_id,
                    level=seed_level,
                    prompt_tokens=solve_outcome.prompt_tokens or [],
                    sampled_tokens_list=solve_outcome.sampled_tokens,
                    sampled_logprobs_list=solve_outcome.sampled_logprobs,
                    rewards=solve_outcome.rewards,
                    entry_ids=outcome_entry_ids,
                    sampler_path=solver.sampler_path,
                )

            should_mini_reprofile = (
                args.mini_reprofile_every > 0
                and slot_idx < slot_count
                and (slot_idx % args.mini_reprofile_every == 0)
            )
            if should_mini_reprofile:
                probe_tasks, probe_cursor = _next_probe_tasks(
                    probe_pool,
                    cursor=probe_cursor,
                    probe_tasks=args.mini_reprofile_probe_tasks,
                )
                if probe_tasks:
                    mini_rows, mini_wall = _profile_solver(
                        solver,
                        probe_tasks,
                        profile_k=args.mini_reprofile_k,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        eval_workers=args.eval_workers,
                    )
                    profile_rows.extend(mini_rows)
                    profiles = teacher.update_profile(
                        profile_rows,
                        epoch=epoch,
                        split=f"{args.eval_subset}_rolling",
                    )
                    for profile in profiles:
                        _append_jsonl(capability_profiles_path, asdict(profile))
                    teacher_failure_context = _build_teacher_failure_context(
                        replay_buffer,
                        profiles,
                        recency_window=args.replay_recency_window,
                        per_zone_limit=args.teacher_failure_exemplars_per_zone,
                    )
                    teacher_decision = teacher.select_frontier_target(
                        target_min_completion=args.learnability_low,
                        target_max_completion=args.learnability_high,
                        failure_exemplars=teacher_failure_context,
                    )
                    ranked_train_tasks = teacher.rank_tasks(
                        train_tasks,
                        strategy=args.teacher_strategy,
                    )
                    profiles_by_zone = teacher.profiles_by_zone()
                    if bootstrap_mode and not _bootstrap_mode_active(profiles_by_zone):
                        bootstrap_exit_confirmations += 1
                        if bootstrap_exit_confirmations >= args.bootstrap_exit_patience:
                            bootstrap_mode = False
                            bootstrap_exit_confirmations = 0
                    elif bootstrap_mode:
                        bootstrap_exit_confirmations = 0
                    observed_zone_counts = _zone_counts_from_profiles(profiles)
                    if args.curriculum_controller == "adaptive":
                        adjusted_quotas, quota_adjustment = _adaptive_zone_quotas(
                            base_quotas,
                            observed_zone_counts,
                            target_learning_share=args.curriculum_target_learning_share,
                            max_adjustment=args.curriculum_max_adjustment,
                        )
                        quota_adjustment["controller"] = "adaptive"
                    else:
                        adjusted_quotas = dict(base_quotas)
                        total = sum(observed_zone_counts.values())
                        learning_share = (
                            observed_zone_counts.get(ZONE_LEARNING, 0.0) / total if total > 0 else 0.0
                        )
                        quota_adjustment = {
                            "controller": "fixed",
                            "learning_share": learning_share,
                            "delta_learning": 0.0,
                        }
                    desired_zone_counts = _allocate_zone_counts(slot_count, adjusted_quotas)
                    remaining_slots = slot_count - slot_idx
                    if remaining_slots > 0:
                        remaining_zone_counts = _allocate_zone_counts(remaining_slots, adjusted_quotas)
                        remaining_plan = _build_zone_plan(remaining_zone_counts)
                        if len(remaining_plan) < remaining_slots:
                            remaining_plan.extend([ZONE_LEARNING] * (remaining_slots - len(remaining_plan)))
                        zone_plan = zone_plan[:slot_idx] + remaining_plan[:remaining_slots]
                    mini_reprofile_events += 1

        training_activity: dict[str, Any] = {
            "attempted": bool(args.enable_training),
            "executed": False,
            "datum_count": 0,
            "outcomes_used": 0,
            "current_frontier_count": 0,
            "replay_count": 0,
            "current_fraction": 0.0,
            "replay_fraction_actual": 0.0,
            "mean_datum_weight": 0.0,
            "replay_epochs_represented": [],
        }
        train_gated = (
            args.enable_training
            and epoch >= args.min_discovery_epochs
            and not train_skip_next
        )
        if train_skip_next:
            train_skip_next = False
        if train_gated:
            mixed_outcomes, datum_weights = _build_training_batch(
                current_outcomes=solve_outcomes,
                current_zones=solve_zones,
                current_utilities=solve_utilities,
                training_replay=training_replay,
                current_epoch=epoch,
                replay_fraction=args.replay_fraction,
                recency_epochs=args.replay_recency_epochs,
                decay_rate=args.replay_decay_rate,
                std_threshold=args.effective_task_std_threshold,
            )
            n_current = len(solve_outcomes)
            n_replay = max(0, len(mixed_outcomes) - n_current)
            n_total = len(mixed_outcomes)
            replay_epoch_set = sorted({
                a.epoch
                for a in training_replay.sample_replay(
                    current_epoch=epoch,
                    recency_epochs=args.replay_recency_epochs,
                    zone_filter=ZONE_LEARNING,
                )
            })
            subsample = args.train_subsample_per_outcome or None
            train_result = solver.train_on_outcomes(
                mixed_outcomes,
                epoch=epoch,
                datum_weights=datum_weights,
                max_samples_per_outcome=subsample,
            )
            solver.sampler_path = train_result.sampler_path
            training_activity = {
                "attempted": True,
                "executed": bool(train_result.training_executed),
                "datum_count": int(train_result.datum_count),
                "outcomes_used": int(train_result.outcomes_used),
                "current_frontier_count": n_current,
                "replay_count": n_replay,
                "current_fraction": n_current / n_total if n_total > 0 else 0.0,
                "replay_fraction_actual": n_replay / n_total if n_total > 0 else 0.0,
                "mean_datum_weight": (
                    statistics.mean(datum_weights) if datum_weights else 0.0
                ),
                "replay_epochs_represented": replay_epoch_set,
            }
        forgetting = _compute_forgetting_indicator(profiles_by_zone, prior_mastered_categories)
        training_activity["forgetting_indicator"] = forgetting
        learning_zone_profiles = profiles_by_zone.get(ZONE_LEARNING, [])
        training_activity["learning_zone_fraction"] = (
            len(learning_zone_profiles)
            / max(1, sum(len(v) for v in profiles_by_zone.values()))
        )

        signal_stats = _gradient_signal_summary(
            solve_outcomes,
            std_threshold=args.effective_task_std_threshold,
        )
        effective_tasks = int(signal_stats["effective_tasks"])
        zero_signal_tasks = int(signal_stats["zero_signal_tasks"])
        required_effective_tasks = min(args.min_effective_tasks, len(solve_outcomes))
        if effective_tasks < required_effective_tasks:
            low_signal_streak += 1
        else:
            low_signal_streak = 0
        signal_stats["required_effective_tasks"] = float(required_effective_tasks)
        signal_stats["low_signal_streak"] = float(low_signal_streak)

        aggregate_fast_1 = (
            statistics.mean([float(row.get("fast_1", 0.0)) for row in profile_rows]) if profile_rows else 0.0
        )
        rollback_event: dict[str, Any] | None = None
        if (
            args.rollback_threshold > 0
            and prev_agg_fast_1 is not None
            and prev_agg_fast_1 > 0
        ):
            drop_frac = (prev_agg_fast_1 - aggregate_fast_1) / prev_agg_fast_1
            if drop_frac > args.rollback_threshold:
                solver.learning_rate /= 2.0
                train_skip_next = True
                rollback_event = {
                    "epoch": epoch,
                    "prev_agg_fast_1": prev_agg_fast_1,
                    "current_agg_fast_1": aggregate_fast_1,
                    "drop_fraction": drop_frac,
                    "new_learning_rate": solver.learning_rate,
                    "action": "skip_next_train_and_halve_lr",
                }
        prev_agg_fast_1 = aggregate_fast_1
        frontier_size = observed_zone_counts.get(ZONE_LEARNING, 0.0)
        too_hard_size = observed_zone_counts.get(ZONE_TOO_HARD, 0.0)
        frontier_delta = (
            0.0 if prev_frontier_size is None else frontier_size - prev_frontier_size
        )
        too_hard_reduction = (
            0.0 if prev_too_hard_size is None else (prev_too_hard_size - too_hard_size)
        )
        kpi_payload = {
            "epoch": epoch,
            "aggregate_fast_1": aggregate_fast_1,
            "frontier_size": frontier_size,
            "frontier_size_delta": frontier_delta,
            "too_hard_size": too_hard_size,
            "too_hard_reduction": too_hard_reduction,
            "effective_tasks": effective_tasks,
            "zero_signal_tasks": zero_signal_tasks,
            "required_effective_tasks": required_effective_tasks,
            "signal_task_rate": signal_stats["effective_task_rate"],
            "low_signal_streak": low_signal_streak,
            "mini_reprofile_events": mini_reprofile_events,
            "admission_attempted_tasks": admission_attempted_tasks,
            "admission_admitted_tasks": admission_admitted_tasks,
            "admission_rejected_tasks": admission_rejected_tasks,
            "admission_reject_rate": (
                admission_rejected_tasks / admission_attempted_tasks
                if admission_attempted_tasks > 0
                else 0.0
            ),
            "admission_mean_success_rate": (
                admission_success_rate_sum / admission_attempted_tasks
                if admission_attempted_tasks > 0
                else 0.0
            ),
            "admission_gpu_hours": admission_gpu_hours,
            "solver_rollouts_skipped_by_admission": solver_rollouts_skipped_by_admission,
            "admission_bootstrap_passthrough_used": admission_bootstrap_passthrough_used,
            "bootstrap_guard_rejections": bootstrap_guard_rejections,
            "estimated_solver_compute_saved_samples": (
                solver_rollouts_skipped_by_admission * max(0, int(args.k))
            ),
            "training_attempted": training_activity["attempted"],
            "training_executed": training_activity["executed"],
            "training_datum_count": training_activity["datum_count"],
            "training_outcomes_used": training_activity["outcomes_used"],
            "training_current_frontier_count": training_activity["current_frontier_count"],
            "training_replay_count": training_activity["replay_count"],
            "training_replay_fraction_actual": training_activity["replay_fraction_actual"],
            "training_mean_datum_weight": training_activity["mean_datum_weight"],
            "training_replay_epochs_represented": training_activity["replay_epochs_represented"],
            "bootstrap_mode_entered": bootstrap_mode_entered,
            "bootstrap_mode_active_end": bootstrap_mode,
            "bootstrap_count": bootstrap_count,
            "bootstrap_seed_source_counts": dict(bootstrap_seed_source_counts),
            "rollback_event": rollback_event,
        }
        forgetting = _compute_forgetting_indicator(profiles_by_zone, prior_mastered_categories)
        kpi_payload["forgetting_indicator"] = forgetting
        current_mastered = {p.category_id for p in profiles_by_zone.get(ZONE_MASTERED, [])}
        prior_mastered_categories = prior_mastered_categories | current_mastered
        _append_jsonl(kpi_dashboard_path, kpi_payload)
        prev_frontier_size = frontier_size
        prev_too_hard_size = too_hard_size

        checkpoint_payload = {
            "next_epoch": epoch + 1,
            "current_sampler_path": solver.sampler_path,
            "timestamp": time.time(),
        }
        _write_json(checkpoint_state_path, checkpoint_payload)
        _append_jsonl(
            mutator_stats_path,
            {
                "epoch": epoch,
                "primary": mutator_primary.stats.as_dict(),
                "frontier": mutator_frontier.stats.as_dict(),
            },
        )
        _append_jsonl(
            epoch_summary_path,
            {
                "epoch": epoch,
                "experiment_arm": resolved_arm,
                "records_added": epoch_records,
                "records_successful": successful_records,
                "success_rate": (successful_records / epoch_records) if epoch_records else 0.0,
                "sampler_path": solver.sampler_path,
                "teacher_decision": asdict(teacher_decision),
                "phase1_claim_scope": PHASE1_CLAIM_SCOPE,
                "curriculum": {
                    "controller": args.curriculum_controller,
                    "base_quotas": base_quotas,
                    "adjusted_quotas": adjusted_quotas,
                    "observed_zone_counts": observed_zone_counts,
                    "quota_adjustment": quota_adjustment,
                    "desired_zone_counts": desired_zone_counts,
                    "realized_zone_counts": realized_zone_counts,
                    "realized_zone_pct": {
                        zone: (count / max(1, sum(realized_zone_counts.values())))
                        for zone, count in realized_zone_counts.items()
                    },
                    "teacher_failure_context_count": len(teacher_failure_context),
                    "mini_reprofile_events": mini_reprofile_events,
                    "admission_gate_enabled": bool(args.enable_admission_gate),
                    "admission_attempted_tasks": admission_attempted_tasks,
                    "admission_admitted_tasks": admission_admitted_tasks,
                    "admission_rejected_tasks": admission_rejected_tasks,
                    "admission_reject_rate": (
                        admission_rejected_tasks / admission_attempted_tasks
                        if admission_attempted_tasks > 0
                        else 0.0
                    ),
                    "admission_bootstrap_passthrough_used": admission_bootstrap_passthrough_used,
                    "bootstrap_guard_rejections": bootstrap_guard_rejections,
                    "training_attempted": training_activity["attempted"],
                    "training_executed": training_activity["executed"],
                    "training_datum_count": training_activity["datum_count"],
                    "training_outcomes_used": training_activity["outcomes_used"],
                    "training_current_frontier_count": training_activity["current_frontier_count"],
                    "training_replay_count": training_activity["replay_count"],
                    "training_replay_fraction_actual": training_activity["replay_fraction_actual"],
                    "training_mean_datum_weight": training_activity["mean_datum_weight"],
                    "learning_zone_fraction": training_activity.get("learning_zone_fraction", 0.0),
                    "forgetting_indicator": forgetting,
                    "bootstrap_mode_entered": bootstrap_mode_entered,
                    "bootstrap_mode_active_end": bootstrap_mode,
                    "bootstrap_count": bootstrap_count,
                    "bootstrap_seed_source_counts": dict(bootstrap_seed_source_counts),
                },
                "gradient_signal": {
                    "effective_tasks": effective_tasks,
                    "zero_signal_tasks": zero_signal_tasks,
                    "required_effective_tasks": required_effective_tasks,
                    "effective_task_rate": signal_stats["effective_task_rate"],
                    "std_threshold": args.effective_task_std_threshold,
                    "low_signal_streak": low_signal_streak,
                },
                "kpi": kpi_payload,
                "mutator_stats": {
                    "primary": mutator_primary.stats.as_dict(),
                    "frontier": mutator_frontier.stats.as_dict(),
                },
            },
        )

        if (
            required_effective_tasks > 0
            and low_signal_streak >= args.min_effective_tasks_patience
        ):
            _append_jsonl(
                epoch_summary_path,
                {
                    "epoch": epoch,
                    "status": "halted_low_gradient_signal",
                    "experiment_arm": resolved_arm,
                    "effective_tasks": effective_tasks,
                    "required_effective_tasks": required_effective_tasks,
                    "zero_signal_tasks": zero_signal_tasks,
                    "low_signal_streak": low_signal_streak,
                    "min_effective_tasks_patience": args.min_effective_tasks_patience,
                },
            )
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
