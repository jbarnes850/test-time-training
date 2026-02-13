from __future__ import annotations

import ast
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.env.schema import CapabilityProfile


_MATMUL_PATTERNS = ("matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "gemm")
_CONV_PATTERNS = ("conv", "conv1d", "conv2d", "conv3d", "convtranspose")
_ACTIVATION_PATTERNS = (
    "relu",
    "leakyrelu",
    "hardtanh",
    "sigmoid",
    "tanh",
    "softmax",
    "gelu",
    "silu",
    "mish",
    "elu",
    "selu",
    "logsigmoid",
)
_NORMALIZATION_PATTERNS = (
    "batchnorm",
    "layernorm",
    "instancenorm",
    "groupnorm",
    "rmsnorm",
    "normalize",
)
_POOLING_PATTERNS = ("maxpool", "avgpool", "adaptiveavgpool", "adaptivemaxpool")
_REDUCTION_PATTERNS = (
    "sum",
    "mean",
    "prod",
    "cumsum",
    "cumprod",
    "amin",
    "amax",
    "max",
    "min",
    "logsumexp",
    "argmax",
    "argmin",
)
_LOSS_PATTERNS = (
    "crossentropy",
    "nllloss",
    "mseloss",
    "l1loss",
    "smoothl1loss",
    "hinge",
    "binarycrossentropy",
    "bce",
    "kl_div",
)


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


def _matches_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    lname = name.lower()
    return any(p in lname for p in patterns)


def infer_task_categories(reference_code: str) -> set[str]:
    try:
        tree = ast.parse(reference_code)
    except SyntaxError:
        return {"unknown"}

    tags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if not call_name:
            continue

        if _matches_pattern(call_name, _MATMUL_PATTERNS):
            tags.add("matmul")
        if _matches_pattern(call_name, _CONV_PATTERNS):
            tags.add("conv")
        if _matches_pattern(call_name, _ACTIVATION_PATTERNS):
            tags.add("activation")
        if _matches_pattern(call_name, _NORMALIZATION_PATTERNS):
            tags.add("normalization")
        if _matches_pattern(call_name, _POOLING_PATTERNS):
            tags.add("pooling")
        if _matches_pattern(call_name, _REDUCTION_PATTERNS):
            tags.add("reduction")
        if _matches_pattern(call_name, _LOSS_PATTERNS):
            tags.add("loss")

    return tags or {"unknown"}


def category_id(tags: set[str]) -> str:
    if not tags:
        return "unknown"
    ordered = sorted(tags)
    if ordered == ["unknown"]:
        return "unknown"
    if len(ordered) == 1:
        return ordered[0]
    return f"composite:{'+'.join(ordered)}"


def _fast_1_from_row(row: dict[str, Any]) -> float:
    if "fast_1" in row:
        return float(row["fast_1"])
    correctness = bool(row["correctness"])
    speedup = float(row["speedup"])
    return 1.0 if correctness and speedup > 1.0 else 0.0


def _task_category(task: Any) -> str:
    if isinstance(task, dict):
        return str(task.get("category_id", "unknown"))
    return str(getattr(task, "category_id", "unknown"))


@dataclass
class CurriculumTeacher:
    seed: int = 42
    _rng: random.Random = field(init=False, repr=False)
    _latest_profile: dict[str, CapabilityProfile] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def update_profile(
        self,
        eval_rows: list[dict[str, Any]],
        epoch: int,
        split: str = "eval",
    ) -> list[CapabilityProfile]:
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eval_rows:
            by_category[str(row.get("category_id", "unknown"))].append(row)

        profiles: list[CapabilityProfile] = []
        for cat in sorted(by_category.keys()):
            rows = by_category[cat]
            speedups = [float(r["speedup"]) for r in rows]
            correctness = [1.0 if bool(r["correctness"]) else 0.0 for r in rows]
            fast_1_values = [_fast_1_from_row(r) for r in rows]
            task_ids = {str(r.get("task_id", f"row_{idx}")) for idx, r in enumerate(rows)}

            profile = CapabilityProfile(
                epoch=epoch,
                split=split,
                category_id=cat,
                n_tasks=len(task_ids),
                correctness_rate=sum(correctness) / len(correctness),
                mean_speedup=statistics.mean(speedups),
                speedup_var=statistics.pvariance(speedups) if len(speedups) > 1 else 0.0,
                fast_1_rate=sum(fast_1_values) / len(fast_1_values),
                failure_rate=1.0 - (sum(correctness) / len(correctness)),
                sample_count=len(rows),
            )
            profiles.append(profile)

        self._latest_profile = {p.category_id: p for p in profiles}
        return profiles

    def rank_tasks(
        self,
        tasks: list[Any],
        strategy: str = "inverse_correctness",
    ) -> list[Any]:
        if strategy == "random":
            ranked = list(tasks)
            self._rng.shuffle(ranked)
            return ranked

        def _profile_for(task: Any) -> CapabilityProfile | None:
            return self._latest_profile.get(_task_category(task))

        if strategy == "easy_to_hard_static":
            return sorted(
                tasks,
                key=lambda t: (
                    -((_profile_for(t).correctness_rate) if _profile_for(t) else 0.0),
                    _task_category(t),
                ),
            )

        if strategy == "loss_proportional":
            return sorted(
                tasks,
                key=lambda t: (
                    -self._loss_score(_profile_for(t)),
                    _task_category(t),
                ),
            )

        if strategy == "inverse_correctness":
            return sorted(
                tasks,
                key=lambda t: (
                    -self._inverse_correctness_score(_profile_for(t)),
                    _task_category(t),
                ),
            )

        raise ValueError(f"Unknown teacher strategy: {strategy}")

    @staticmethod
    def _inverse_correctness_score(profile: CapabilityProfile | None) -> float:
        if profile is None:
            return 1.0
        return max(0.0, 1.0 - profile.correctness_rate)

    @staticmethod
    def _loss_score(profile: CapabilityProfile | None) -> float:
        if profile is None:
            return 1.0
        correctness_term = max(0.0, 1.0 - profile.correctness_rate)
        speed_term = max(0.0, 1.0 - profile.mean_speedup)
        return correctness_term + speed_term
