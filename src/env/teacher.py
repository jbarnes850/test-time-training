from __future__ import annotations

import ast
import json
import math
import random
import re
import statistics
from collections import defaultdict
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.env.schema import CapabilityProfile, KernelTask
from src.utils.tinker_utils import ensure_tinker_cookbook_on_path


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

ZONE_MASTERED = "mastered"
ZONE_LEARNING = "learning"
ZONE_TOO_HARD = "too_hard"
ZONE_UNKNOWN = "unknown"
ZONE_ORDER = (ZONE_LEARNING, ZONE_TOO_HARD, ZONE_MASTERED, ZONE_UNKNOWN)

DECISION_MODE_LEARNING = "learning"
DECISION_MODE_MASTERED_WARMUP = "mastered_warmup"
DECISION_MODE_TOO_HARD_DECOMPOSE = "too_hard_decompose"
DECISION_MODE_FALLBACK = "fallback"
VALID_DECISION_MODES = {
    DECISION_MODE_LEARNING,
    DECISION_MODE_MASTERED_WARMUP,
    DECISION_MODE_TOO_HARD_DECOMPOSE,
    DECISION_MODE_FALLBACK,
}

VALID_REASON_CODES = {
    "edge_signal",
    "max_variance",
    "data_sparse",
    "decompose",
    "warmup",
    "fallback",
}


def classify_task_zone(
    task_rows: list[dict[str, Any]],
    *,
    pass_speedup_threshold: float = 1.5,
    mastered_speedup_threshold: float = 2.0,
    too_hard_speedup_ceiling: float = 1.1,
) -> str:
    if not task_rows:
        return ZONE_UNKNOWN

    k = len(task_rows)
    pass_count = 0
    strong_count = 0
    correct_speedups: list[float] = []
    fast_1 = False
    for row in task_rows:
        correctness = bool(row.get("correctness", False))
        speedup = float(row.get("speedup", 0.0))
        if correctness:
            correct_speedups.append(speedup)
            if speedup > 1.0:
                fast_1 = True
        if correctness and speedup > pass_speedup_threshold:
            pass_count += 1
        if correctness and speedup > mastered_speedup_threshold:
            strong_count += 1

    if strong_count == k and k > 0:
        return ZONE_MASTERED

    max_correct_speedup = max(correct_speedups) if correct_speedups else 0.0
    if fast_1 and max_correct_speedup < too_hard_speedup_ceiling:
        return ZONE_TOO_HARD
    if pass_count == 0 and correct_speedups:
        return ZONE_TOO_HARD
    if pass_count == 0 and not correct_speedups:
        return ZONE_TOO_HARD

    if 0 < pass_count < k:
        return ZONE_LEARNING
    if pass_count == k:
        mean_correct_speedup = (
            statistics.mean(correct_speedups) if correct_speedups else 0.0
        )
        if mean_correct_speedup >= mastered_speedup_threshold:
            return ZONE_MASTERED
        return ZONE_LEARNING
    return ZONE_LEARNING


def task_frontier_utility(
    task_rows: list[dict[str, Any]],
    *,
    mu: float = 1.5,
    sigma: float = 0.5,
    pass_speedup_threshold: float = 1.5,
    min_runtime_us: float = 100.0,
) -> tuple[float, float, float, float]:
    if not task_rows:
        return 0.0, 0.0, 0.0, 0.0

    k = len(task_rows)
    pass_count = 0
    best_speedup = 0.0
    runtime_us_values: list[float] = []
    for row in task_rows:
        correctness = bool(row.get("correctness", False))
        speedup = float(row.get("speedup", 0.0))
        runtime_us = float(row.get("runtime_us", 0.0))
        if runtime_us > 0:
            runtime_us_values.append(runtime_us)
        if correctness and speedup > pass_speedup_threshold:
            pass_count += 1
        if correctness:
            best_speedup = max(best_speedup, speedup)

    pass_rate = pass_count / k if k > 0 else 0.0
    learnability_gate = max(0.0, min(1.0, 4.0 * pass_rate * (1.0 - pass_rate)))
    speed_term = math.exp(-((best_speedup - mu) ** 2) / (2.0 * (sigma**2)))
    utility = speed_term * learnability_gate

    mean_runtime_us = statistics.mean(runtime_us_values) if runtime_us_values else 0.0
    runtime_norm = max(min_runtime_us, mean_runtime_us) / 1_000_000.0
    normalized_utility = utility / runtime_norm
    return utility, normalized_utility, best_speedup, mean_runtime_us


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


def _normalize_model_id(model_id: str) -> str:
    return model_id.split(":", 1)[0]


def _infer_renderer_name(
    model_id: str,
    *,
    explicit_renderer_name: str | None,
    get_recommended_renderer_name,
) -> str:
    if explicit_renderer_name:
        return explicit_renderer_name

    normalized = _normalize_model_id(model_id)
    try:
        return str(get_recommended_renderer_name(normalized))
    except Exception:
        pass

    if "/" in normalized:
        org, model_name = normalized.split("/", 1)
    else:
        org, model_name = "", normalized
    org = org.lower()
    model_name = model_name.lower()

    if org == "qwen" or "qwen" in model_name:
        if "instruct-2507" in model_name:
            return "qwen3_instruct"
        return "qwen3"
    if org == "openai" or "gpt-oss" in model_name:
        return "gpt_oss_no_sysprompt"
    if org == "moonshotai" or "kimi" in model_name:
        if "k2.5" in model_name:
            return "kimi_k25_disable_thinking"
        return "kimi_k2"
    if org == "deepseek-ai" or "deepseek" in model_name:
        return "deepseekv3"
    if org == "meta-llama" or "llama" in model_name:
        return "llama3"
    return "role_colon"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for chunk in matches:
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _decision_mode_for_zone(zone: str) -> str:
    if zone == ZONE_LEARNING:
        return DECISION_MODE_LEARNING
    if zone == ZONE_TOO_HARD:
        return DECISION_MODE_TOO_HARD_DECOMPOSE
    if zone == ZONE_MASTERED:
        return DECISION_MODE_MASTERED_WARMUP
    return DECISION_MODE_FALLBACK


def _target_speedup_band_for_zone(zone: str) -> tuple[float, float]:
    if zone == ZONE_LEARNING:
        return (1.3, 1.8)
    if zone == ZONE_TOO_HARD:
        return (1.2, 1.6)
    if zone == ZONE_MASTERED:
        return (1.8, 2.5)
    return (1.2, 1.8)


def _mutation_instruction_for_zone(zone: str) -> str:
    if zone == ZONE_TOO_HARD:
        return (
            "Decompose complexity by reducing one operation/fusion while preserving the "
            "exact interface. Target a solver speedup in the 1.2x-1.6x band."
        )
    if zone == ZONE_MASTERED:
        return (
            "Increase compositional complexity by adding one operation while preserving the "
            "exact interface. Target 1.8x-2.5x speedup to bridge toward harder regimes."
        )
    if zone == ZONE_LEARNING:
        return (
            "Generate a structurally harder variant by adding exactly one operation while "
            "preserving interface. Target a solver speedup in the 1.3x-1.8x band."
        )
    return (
        "Generate a valid interface-preserving mutation with moderate difficulty "
        "targeting 1.2x-1.8x speedup."
    )


def _normalize_speedup_band(value: Any, *, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            low = float(value[0])
            high = float(value[1])
        except (TypeError, ValueError):
            return fallback
        if low > high:
            low, high = high, low
        if low <= 0 or high <= 0:
            return fallback
        return (low, high)
    return fallback


def _compact_failure_exemplars(
    failure_exemplars: list[dict[str, Any]] | None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not failure_exemplars:
        return []
    compact: list[dict[str, Any]] = []
    for row in failure_exemplars[: max(0, int(limit))]:
        compact.append(
            {
                "entry_id": str(row.get("entry_id", "")),
                "category_id": str(row.get("category_id", "")),
                "zone": str(row.get("zone", "")),
                "correctness": bool(row.get("correctness", False)),
                "compiled": bool(row.get("compiled", False)),
                "speedup": float(row.get("speedup", 0.0)),
                "error_message": str(row.get("error_message", "")),
            }
        )
    return compact


def _fallback_seed_plan(
    *,
    zone: str,
    target_speedup_band: tuple[float, float],
    failure_exemplar_count: int,
) -> "SeedMutationPlan":
    return SeedMutationPlan(
        decision_mode=_decision_mode_for_zone(zone),
        reason_code="fallback",
        target_speedup_band=_target_speedup_band_for_zone(zone),
        mutation_instruction=_mutation_instruction_for_zone(zone),
        rationale=(
            f"Fallback seed plan for zone={zone}. "
            f"Provided failures={failure_exemplar_count}, "
            f"requested_band=[{target_speedup_band[0]:.2f},{target_speedup_band[1]:.2f}]"
        ),
    )


def _best_profile_for_zone(profiles: list[CapabilityProfile], zone: str) -> CapabilityProfile | None:
    candidates = [p for p in profiles if p.zone == zone]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda p: (
            p.normalized_utility,
            p.utility_score,
            p.speedup_var,
            -p.sample_count,
            p.category_id,
        ),
    )


def _required_zone_by_policy(profiles: list[CapabilityProfile]) -> str:
    if any(p.zone == ZONE_LEARNING for p in profiles):
        return ZONE_LEARNING
    if any(p.zone == ZONE_TOO_HARD for p in profiles):
        return ZONE_TOO_HARD
    if any(p.zone == ZONE_MASTERED for p in profiles):
        return ZONE_MASTERED
    return ZONE_UNKNOWN


@dataclass(frozen=True)
class TeacherDecision:
    target_category: str
    target_min_completion: float
    target_max_completion: float
    hard_frontier: bool
    backend: str
    model_id: str
    rationale: str
    decision_mode: str = DECISION_MODE_FALLBACK
    reason_code: str = "fallback"
    target_speedup_band: tuple[float, float] = (1.2, 1.8)
    mutation_instruction: str = ""
    zone: str = ZONE_UNKNOWN
    utility_score: float = 0.0
    normalized_utility: float = 0.0


@dataclass(frozen=True)
class SeedMutationPlan:
    decision_mode: str
    reason_code: str
    target_speedup_band: tuple[float, float]
    mutation_instruction: str
    rationale: str


class TeacherPolicyBackend(Protocol):
    @property
    def backend_name(self) -> str:
        ...

    @property
    def model_id(self) -> str:
        ...

    def decide(
        self,
        profiles: list[CapabilityProfile],
        *,
        target_min_completion: float,
        target_max_completion: float,
        failure_exemplars: list[dict[str, Any]] | None = None,
    ) -> TeacherDecision:
        ...

    def plan_seed_mutation(
        self,
        *,
        seed_task: KernelTask,
        target_category: str,
        zone: str,
        target_speedup_band: tuple[float, float],
        solver_trace_summary: str,
        failure_exemplars: list[dict[str, Any]] | None = None,
    ) -> SeedMutationPlan:
        ...


class HeuristicTeacherBackend:
    @property
    def backend_name(self) -> str:
        return "heuristic"

    @property
    def model_id(self) -> str:
        return "heuristic"

    def decide(
        self,
        profiles: list[CapabilityProfile],
        *,
        target_min_completion: float,
        target_max_completion: float,
        failure_exemplars: list[dict[str, Any]] | None = None,
    ) -> TeacherDecision:
        if not profiles:
            return TeacherDecision(
                target_category="unknown",
                target_min_completion=target_min_completion,
                target_max_completion=target_max_completion,
                hard_frontier=False,
                backend=self.backend_name,
                model_id=self.model_id,
                rationale="No capability profile yet; fallback to unknown.",
                decision_mode=DECISION_MODE_FALLBACK,
                reason_code="fallback",
                target_speedup_band=_target_speedup_band_for_zone(ZONE_UNKNOWN),
                mutation_instruction=_mutation_instruction_for_zone(ZONE_UNKNOWN),
                zone=ZONE_UNKNOWN,
            )

        learning = [p for p in profiles if p.zone == ZONE_LEARNING]
        too_hard = [p for p in profiles if p.zone == ZONE_TOO_HARD]
        mastered = [p for p in profiles if p.zone == ZONE_MASTERED]

        if learning:
            chosen = max(
                learning,
                key=lambda p: (
                    p.normalized_utility,
                    p.utility_score,
                    p.speedup_var,
                    -p.sample_count,
                    p.category_id,
                ),
            )
            zone = ZONE_LEARNING
            reason = "Selected highest normalized utility in learning zone."
            reason_code = "edge_signal"
        elif too_hard:
            chosen = max(
                too_hard,
                key=lambda p: (
                    p.normalized_utility,
                    p.utility_score,
                    p.speedup_var,
                    -p.sample_count,
                    p.category_id,
                ),
            )
            zone = ZONE_TOO_HARD
            reason = "No learning tasks available; selected decomposable too-hard zone task."
            reason_code = "decompose"
        elif mastered:
            chosen = min(
                mastered,
                key=lambda p: (
                    p.mean_best_speedup,
                    -p.normalized_utility,
                    p.category_id,
                ),
            )
            zone = ZONE_MASTERED
            reason = "Only mastered tasks available; selected weakest mastered task for warmup."
            reason_code = "warmup"
        else:
            chosen = max(
                profiles,
                key=lambda p: (
                    p.normalized_utility,
                    p.utility_score,
                    -p.correctness_rate,
                    p.category_id,
                ),
            )
            zone = chosen.zone if chosen.zone in ZONE_ORDER else ZONE_UNKNOWN
            reason = "Fallback to highest utility category."
            reason_code = "fallback"

        decision_mode = _decision_mode_for_zone(zone)
        target_speedup_band = _target_speedup_band_for_zone(zone)
        mutation_instruction = _mutation_instruction_for_zone(zone)

        return TeacherDecision(
            target_category=chosen.category_id,
            target_min_completion=target_min_completion,
            target_max_completion=target_max_completion,
            hard_frontier=zone == ZONE_TOO_HARD or chosen.correctness_rate < target_min_completion,
            backend=self.backend_name,
            model_id=self.model_id,
            rationale=(
                f"{reason} category={chosen.category_id}, correctness={chosen.correctness_rate:.3f}, "
                f"fast_1={chosen.fast_1_rate:.3f}, mean_speedup={chosen.mean_speedup:.3f}, "
                f"utility={chosen.utility_score:.4f}, normalized={chosen.normalized_utility:.4f}"
            ),
            decision_mode=decision_mode,
            reason_code=reason_code,
            target_speedup_band=target_speedup_band,
            mutation_instruction=mutation_instruction,
            zone=zone,
            utility_score=chosen.utility_score,
            normalized_utility=chosen.normalized_utility,
        )

    def plan_seed_mutation(
        self,
        *,
        seed_task: KernelTask,
        target_category: str,
        zone: str,
        target_speedup_band: tuple[float, float],
        solver_trace_summary: str,
        failure_exemplars: list[dict[str, Any]] | None = None,
    ) -> SeedMutationPlan:
        compact_failures = _compact_failure_exemplars(failure_exemplars, limit=4)
        failure_hint = ""
        if compact_failures:
            first = compact_failures[0]
            failure_hint = (
                f" Prior failure id={first.get('entry_id','')} "
                f"speedup={float(first.get('speedup', 0.0)):.3f} "
                f"error={first.get('error_message','')[:120]}."
            )
        return SeedMutationPlan(
            decision_mode=_decision_mode_for_zone(zone),
            reason_code="fallback",
            target_speedup_band=_target_speedup_band_for_zone(zone),
            mutation_instruction=_mutation_instruction_for_zone(zone),
            rationale=(
                f"Heuristic seed plan for category={target_category}.{failure_hint} "
                f"trace={solver_trace_summary[:180]}"
            ),
        )


class TinkerLLMTeacherBackend:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        renderer_name: str | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
        request_timeout_s: float = 60.0,
        fallback_backend: TeacherPolicyBackend | None = None,
    ):
        import tinker

        ensure_tinker_cookbook_on_path()
        from tinker_cookbook.model_info import get_recommended_renderer_name
        from tinker_cookbook.renderers import get_renderer, get_text_content
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from transformers.models.auto.tokenization_auto import AutoTokenizer

        self._tinker = tinker
        self._get_text_content = get_text_content
        self._model_id = model_id
        self._renderer_name = _infer_renderer_name(
            model_id,
            explicit_renderer_name=renderer_name,
            get_recommended_renderer_name=get_recommended_renderer_name,
        )
        self._temperature = max(0.0, float(temperature))
        self._max_tokens = max(64, int(max_tokens))
        self._request_timeout_s = max(1.0, float(request_timeout_s))
        self._fallback_backend = fallback_backend or HeuristicTeacherBackend()

        self._service_client = tinker.ServiceClient()
        if model_id.startswith("tinker://"):
            self._sampling_client = self._service_client.create_sampling_client(model_path=model_id)
            self._resolved_model_path = model_id
        else:
            self._sampling_client = self._service_client.create_sampling_client(base_model=model_id)
            self._resolved_model_path = f"base_model:{model_id}"

        normalized_model_id = _normalize_model_id(model_id)
        try:
            tokenizer = get_tokenizer(normalized_model_id)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    normalized_model_id,
                    use_fast=True,
                    trust_remote_code=True,
                )
            except TypeError:
                tokenizer = AutoTokenizer.from_pretrained(
                    normalized_model_id,
                    trust_remote_code=True,
                )
        self._renderer = get_renderer(self._renderer_name, tokenizer)

    @property
    def backend_name(self) -> str:
        return "tinker"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def resolved_model_path(self) -> str:
        return self._resolved_model_path

    def decide(
        self,
        profiles: list[CapabilityProfile],
        *,
        target_min_completion: float,
        target_max_completion: float,
        failure_exemplars: list[dict[str, Any]] | None = None,
    ) -> TeacherDecision:
        if not profiles:
            return self._fallback_backend.decide(
                profiles,
                target_min_completion=target_min_completion,
                target_max_completion=target_max_completion,
            )

        system_prompt = (
            "You are CurriculumTeacher for adaptive RL training. Choose exactly one target category "
            "from the provided categories to maximize expected capability gain per unit compute.\n"
            "Policy order:\n"
            "1) Prefer learning-zone categories with highest normalized_utility.\n"
            "2) If no learning zone exists, choose too_hard for decomposition.\n"
            "3) If only mastered exists, choose weakest mastered for warmup.\n"
            "Hard constraints:\n"
            "- If any learning-zone categories exist, target_category MUST be from learning zone.\n"
            "- If no learning categories exist but too_hard exists, target_category MUST be from too_hard zone.\n"
            "- Only choose mastered when neither learning nor too_hard exists.\n"
            "Return STRICT JSON only with keys:\n"
            "target_category, decision_mode, reason_code, target_speedup_band, mutation_instruction, rationale.\n"
            "decision_mode must be one of: learning, mastered_warmup, too_hard_decompose, fallback.\n"
            "reason_code must be one of: edge_signal, max_variance, data_sparse, decompose, warmup, fallback.\n"
            "target_speedup_band must be [low, high] numeric and follow mode policy:\n"
            "- learning: [1.3, 1.8]\n"
            "- too_hard_decompose: [1.2, 1.6]\n"
            "- mastered_warmup: [1.8, 2.5]\n"
            "- fallback: [1.2, 1.8]"
        )
        profile_lines = [
            {
                "category_id": p.category_id,
                "zone": p.zone,
                "correctness_rate": round(p.correctness_rate, 4),
                "fast_1_rate": round(p.fast_1_rate, 4),
                "mean_speedup": round(p.mean_speedup, 4),
                "mean_best_speedup": round(p.mean_best_speedup, 4),
                "speedup_var": round(p.speedup_var, 4),
                "utility_score": round(p.utility_score, 6),
                "normalized_utility": round(p.normalized_utility, 6),
                "sample_count": p.sample_count,
            }
            for p in sorted(profiles, key=lambda x: x.category_id)
        ]
        compact_failures = _compact_failure_exemplars(failure_exemplars, limit=8)
        user_prompt = (
            f"min_completion={target_min_completion:.4f}\n"
            f"max_completion={target_max_completion:.4f}\n"
            f"profiles={json.dumps(profile_lines, separators=(',', ':'))}\n\n"
            f"failure_exemplars={json.dumps(compact_failures, separators=(',', ':'))}\n\n"
            "Choose one category and emit strict JSON."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        model_input = self._renderer.build_generation_prompt(messages)
        future = self._sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=self._tinker.SamplingParams(
                max_tokens=self._max_tokens,
                stop=self._renderer.get_stop_sequences(),
                temperature=self._temperature,
            ),
        )

        try:
            result = future.result(timeout=self._request_timeout_s)
        except FuturesTimeoutError:
            return self._fallback_backend.decide(
                profiles,
                target_min_completion=target_min_completion,
                target_max_completion=target_max_completion,
            )
        except Exception:
            return self._fallback_backend.decide(
                profiles,
                target_min_completion=target_min_completion,
                target_max_completion=target_max_completion,
            )

        if not result.sequences:
            return self._fallback_backend.decide(
                profiles,
                target_min_completion=target_min_completion,
                target_max_completion=target_max_completion,
            )

        parsed_message, _ = self._renderer.parse_response(result.sequences[0].tokens)
        text = self._get_text_content(parsed_message)
        payload = _extract_json_object(text)
        if payload is None:
            return self._fallback_backend.decide(
                profiles,
                target_min_completion=target_min_completion,
                target_max_completion=target_max_completion,
            )

        valid_categories = {p.category_id for p in profiles}
        target_category = str(payload.get("target_category", "")).strip()
        if target_category not in valid_categories:
            return self._fallback_backend.decide(
                profiles,
                target_min_completion=target_min_completion,
                target_max_completion=target_max_completion,
            )

        profile_by_category = {p.category_id: p for p in profiles}
        chosen_profile = profile_by_category[target_category]

        inferred_zone = chosen_profile.zone if chosen_profile.zone in ZONE_ORDER else ZONE_UNKNOWN
        required_zone = _required_zone_by_policy(profiles)
        policy_override_applied = False
        if required_zone in {ZONE_LEARNING, ZONE_TOO_HARD, ZONE_MASTERED} and inferred_zone != required_zone:
            override_profile = _best_profile_for_zone(profiles, required_zone)
            if override_profile is not None:
                target_category = override_profile.category_id
                chosen_profile = override_profile
                inferred_zone = required_zone
                policy_override_applied = True

        inferred_mode = _decision_mode_for_zone(inferred_zone)
        model_mode = str(payload.get("decision_mode", "")).strip().lower()
        if model_mode not in VALID_DECISION_MODES or model_mode != inferred_mode:
            model_mode = inferred_mode

        reason_code = str(payload.get("reason_code", "fallback")).strip().lower()
        if reason_code not in VALID_REASON_CODES:
            reason_code = "fallback"
        if policy_override_applied:
            reason_code = "fallback"

        # Keep speedup bands deterministic by zone for stable curriculum control.
        target_speedup_band = _target_speedup_band_for_zone(inferred_zone)
        mutation_instruction = str(
            payload.get("mutation_instruction") or _mutation_instruction_for_zone(inferred_zone)
        ).strip()
        if not mutation_instruction:
            mutation_instruction = _mutation_instruction_for_zone(inferred_zone)

        inferred_hard = False
        for p in profiles:
            if p.category_id == target_category:
                inferred_hard = p.correctness_rate < target_min_completion
                break
        hard_frontier = inferred_hard or inferred_zone == ZONE_TOO_HARD
        rationale = str(payload.get("rationale", "LLM teacher decision."))
        if policy_override_applied:
            rationale = (
                f"Policy override to zone={inferred_zone} due to zone-priority constraints. "
                f"Original model choice was adjusted for deterministic curriculum control."
            )

        return TeacherDecision(
            target_category=target_category,
            target_min_completion=target_min_completion,
            target_max_completion=target_max_completion,
            hard_frontier=hard_frontier,
            backend=self.backend_name,
            model_id=self.model_id,
            rationale=rationale,
            decision_mode=model_mode,
            reason_code=reason_code,
            target_speedup_band=target_speedup_band,
            mutation_instruction=mutation_instruction,
            zone=inferred_zone,
            utility_score=chosen_profile.utility_score,
            normalized_utility=chosen_profile.normalized_utility,
        )

    def plan_seed_mutation(
        self,
        *,
        seed_task: KernelTask,
        target_category: str,
        zone: str,
        target_speedup_band: tuple[float, float],
        solver_trace_summary: str,
        failure_exemplars: list[dict[str, Any]] | None = None,
    ) -> SeedMutationPlan:
        compact_failures = _compact_failure_exemplars(failure_exemplars, limit=8)
        fallback = _fallback_seed_plan(
            zone=zone,
            target_speedup_band=target_speedup_band,
            failure_exemplar_count=len(compact_failures),
        )
        system_prompt = (
            "You are CurriculumTeacherSeedPlanner for adaptive RL training. "
            "Given a concrete seed task plus solver failure exemplars, produce a mutation plan "
            "that is difficult-but-learnable and interface-preserving.\n"
            "Return STRICT JSON only with keys:\n"
            "decision_mode, reason_code, target_speedup_band, mutation_instruction, rationale.\n"
            "Allowed decision_mode: learning, mastered_warmup, too_hard_decompose, fallback.\n"
            "Allowed reason_code: edge_signal, max_variance, data_sparse, decompose, warmup, fallback.\n"
            "Hard constraints:\n"
            "- Keep zone-consistent mode.\n"
            "- mutation_instruction must be specific to seed failure patterns.\n"
            "- target_speedup_band must be 2 numbers [low, high]."
        )
        user_payload = {
            "target_category": target_category,
            "zone": zone,
            "target_speedup_band": [round(target_speedup_band[0], 4), round(target_speedup_band[1], 4)],
            "solver_trace_summary": solver_trace_summary,
            "failure_exemplars": compact_failures,
            "seed_problem_id": seed_task.problem_id,
            "seed_name": seed_task.name,
            "seed_reference_code": seed_task.reference_code,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, separators=(",", ":")),
            },
        ]
        model_input = self._renderer.build_generation_prompt(messages)
        future = self._sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=self._tinker.SamplingParams(
                max_tokens=self._max_tokens,
                stop=self._renderer.get_stop_sequences(),
                temperature=self._temperature,
            ),
        )
        try:
            result = future.result(timeout=self._request_timeout_s)
        except Exception:
            return fallback
        if not result.sequences:
            return fallback
        parsed_message, _ = self._renderer.parse_response(result.sequences[0].tokens)
        text = self._get_text_content(parsed_message)
        payload = _extract_json_object(text)
        if payload is None:
            return fallback

        inferred_mode = _decision_mode_for_zone(zone)
        mode = str(payload.get("decision_mode", "")).strip().lower()
        if mode not in VALID_DECISION_MODES or mode != inferred_mode:
            mode = inferred_mode

        reason_code = str(payload.get("reason_code", "fallback")).strip().lower()
        if reason_code not in VALID_REASON_CODES:
            reason_code = "fallback"

        band = _normalize_speedup_band(
            payload.get("target_speedup_band"),
            fallback=_target_speedup_band_for_zone(zone),
        )
        # Keep deterministic zone-safe band policy.
        band = _target_speedup_band_for_zone(zone)
        mutation_instruction = str(payload.get("mutation_instruction", "")).strip()
        if not mutation_instruction:
            mutation_instruction = _mutation_instruction_for_zone(zone)
        rationale = str(payload.get("rationale", "LLM seed planner decision.")).strip()
        if not rationale:
            rationale = "LLM seed planner decision."

        return SeedMutationPlan(
            decision_mode=mode,
            reason_code=reason_code,
            target_speedup_band=band,
            mutation_instruction=mutation_instruction,
            rationale=rationale,
        )


@dataclass
class CurriculumTeacher:
    seed: int = 42
    policy_backend: TeacherPolicyBackend | None = None
    target_min_completion: float = 0.25
    target_max_completion: float = 0.75
    _rng: random.Random = field(init=False, repr=False)
    _latest_profile: dict[str, CapabilityProfile] = field(default_factory=dict, init=False, repr=False)
    _latest_decision: TeacherDecision | None = field(default=None, init=False, repr=False)

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

            rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for idx, row in enumerate(rows):
                task_key = str(row.get("task_id", f"row_{idx}"))
                rows_by_task[task_key].append(row)

            task_zones: list[str] = []
            task_utilities: list[float] = []
            task_norm_utilities: list[float] = []
            task_best_speedups: list[float] = []
            task_runtime_us: list[float] = []
            for task_rows in rows_by_task.values():
                zone = classify_task_zone(task_rows)
                utility, normalized_utility, best_speedup, mean_runtime_us = task_frontier_utility(task_rows)
                task_zones.append(zone)
                task_utilities.append(utility)
                task_norm_utilities.append(normalized_utility)
                task_best_speedups.append(best_speedup)
                if mean_runtime_us > 0:
                    task_runtime_us.append(mean_runtime_us)

            zone_counts = {
                ZONE_MASTERED: sum(1 for z in task_zones if z == ZONE_MASTERED),
                ZONE_LEARNING: sum(1 for z in task_zones if z == ZONE_LEARNING),
                ZONE_TOO_HARD: sum(1 for z in task_zones if z == ZONE_TOO_HARD),
            }
            total_tasks = max(1, len(task_zones))
            dominant_zone = max(
                [ZONE_LEARNING, ZONE_TOO_HARD, ZONE_MASTERED],
                key=lambda z: (zone_counts[z], -ZONE_ORDER.index(z)),
            )
            mastered_rate = zone_counts[ZONE_MASTERED] / total_tasks
            learning_rate = zone_counts[ZONE_LEARNING] / total_tasks
            too_hard_rate = zone_counts[ZONE_TOO_HARD] / total_tasks
            utility_score = statistics.mean(task_utilities) if task_utilities else 0.0
            normalized_utility = (
                statistics.mean(task_norm_utilities) if task_norm_utilities else 0.0
            )
            mean_best_speedup = (
                statistics.mean(task_best_speedups) if task_best_speedups else 0.0
            )
            mean_runtime_us = statistics.mean(task_runtime_us) if task_runtime_us else 0.0

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
                zone=dominant_zone,
                utility_score=utility_score,
                normalized_utility=normalized_utility,
                mean_best_speedup=mean_best_speedup,
                mean_runtime_us=mean_runtime_us,
                mastered_task_rate=mastered_rate,
                learning_task_rate=learning_rate,
                too_hard_task_rate=too_hard_rate,
            )
            profiles.append(profile)

        self._latest_profile = {p.category_id: p for p in profiles}
        return profiles

    def latest_profile(self) -> dict[str, CapabilityProfile]:
        return dict(self._latest_profile)

    def profiles_by_zone(self) -> dict[str, list[CapabilityProfile]]:
        buckets: dict[str, list[CapabilityProfile]] = {z: [] for z in ZONE_ORDER}
        for profile in self._latest_profile.values():
            zone = profile.zone if profile.zone in buckets else ZONE_UNKNOWN
            buckets[zone].append(profile)
        for zone in buckets:
            buckets[zone] = sorted(
                buckets[zone],
                key=lambda p: (
                    -p.normalized_utility,
                    -p.utility_score,
                    p.category_id,
                ),
            )
        return buckets

    def select_frontier_target(
        self,
        *,
        target_min_completion: float | None = None,
        target_max_completion: float | None = None,
        failure_exemplars: list[dict[str, Any]] | None = None,
    ) -> TeacherDecision:
        min_c = (
            self.target_min_completion
            if target_min_completion is None
            else float(target_min_completion)
        )
        max_c = (
            self.target_max_completion
            if target_max_completion is None
            else float(target_max_completion)
        )
        backend = self.policy_backend or HeuristicTeacherBackend()
        decision = backend.decide(
            list(self._latest_profile.values()),
            target_min_completion=min_c,
            target_max_completion=max_c,
            failure_exemplars=failure_exemplars,
        )
        self._latest_decision = decision
        return decision

    def latest_decision(self) -> TeacherDecision | None:
        return self._latest_decision

    def plan_seed_mutation(
        self,
        *,
        seed_task: KernelTask,
        target_category: str,
        zone: str,
        target_speedup_band: tuple[float, float],
        solver_trace_summary: str,
        failure_exemplars: list[dict[str, Any]] | None = None,
    ) -> SeedMutationPlan:
        backend = self.policy_backend or HeuristicTeacherBackend()
        return backend.plan_seed_mutation(
            seed_task=seed_task,
            target_category=target_category,
            zone=zone,
            target_speedup_band=target_speedup_band,
            solver_trace_summary=solver_trace_summary,
            failure_exemplars=failure_exemplars,
        )

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

        if strategy == "frontier_band":
            latest = self.latest_decision()
            target = latest.target_category if latest is not None else self.select_frontier_target().target_category
            return sorted(
                tasks,
                key=lambda t: (
                    0 if _task_category(t) == target else 1,
                    -((_profile_for(t).normalized_utility) if _profile_for(t) else 0.0),
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
