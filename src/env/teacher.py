from __future__ import annotations

import ast
import json
import random
import re
import statistics
from collections import defaultdict
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.env.schema import CapabilityProfile
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


@dataclass(frozen=True)
class TeacherDecision:
    target_category: str
    target_min_completion: float
    target_max_completion: float
    hard_frontier: bool
    backend: str
    model_id: str
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
    ) -> TeacherDecision:
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
            )

        midpoint = (target_min_completion + target_max_completion) / 2.0
        in_band = [
            p
            for p in profiles
            if target_min_completion <= p.correctness_rate <= target_max_completion
        ]
        if in_band:
            chosen = min(
                in_band,
                key=lambda p: (
                    abs(p.correctness_rate - midpoint),
                    -p.speedup_var,
                    p.sample_count,
                    p.category_id,
                ),
            )
            reason = "Selected in-band category closest to midpoint for maximal signal."
        else:
            below = [p for p in profiles if p.correctness_rate < target_min_completion]
            above = [p for p in profiles if p.correctness_rate > target_max_completion]
            if below:
                chosen = max(
                    below,
                    key=lambda p: (
                        p.correctness_rate,
                        p.speedup_var,
                        -p.sample_count,
                        p.category_id,
                    ),
                )
                reason = "Selected hardest learnable category just below lower band edge."
            elif above:
                chosen = min(
                    above,
                    key=lambda p: (
                        p.correctness_rate,
                        -p.speedup_var,
                        p.sample_count,
                        p.category_id,
                    ),
                )
                reason = "All categories too easy; selected easiest-above-band for hardening."
            else:
                chosen = min(profiles, key=lambda p: abs(p.correctness_rate - midpoint))
                reason = "Fallback to closest category by correctness rate."

        return TeacherDecision(
            target_category=chosen.category_id,
            target_min_completion=target_min_completion,
            target_max_completion=target_max_completion,
            hard_frontier=chosen.correctness_rate < target_min_completion,
            backend=self.backend_name,
            model_id=self.model_id,
            rationale=(
                f"{reason} category={chosen.category_id}, correctness={chosen.correctness_rate:.3f}, "
                f"fast_1={chosen.fast_1_rate:.3f}, mean_speedup={chosen.mean_speedup:.3f}"
            ),
        )


class TinkerLLMTeacherBackend:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
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
    ) -> TeacherDecision:
        if not profiles:
            return self._fallback_backend.decide(
                profiles,
                target_min_completion=target_min_completion,
                target_max_completion=target_max_completion,
            )

        system_prompt = (
            "You are an RL curriculum teacher. Select one target category to keep the solver at the "
            "edge of learnability. Target completion band is [min_completion, max_completion]. "
            "Prioritize categories near this band that maximize learning signal and transfer. "
            "Return JSON only with keys: target_category, hard_frontier, rationale."
        )
        profile_lines = [
            {
                "category_id": p.category_id,
                "correctness_rate": round(p.correctness_rate, 4),
                "fast_1_rate": round(p.fast_1_rate, 4),
                "mean_speedup": round(p.mean_speedup, 4),
                "speedup_var": round(p.speedup_var, 4),
                "sample_count": p.sample_count,
            }
            for p in sorted(profiles, key=lambda x: x.category_id)
        ]
        user_prompt = (
            f"min_completion={target_min_completion:.4f}\n"
            f"max_completion={target_max_completion:.4f}\n"
            f"profiles={json.dumps(profile_lines, separators=(',', ':'))}\n\n"
            "Choose one category. hard_frontier=true only if chosen category is below min_completion."
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

        # Infer hard_frontier from measured capability profile to avoid LLM inconsistency.
        inferred_hard = False
        for p in profiles:
            if p.category_id == target_category:
                inferred_hard = p.correctness_rate < target_min_completion
                break
        hard_frontier = inferred_hard
        rationale = str(payload.get("rationale", "LLM teacher decision."))

        return TeacherDecision(
            target_category=target_category,
            target_min_completion=target_min_completion,
            target_max_completion=target_max_completion,
            hard_frontier=hard_frontier,
            backend=self.backend_name,
            model_id=self.model_id,
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

    def latest_profile(self) -> dict[str, CapabilityProfile]:
        return dict(self._latest_profile)

    def select_frontier_target(
        self,
        *,
        target_min_completion: float | None = None,
        target_max_completion: float | None = None,
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
        )
        self._latest_decision = decision
        return decision

    def latest_decision(self) -> TeacherDecision | None:
        return self._latest_decision

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
            target = self.select_frontier_target().target_category
            return sorted(
                tasks,
                key=lambda t: (
                    0 if _task_category(t) == target else 1,
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
