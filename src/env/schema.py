from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class KernelTask:
    problem_id: int
    name: str
    reference_code: str


@dataclass(frozen=True)
class Action:
    kernel_code: str


@dataclass(frozen=True)
class Observation:
    problem_id: int
    prompt: str


@dataclass(frozen=True)
class EvalResult:
    compiled: bool
    correctness: bool
    runtime_us: float
    ref_runtime_us: float
    speedup: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class StepResult:
    reward: float
    eval_result: EvalResult
    done: bool = True
    info: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class MutatedTask:
    task_id: str
    parent_task_id: str
    seed_problem_id: Optional[int]
    name: str
    reference_code: str
    interface_signature_hash: str
    category_tags: tuple[str, ...]
    category_id: str
    mutation_backend: str
    mutation_model_id: str
    mutation_prompt_hash: str
    novelty_hash: str
    epoch_created: int
    mutation_type: str = "unspecified"
    optimization_prompt: str = ""
    teacher_decision_mode: str = ""
    teacher_reason_code: str = ""
    teacher_target_speedup_band: tuple[float, float] = (0.0, 0.0)
    teacher_mutation_instruction: str = ""
    solver_trace_summary: str = ""
    teacher_failure_entry_ids: tuple[str, ...] = ()
    teacher_seed_rationale: str = ""


@dataclass(frozen=True)
class CapabilityProfile:
    epoch: int
    split: str
    category_id: str
    n_tasks: int
    correctness_rate: float
    mean_speedup: float
    speedup_var: float
    fast_1_rate: float
    failure_rate: float
    sample_count: int
    zone: str = "unknown"
    utility_score: float = 0.0
    normalized_utility: float = 0.0
    mean_best_speedup: float = 0.0
    mean_runtime_us: float = 0.0
    mastered_task_rate: float = 0.0
    learning_task_rate: float = 0.0
    too_hard_task_rate: float = 0.0


@dataclass(frozen=True)
class ReplayEntry:
    entry_id: str
    task_id: str
    parent_task_id: Optional[str]
    problem_id: Optional[int]
    level: Optional[int]
    category_id: str
    task_reference_code: str
    kernel_code: str
    eval_result: EvalResult
    reward: float
    sampler_path: str
    backend: str
    timestamp: float
    epoch: int
    is_mutated: bool = False


@dataclass(frozen=True)
class TrainingArtifact:
    entry_id: str
    outcome_id: str
    epoch: int
    zone: str
    utility_score: float
    category_id: str
    problem_id: Optional[int]
    level: Optional[int]
    prompt_tokens: list[int]
    sampled_tokens: list[int]
    sampled_logprobs: list[float]
    reward: float
    sampler_path: str


def to_json_dict(obj) -> Dict[str, Any]:
    return asdict(obj)
