from dataclasses import dataclass, asdict
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


def to_json_dict(obj) -> Dict[str, Any]:
    return asdict(obj)
