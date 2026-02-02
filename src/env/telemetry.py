from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import time
import hashlib
import json
import numbers


REQUIRED_FIELDS = {
    "run_id": str,
    "seed": int,
    "model_id": str,
    "problem_id": int,
    "level": int,
    "prompt_hash": str,
    "kernel_id": str,
    "compiled": bool,
    "correctness": bool,
    "runtime_us": float,
    "ref_runtime_us": float,
    "speedup": float,
    "reward": float,
    "timing_method": str,
    "backend": str,
    "precision": str,
    "eval_metadata": dict,
    "error_message": str,
    "error_trace": str,
    "timestamp": float,
}


@dataclass(frozen=True)
class TelemetryRecord:
    run_id: str
    seed: int
    model_id: str
    problem_id: int
    level: int
    prompt_hash: str
    kernel_id: str
    compiled: bool
    correctness: bool
    runtime_us: float
    ref_runtime_us: float
    speedup: float
    reward: float
    timing_method: str
    backend: str
    precision: str
    eval_metadata: Dict[str, Any]
    error_message: str
    error_trace: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def make_kernel_id(kernel_code: str) -> str:
    return hashlib.sha256(kernel_code.encode("utf-8")).hexdigest()


def validate_telemetry(record: Dict[str, Any]) -> None:
    for field, typ in REQUIRED_FIELDS.items():
        if field not in record:
            raise ValueError(f"Missing telemetry field: {field}")
        value = record[field]
        if typ is int:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"Field {field} has wrong type: {type(value)} expected int")
            continue
        if typ is float:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"Field {field} has wrong type: {type(value)} expected float")
            continue
        if not isinstance(value, typ):
            raise TypeError(f"Field {field} has wrong type: {type(value)} expected {typ}")


def build_record(
    run_id: str,
    seed: int,
    model_id: str,
    problem_id: int,
    level: int,
    prompt: str,
    kernel_code: str,
    compiled: bool,
    correctness: bool,
    runtime_us: float,
    ref_runtime_us: float,
    speedup: float,
    reward: float,
    timing_method: str,
    backend: str,
    precision: str,
    eval_metadata: Optional[Dict[str, Any]] = None,
    error_message: str = "",
    error_trace: str = "",
) -> TelemetryRecord:
    return TelemetryRecord(
        run_id=run_id,
        seed=seed,
        model_id=model_id,
        problem_id=problem_id,
        level=level,
        prompt_hash=hash_prompt(prompt),
        kernel_id=make_kernel_id(kernel_code),
        compiled=compiled,
        correctness=correctness,
        runtime_us=runtime_us,
        ref_runtime_us=ref_runtime_us,
        speedup=speedup,
        reward=reward,
        timing_method=timing_method,
        backend=backend,
        precision=precision,
        eval_metadata=eval_metadata or {},
        error_message=error_message,
        error_trace=error_trace,
        timestamp=time.time(),
    )


def dumps_jsonl(record: TelemetryRecord) -> str:
    return json.dumps(record.to_dict())
