import time
import pytest

from src.env.telemetry import build_record, validate_telemetry


def test_validate_telemetry():
    record = build_record(
        run_id="demo",
        seed=42,
        model_id="gpt-oss-20b",
        problem_id=1,
        level=1,
        prompt="prompt",
        kernel_code="kernel",
        compiled=True,
        correctness=True,
        runtime_us=1.0,
        ref_runtime_us=2.0,
        speedup=2.0,
        reward=1.0,
        timing_method="cuda_event",
        backend="cuda",
        precision="fp32",
        eval_metadata={"note": "ok"},
        error_message="",
        error_trace="",
    )
    data = record.to_dict()
    validate_telemetry(data)
    assert data["timestamp"] <= time.time()


def test_validate_telemetry_missing_field():
    record = build_record(
        run_id="demo",
        seed=42,
        model_id="gpt-oss-20b",
        problem_id=1,
        level=1,
        prompt="prompt",
        kernel_code="kernel",
        compiled=True,
        correctness=True,
        runtime_us=1.0,
        ref_runtime_us=2.0,
        speedup=2.0,
        reward=1.0,
        timing_method="cuda_event",
        backend="cuda",
        precision="fp32",
        eval_metadata={},
        error_message="",
        error_trace="",
    ).to_dict()
    record.pop("model_id")
    with pytest.raises(ValueError):
        validate_telemetry(record)
