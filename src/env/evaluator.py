from dataclasses import dataclass
import logging
import os
import sys
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

from src.env.schema import EvalResult
from src.utils.feedback_utils import extract_error_info
from src.utils.path_utils import repo_root
from src.utils.dataset_utils import load_kernelbench_level


@dataclass(frozen=True)
class EvalConfig:
    num_correct_trials: int = 5
    num_perf_trials: int = 50
    timing_method: str = "cuda_event"
    backend: str = "cuda"
    precision: str = "fp32"
    measure_performance: bool = True
    build_dir_prefix: str = "runs/build"


def _ensure_kernelbench_on_path() -> Path:
    root = repo_root()
    settings_path = root / "configs" / "settings.json"
    if not settings_path.exists():
        raise FileNotFoundError(
            f"Missing {settings_path}. "
            "Create it with: "
            'echo \'{"kernelbench_root": "vendor/KernelBench"}\' > configs/settings.json'
        )
    import json
    cfg = json.loads(settings_path.read_text())
    kb_root_value = cfg.get("kernelbench_root")
    if not kb_root_value:
        raise ValueError("kernelbench_root not configured")
    kb_root = (root / kb_root_value).resolve()
    kb_src = kb_root / "src"
    if str(kb_src) not in sys.path:
        sys.path.insert(0, str(kb_src))
    return kb_root


def compute_speedup(runtime_us: float, ref_runtime_us: float) -> float:
    if runtime_us <= 0 or ref_runtime_us <= 0:
        return 0.0
    return ref_runtime_us / runtime_us


def compute_reward(speedup: float, correctness: bool, correct_bonus: float = 0.0) -> float:
    if not correctness:
        return 0.0
    # Dense reward to improve learning signal; relative advantages handle scaling.
    return max(0.0, float(speedup)) + float(correct_bonus)


def eval_config_from_env() -> EvalConfig:
    mode = os.getenv("KERNELBENCH_EVAL_MODE", "full").lower()
    if mode in {"fast", "proxy"}:
        num_correct_trials = int(os.getenv("KERNELBENCH_NUM_CORRECT_TRIALS", "1"))
        num_perf_trials = int(os.getenv("KERNELBENCH_NUM_PERF_TRIALS", "5"))
    else:
        num_correct_trials = int(os.getenv("KERNELBENCH_NUM_CORRECT_TRIALS", "5"))
        num_perf_trials = int(os.getenv("KERNELBENCH_NUM_PERF_TRIALS", "50"))

    return EvalConfig(
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
    )


def _error_result(message: str) -> EvalResult:
    metadata = {"error": message}
    error_message, error_trace = extract_error_info(metadata)
    metadata["error_message"] = error_message
    metadata["error_trace"] = error_trace
    return EvalResult(
        compiled=False,
        correctness=False,
        runtime_us=-1.0,
        ref_runtime_us=-1.0,
        speedup=0.0,
        metadata=metadata,
    )


def evaluate_kernel(problem_id: int, kernel_code: str, level: int = 1, config: EvalConfig | None = None) -> EvalResult:
    try:
        _ensure_kernelbench_on_path()
        from kernelbench import eval as kernel_eval
        import torch

        config = config or eval_config_from_env()
        dataset = load_kernelbench_level(level)
        rows = [row for row in dataset if row["problem_id"] == problem_id]
        if not rows:
            return _error_result(f"Problem id {problem_id} not found in level_{level}")
        ref_code = rows[0]["code"]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        kernel_hash = hashlib.sha256(kernel_code.encode("utf-8")).hexdigest()
        build_dir = str(Path(config.build_dir_prefix) / f"problem_{problem_id}" / kernel_hash)

        eval_result = kernel_eval.eval_kernel_against_ref(
            original_model_src=ref_code,
            custom_model_src=kernel_code,
            measure_performance=config.measure_performance,
            timing_method=config.timing_method,
            num_correct_trials=config.num_correct_trials,
            num_perf_trials=config.num_perf_trials,
            build_dir=build_dir,
            device=device,
            backend=config.backend,
            precision=kernel_eval.get_torch_dtype_from_string(config.precision),
        )

        runtime_us = float(eval_result.runtime)
        ref_runtime_us = float(eval_result.ref_runtime)
        speedup = compute_speedup(runtime_us, ref_runtime_us)

        metadata = dict(eval_result.metadata or {})
        if hasattr(kernel_eval, "check_metadata_serializable_all_types"):
            metadata = kernel_eval.check_metadata_serializable_all_types(metadata)
        error_message, error_trace = extract_error_info(metadata)
        if error_message:
            metadata["error_message"] = error_message
        if error_trace:
            metadata["error_trace"] = error_trace

        return EvalResult(
            compiled=bool(eval_result.compiled),
            correctness=bool(eval_result.correctness),
            runtime_us=runtime_us,
            ref_runtime_us=ref_runtime_us,
            speedup=speedup,
            metadata=metadata,
        )
    except Exception as exc:
        logger.error("evaluate_kernel(problem_id=%d, level=%d) failed: %s", problem_id, level, exc)
        return _error_result(str(exc))
