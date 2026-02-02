from __future__ import annotations

from typing import Any, Dict, Tuple


_MAX_ERROR_CHARS = 300
_MAX_ERROR_LINES = 3


def _truncate_text(text: str, max_chars: int = _MAX_ERROR_CHARS, max_lines: int = _MAX_ERROR_LINES) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if max_lines is not None and max_lines > 0:
        lines = lines[:max_lines]
    compact = " | ".join(lines)
    if max_chars is not None and max_chars > 0 and len(compact) > max_chars:
        return compact[: max_chars - 3] + "..."
    return compact


def extract_error_info(metadata: Dict[str, Any] | None) -> Tuple[str, str]:
    if not metadata:
        return "", ""

    def _to_str(val: Any) -> str:
        if val is None:
            return ""
        try:
            return str(val)
        except Exception:
            return ""

    error_message = ""
    error_trace = ""

    if "compilation_error" in metadata or "compilation_error_name" in metadata:
        name = _to_str(metadata.get("compilation_error_name", "CompilationError"))
        msg = _to_str(metadata.get("compilation_error"))
        error_message = f"{name}: {msg}" if msg else name
    elif "runtime_error" in metadata or "runtime_error_name" in metadata:
        name = _to_str(metadata.get("runtime_error_name", "RuntimeError"))
        msg = _to_str(metadata.get("runtime_error"))
        error_message = f"{name}: {msg}" if msg else name
        error_trace = _to_str(metadata.get("runtime_error_traceback"))
    elif "correctness_issue" in metadata or "correctness_issue_name" in metadata:
        name = _to_str(metadata.get("correctness_issue_name", "CorrectnessIssue"))
        msg = _to_str(metadata.get("correctness_issue"))
        error_message = f"{name}: {msg}" if msg else name
    elif "error" in metadata:
        msg = _to_str(metadata.get("error"))
        error_message = f"EvalError: {msg}" if msg else "EvalError"
        error_trace = _to_str(metadata.get("error_trace"))

    return _truncate_text(error_message), _truncate_text(error_trace)


def build_execution_feedback(
    compiled: bool,
    correctness: bool,
    speedup: float,
    runtime_us: float,
    ref_runtime_us: float,
    error_message: str = "",
    error_trace: str = "",
) -> str:
    if not compiled:
        status = "compilation_error"
    elif not correctness:
        status = "runtime_error" if error_message else "incorrect_output"
    elif speedup > 1.0:
        status = "success"
    else:
        status = "correct_but_slow"

    lines = [
        f"status={status}",
        f"compiled={compiled}",
        f"correct={correctness}",
        f"speedup={speedup:.4f}",
        f"runtime_us={runtime_us:.2f}",
        f"ref_runtime_us={ref_runtime_us:.2f}",
    ]
    if error_message:
        lines.append(f"error_message={_truncate_text(error_message)}")
    if error_trace:
        lines.append(f"error_trace={_truncate_text(error_trace)}")

    return "EXECUTION_FEEDBACK\n" + "\n".join(lines)


def build_teacher_context(
    task_prompt: str,
    student_code: str,
    feedback: str,
    successful_solution: str | None = None,
) -> str:
    prompt = task_prompt.strip()
    feedback_block = feedback.strip()
    solution = (successful_solution or "").strip()

    sections = [prompt]
    if solution:
        sections.append(f"Correct solution:\n{solution}")
    if feedback_block:
        sections.append(
            "The following is feedback from your unsuccessful earlier attempt:\n"
            f"{feedback_block}"
        )
    sections.append("Correctly solve the original question.")

    return "\n\n".join(sections).strip() + "\n"
