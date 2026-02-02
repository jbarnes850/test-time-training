from src.utils.feedback_utils import extract_error_info, build_execution_feedback


def test_extract_error_info_compilation():
    metadata = {
        "compilation_error_name": "SyntaxError",
        "compilation_error": "bad syntax at line 1",
    }
    msg, trace = extract_error_info(metadata)
    assert "SyntaxError" in msg
    assert "bad syntax" in msg
    assert trace == ""


def test_extract_error_info_runtime_trace():
    metadata = {
        "runtime_error_name": "RuntimeError",
        "runtime_error": "cuda launch failed",
        "runtime_error_traceback": "Traceback (most recent call last):\n  line1\n  line2",
    }
    msg, trace = extract_error_info(metadata)
    assert "RuntimeError" in msg
    assert "cuda launch failed" in msg
    assert "Traceback" in trace


def test_build_execution_feedback_status():
    feedback = build_execution_feedback(
        compiled=False,
        correctness=False,
        speedup=0.0,
        runtime_us=-1.0,
        ref_runtime_us=-1.0,
        error_message="SyntaxError: bad syntax",
        error_trace="Traceback...",
    )
    assert "status=compilation_error" in feedback
    assert "error_message=" in feedback

    feedback_ok = build_execution_feedback(
        compiled=True,
        correctness=True,
        speedup=1.2,
        runtime_us=10.0,
        ref_runtime_us=12.0,
    )
    assert "status=success" in feedback_ok
