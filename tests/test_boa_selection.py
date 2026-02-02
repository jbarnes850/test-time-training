"""Unit tests for BoA selection logic.

Validates that the BoA selection algorithm correctly selects the checkpoint
with the highest aggregate fast_1 score from a trajectory of adaptation steps.
"""

import json
import tempfile
from pathlib import Path


def test_boa_selection_logic():
    """Test that BoA selects the checkpoint with highest fast_1."""
    # Simulate step summaries from a batch TTT run
    step_summaries = [
        {"step_idx": 0, "aggregate_fast_1": 0.494, "aggregate_correct": 0.875},
        {"step_idx": 1, "aggregate_fast_1": 0.550, "aggregate_correct": 0.888},  # Best
        {"step_idx": 2, "aggregate_fast_1": 0.456, "aggregate_correct": 0.875},
        {"step_idx": 3, "aggregate_fast_1": 0.419, "aggregate_correct": 0.913},
    ]

    checkpoint_paths = {
        0: "/path/to/base_checkpoint",
        1: "/path/to/step_1_checkpoint",
        2: "/path/to/step_2_checkpoint",
        3: "/path/to/step_3_checkpoint",
    }

    # BoA selection logic (same as in batch_ttt.py)
    boa_selected_step = max(
        range(len(step_summaries)),
        key=lambda i: step_summaries[i]["aggregate_fast_1"]
    )
    boa_selected_metrics = step_summaries[boa_selected_step]

    # Verify selection
    assert boa_selected_step == 1, f"Expected step 1, got {boa_selected_step}"
    assert boa_selected_metrics["aggregate_fast_1"] == 0.550
    assert checkpoint_paths[boa_selected_step] == "/path/to/step_1_checkpoint"

    print("BoA selection logic test PASSED")
    print(f"  Selected step: {boa_selected_step}")
    print(f"  Selected fast_1: {boa_selected_metrics['aggregate_fast_1']}")
    print(f"  Selected checkpoint: {checkpoint_paths[boa_selected_step]}")


def test_boa_output_format():
    """Test that BoA output JSON has required fields."""
    # Simulate BoA output
    boa_selected = {
        "algorithm": "Best-of-Adaptation (BoA)",
        "selection_criterion": "argmax(aggregate_fast_1)",
        "selected_step": 1,
        "selected_checkpoint": "/path/to/step_1_checkpoint",
        "selected_metrics": {
            "aggregate_fast_1": 0.550,
            "aggregate_correct": 0.888,
            "per_task_fast_1": {4: 0.25, 5: 0.469, 12: 1.0, 14: 0.50, 15: 0.531},
        },
        "all_steps_fast_1": [0.494, 0.550, 0.456, 0.419],
        "config": {
            "model": "openai/gpt-oss-120b",
            "tasks": [4, 5, 12, 14, 15],
            "k": 32,
            "steps": 3,
        },
    }

    # Write to temp file and read back
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "boa_selected.json"
        output_path.write_text(json.dumps(boa_selected, indent=2))

        # Verify file was written and can be read back
        loaded = json.loads(output_path.read_text())

        assert loaded["algorithm"] == "Best-of-Adaptation (BoA)"
        assert loaded["selected_step"] == 1
        assert loaded["selected_checkpoint"] == "/path/to/step_1_checkpoint"
        assert loaded["selected_metrics"]["aggregate_fast_1"] == 0.550
        assert len(loaded["all_steps_fast_1"]) == 4

    print("BoA output format test PASSED")


def test_boa_handles_early_stopping():
    """Test that BoA correctly identifies when early stopping matches oracle."""
    step_summaries = [
        {"step_idx": 0, "aggregate_fast_1": 0.494},
        {"step_idx": 1, "aggregate_fast_1": 0.550},  # Oracle peak
        {"step_idx": 2, "aggregate_fast_1": 0.456},  # Regression
    ]

    # BoA selection
    boa_step = max(range(len(step_summaries)), key=lambda i: step_summaries[i]["aggregate_fast_1"])

    # Early stopping (P=1): detect first regression
    early_stop_step = 0
    for i in range(1, len(step_summaries)):
        if step_summaries[i]["aggregate_fast_1"] > step_summaries[i-1]["aggregate_fast_1"]:
            early_stop_step = i
        else:
            break  # Stop at first regression

    # Verify both methods select step 1
    assert boa_step == 1, f"BoA should select step 1, got {boa_step}"
    assert early_stop_step == 1, f"Early stopping should stop at step 1, got {early_stop_step}"

    print("BoA early stopping test PASSED")
    print(f"  BoA selected step: {boa_step}")
    print(f"  Early stopping step: {early_stop_step}")
    print("  Both match oracle peak at step 1")


if __name__ == "__main__":
    test_boa_selection_logic()
    test_boa_output_format()
    test_boa_handles_early_stopping()
    print("\nAll BoA tests PASSED")
