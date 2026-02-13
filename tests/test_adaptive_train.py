import json
from pathlib import Path
from types import SimpleNamespace

import scripts.adaptive_train as adaptive_train
from src.env.schema import KernelTask


def _write_split(path: Path) -> None:
    payload = {
        "problem_ids": {
            "train": [1, 2],
            "eval": [3, 4],
        }
    }
    path.write_text(json.dumps(payload))


def _fake_task(problem_id: int, level: int = 1) -> KernelTask:
    code = f"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return x + {problem_id}
"""
    return KernelTask(problem_id=problem_id, name=f"task_{problem_id}", reference_code=code)


def test_cost_tracker_projection_math():
    tracker = adaptive_train.CostTracker(gpu_hour_rate=2.0)
    tracker.add_cost("profile", gpu_hours=1.0, api_usd=3.0, epoch=0)
    assert tracker.total_usd == 5.0
    epoch_cost = tracker.finalize_epoch(epoch=0, start_total_usd=0.0)
    assert epoch_cost == 5.0
    assert tracker.projected_total(remaining_epochs=2, fallback_epoch_cost_usd=10.0) == 15.0


def test_adaptive_train_dry_run_writes_required_artifacts(tmp_path: Path, monkeypatch):
    split_train = tmp_path / "split_train.json"
    split_eval = tmp_path / "split_eval.json"
    _write_split(split_train)
    _write_split(split_eval)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)

    run_dir = tmp_path / "run"
    rc = adaptive_train.main(
        [
            "--dry_run",
            "--split_train",
            str(split_train),
            "--split_eval",
            str(split_eval),
            "--epochs",
            "1",
            "--tasks_per_epoch",
            "1",
            "--k",
            "2",
            "--max_train_tasks",
            "2",
            "--max_eval_tasks",
            "2",
            "--log_path",
            str(run_dir),
        ]
    )
    assert rc == 0

    for name in [
        "run_config.json",
        "epoch_summary.jsonl",
        "capability_profiles.jsonl",
        "replay_entries.jsonl",
        "mutator_stats.jsonl",
        "mutation_events.jsonl",
        "kpi_dashboard.jsonl",
        "cost_tracker.json",
        "checkpoint_state.json",
    ]:
        assert (run_dir / name).exists(), name

    summary = [json.loads(line) for line in (run_dir / "epoch_summary.jsonl").read_text().splitlines()]
    assert summary
    assert summary[0]["epoch"] == 0
    assert "mutator_stats" in summary[0]
    assert "curriculum" in summary[0]
    assert "kpi" in summary[0]

    mutation_events = [
        json.loads(line) for line in (run_dir / "mutation_events.jsonl").read_text().splitlines()
    ]
    assert mutation_events
    assert "decision_mode" in mutation_events[0]
    assert "reason_code" in mutation_events[0]
    assert "target_speedup_band" in mutation_events[0]
    assert "mutation_instruction" in mutation_events[0]

    run_config = json.loads((run_dir / "run_config.json").read_text())
    assert run_config["resolved_solver_backend"] == "dry_run"
    assert run_config["resolved_solver_metadata"]["backend"] == "dry_run"
    assert run_config["resolved_mutator_backend"] == "dry_run"


def test_adaptive_train_resume(tmp_path: Path, monkeypatch):
    split_train = tmp_path / "split_train.json"
    split_eval = tmp_path / "split_eval.json"
    _write_split(split_train)
    _write_split(split_eval)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)
    run_dir = tmp_path / "resume_run"

    rc1 = adaptive_train.main(
        [
            "--dry_run",
            "--split_train",
            str(split_train),
            "--split_eval",
            str(split_eval),
            "--epochs",
            "1",
            "--tasks_per_epoch",
            "1",
            "--k",
            "1",
            "--log_path",
            str(run_dir),
        ]
    )
    assert rc1 == 0

    rc2 = adaptive_train.main(
        [
            "--dry_run",
            "--resume_from",
            str(run_dir),
            "--split_train",
            str(split_train),
            "--split_eval",
            str(split_eval),
            "--epochs",
            "2",
            "--tasks_per_epoch",
            "1",
            "--k",
            "1",
        ]
    )
    assert rc2 == 0
    state = json.loads((run_dir / "checkpoint_state.json").read_text())
    assert state["next_epoch"] == 2


def test_budget_projection_halts_before_epoch(tmp_path: Path, monkeypatch):
    split_train = tmp_path / "split_train.json"
    split_eval = tmp_path / "split_eval.json"
    _write_split(split_train)
    _write_split(split_eval)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)
    run_dir = tmp_path / "halt_run"

    rc = adaptive_train.main(
        [
            "--dry_run",
            "--split_train",
            str(split_train),
            "--split_eval",
            str(split_eval),
            "--epochs",
            "2",
            "--default_epoch_cost_usd",
            "1000",
            "--budget_tier",
            "smoke",
            "--log_path",
            str(run_dir),
        ]
    )
    assert rc == 0
    rows = [json.loads(line) for line in (run_dir / "epoch_summary.jsonl").read_text().splitlines()]
    assert rows
    assert rows[0]["status"] == "halted_budget_projection"


def test_allocate_level_counts_sums_to_total():
    counts = adaptive_train._allocate_level_counts(20, [0.25, 0.45, 0.2, 0.1])
    assert sum(counts) == 20
    assert counts == [5, 9, 4, 2]


def test_build_mixed_train_handles_uses_available_levels(monkeypatch):
    monkeypatch.setattr(adaptive_train, "available_kernelbench_levels", lambda: (1, 2, 3, 4))

    def _fake_level(level: int):
        return [{"problem_id": i} for i in range(1, 11)]

    monkeypatch.setattr(adaptive_train, "load_kernelbench_level", _fake_level)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)

    args = SimpleNamespace(
        seed_levels="1,2,5",
        seed_mix="0.25,0.45,0.30",
        max_train_tasks=10,
        seed=42,
        train_subset="train",
        seed_split_paths="",
    )
    handles = adaptive_train._build_mixed_train_handles(args)
    assert len(handles) == 10
    levels = {h.level for h in handles}
    assert levels == {1, 2}
