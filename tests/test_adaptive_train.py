import json
from pathlib import Path
from types import SimpleNamespace

import scripts.adaptive_train as adaptive_train
from src.env.schema import EvalResult, KernelTask, MutatedTask, ReplayEntry


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
    assert "failure_exemplar_count" in mutation_events[0]

    run_config = json.loads((run_dir / "run_config.json").read_text())
    assert run_config["resolved_experiment_arm"] == "custom"
    assert run_config["phase1_claim_scope"]["world_model_enabled"] is False
    assert run_config["resolved_solver_backend"] == "dry_run"
    assert run_config["resolved_solver_metadata"]["backend"] == "dry_run"
    assert run_config["resolved_mutator_backend"] == "dry_run"
    assert run_config["experiment_arm_policy"]["comparison_scope"] == "user_defined"

    assert "gradient_signal" in summary[0]
    assert "effective_tasks" in summary[0]["gradient_signal"]
    assert "zero_signal_tasks" in summary[0]["gradient_signal"]
    assert "controller" in summary[0]["curriculum"]
    assert "teacher_failure_context_count" in summary[0]["curriculum"]


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


def test_apply_experiment_arm_defaults_b0_clears_sampler_init():
    args = SimpleNamespace(
        experiment_arm="B0",
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        solver_model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
        teacher_model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
        teacher_strategy="inverse_correctness",
        curriculum_controller="fixed",
        seed_use_mixed_pool=False,
        seed_pool_mode="configured_mix",
        solver_sampler_path="tinker://old_sampler",
        sampler_path="tinker://legacy_sampler",
        checkpoint_jsonl="runs/checkpoints.jsonl",
        enable_training=True,
        tasks_per_epoch=20,
    )
    resolved, applied = adaptive_train._apply_experiment_arm_defaults(args)
    assert resolved == "B0"
    assert args.model == adaptive_train.PAPER_BASE_MODEL_ID
    assert args.solver_model_id == ""
    assert args.teacher_model_id == adaptive_train.TEACHER_DEFAULT_MODEL_ID
    assert args.curriculum_controller == "adaptive"
    assert args.seed_use_mixed_pool is True
    assert args.seed_pool_mode == "uniform_levels"
    assert args.solver_sampler_path == ""
    assert args.sampler_path == ""
    assert args.checkpoint_jsonl == ""
    assert "solver_sampler_path" in applied


def test_experiment_arm_b1_enforces_baseline_purity(tmp_path: Path, monkeypatch):
    split_train = tmp_path / "split_train.json"
    split_eval = tmp_path / "split_eval.json"
    _write_split(split_train)
    _write_split(split_eval)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)
    monkeypatch.setattr(adaptive_train, "available_kernelbench_levels", lambda: (1, 2))
    monkeypatch.setattr(
        adaptive_train,
        "load_kernelbench_level",
        lambda level: [{"problem_id": i} for i in range(1, 11)],
    )

    run_dir = tmp_path / "run_b1"
    rc = adaptive_train.main(
        [
            "--dry_run",
            "--experiment_arm",
            "B1",
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
            "4",
            "--max_eval_tasks",
            "2",
            "--log_path",
            str(run_dir),
        ]
    )
    assert rc == 0
    run_config = json.loads((run_dir / "run_config.json").read_text())
    assert run_config["resolved_experiment_arm"] == "B1"
    assert run_config["model"] == adaptive_train.PAPER_BASE_MODEL_ID
    assert run_config["teacher_strategy"] == "random"
    assert run_config["curriculum_controller"] == "fixed"
    assert run_config["seed_pool_mode"] == "uniform_levels"
    assert run_config["experiment_arm_policy"]["arm_role"] == "matched_compute_baseline"

    summary = [json.loads(line) for line in (run_dir / "epoch_summary.jsonl").read_text().splitlines()]
    assert summary[0]["curriculum"]["controller"] == "fixed"


def test_non_custom_arm_rejects_non_paper_model(tmp_path: Path, monkeypatch):
    split_train = tmp_path / "split_train.json"
    split_eval = tmp_path / "split_eval.json"
    _write_split(split_train)
    _write_split(split_eval)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)
    monkeypatch.setattr(adaptive_train, "_apply_experiment_arm_defaults", lambda args: ("B1", {}))
    try:
        adaptive_train.main(
            [
                "--dry_run",
                "--experiment_arm",
                "B1",
                "--model",
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "--split_train",
                str(split_train),
                "--split_eval",
                str(split_eval),
                "--epochs",
                "1",
                "--curriculum_controller",
                "fixed",
                "--seed_pool_mode",
                "uniform_levels",
                "--tasks_per_epoch",
                "1",
                "--k",
                "1",
                "--log_path",
                str(tmp_path / "run_bad_model"),
            ]
        )
    except ValueError as exc:
        assert "pinned to paper base model" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-paper model on standardized arm")


def test_low_gradient_signal_gate_halts_run(tmp_path: Path, monkeypatch):
    split_train = tmp_path / "split_train.json"
    split_eval = tmp_path / "split_eval.json"
    _write_split(split_train)
    _write_split(split_eval)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)
    counter = {"i": 0}

    def _fake_mutate(
        self,
        seed_task,
        *,
        epoch,
        seed_problem_id,
        target_category,
        level,
        target_speedup_band,
        failure_exemplars,
        solver_trace_summary,
        mutation_instruction,
        decision_mode,
        reason_code,
        teacher_seed_rationale,
    ):
        counter["i"] += 1
        return MutatedTask(
            task_id=f"mut_{epoch}_{counter['i']}",
            parent_task_id=f"seed_{seed_problem_id}",
            seed_problem_id=seed_problem_id,
            name=f"mut_task_{seed_problem_id}",
            reference_code=seed_task.reference_code + f"\n# mut {counter['i']}",
            interface_signature_hash="sig",
            category_tags=("activation",),
            category_id=target_category or "activation",
            mutation_backend="dry_run",
            mutation_model_id="dry_run",
            mutation_prompt_hash=f"prompt_{counter['i']}",
            novelty_hash=f"novel_{counter['i']}",
            epoch_created=epoch,
            mutation_type="logic_restructuring",
            optimization_prompt="focus memory access",
            teacher_decision_mode=decision_mode,
            teacher_reason_code=reason_code,
            teacher_target_speedup_band=target_speedup_band,
            teacher_mutation_instruction=mutation_instruction,
            solver_trace_summary=solver_trace_summary,
            teacher_failure_entry_ids=tuple(
                str(row.get("entry_id", "")) for row in (failure_exemplars or [])
            ),
            teacher_seed_rationale=teacher_seed_rationale or "",
        )

    monkeypatch.setattr(adaptive_train.KernelMutator, "mutate", _fake_mutate)

    run_dir = tmp_path / "run_low_signal"
    rc = adaptive_train.main(
        [
            "--dry_run",
            "--split_train",
            str(split_train),
            "--split_eval",
            str(split_eval),
            "--epochs",
            "3",
            "--budget_tier",
            "full",
            "--tasks_per_epoch",
            "1",
            "--k",
            "1",
            "--min_effective_tasks",
            "1",
            "--min_effective_tasks_patience",
            "1",
            "--log_path",
            str(run_dir),
        ]
    )
    assert rc == 0
    rows = [json.loads(line) for line in (run_dir / "epoch_summary.jsonl").read_text().splitlines()]
    statuses = [row.get("status") for row in rows]
    assert "halted_low_gradient_signal" in statuses
    epoch_row = next(row for row in rows if row.get("epoch") == 0 and "gradient_signal" in row)
    assert epoch_row["gradient_signal"]["effective_tasks"] == 0
    assert epoch_row["gradient_signal"]["required_effective_tasks"] == 1


def test_mini_reprofile_refreshes_profiles_and_logs_events(tmp_path: Path, monkeypatch):
    split_train = tmp_path / "split_train.json"
    split_eval = tmp_path / "split_eval.json"
    _write_split(split_train)
    _write_split(split_eval)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)
    counter = {"i": 0}

    def _fake_mutate(
        self,
        seed_task,
        *,
        epoch,
        seed_problem_id,
        target_category,
        level,
        target_speedup_band,
        failure_exemplars,
        solver_trace_summary,
        mutation_instruction,
        decision_mode,
        reason_code,
        teacher_seed_rationale,
    ):
        counter["i"] += 1
        return MutatedTask(
            task_id=f"mut_{epoch}_{counter['i']}",
            parent_task_id=f"seed_{seed_problem_id}",
            seed_problem_id=seed_problem_id,
            name=f"mut_task_{seed_problem_id}",
            reference_code=seed_task.reference_code + f"\n# mut {counter['i']}",
            interface_signature_hash="sig",
            category_tags=("activation",),
            category_id=target_category or "activation",
            mutation_backend="dry_run",
            mutation_model_id="dry_run",
            mutation_prompt_hash=f"prompt_{counter['i']}",
            novelty_hash=f"novel_{counter['i']}",
            epoch_created=epoch,
            mutation_type="logic_restructuring",
            optimization_prompt="focus memory access",
            teacher_decision_mode=decision_mode,
            teacher_reason_code=reason_code,
            teacher_target_speedup_band=target_speedup_band,
            teacher_mutation_instruction=mutation_instruction,
            solver_trace_summary=solver_trace_summary,
            teacher_failure_entry_ids=tuple(
                str(row.get("entry_id", "")) for row in (failure_exemplars or [])
            ),
            teacher_seed_rationale=teacher_seed_rationale or "",
        )

    monkeypatch.setattr(adaptive_train.KernelMutator, "mutate", _fake_mutate)

    run_dir = tmp_path / "run_mini_reprofile"
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
            "2",
            "--k",
            "1",
            "--mini_reprofile_every",
            "1",
            "--mini_reprofile_probe_tasks",
            "1",
            "--mini_reprofile_k",
            "1",
            "--budget_tier",
            "full",
            "--log_path",
            str(run_dir),
        ]
    )
    assert rc == 0
    summary = [json.loads(line) for line in (run_dir / "epoch_summary.jsonl").read_text().splitlines()]
    assert summary
    assert summary[0]["curriculum"]["mini_reprofile_events"] >= 1
    kpi_rows = [json.loads(line) for line in (run_dir / "kpi_dashboard.jsonl").read_text().splitlines()]
    assert kpi_rows[0]["mini_reprofile_events"] >= 1
    profiles = [json.loads(line) for line in (run_dir / "capability_profiles.jsonl").read_text().splitlines()]
    assert any(row["split"] == "eval_rolling" for row in profiles)


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


def test_select_bootstrap_seed_task_prefers_replay_positive(tmp_path: Path):
    replay = adaptive_train.ReplayBuffer(tmp_path / "replay.jsonl")
    replay.append(
        ReplayEntry(
            entry_id="seed1",
            task_id="task_seed_1",
            parent_task_id=None,
            problem_id=77,
            level=2,
            category_id="composite:matmul+activation",
            task_reference_code=_fake_task(77).reference_code,
            kernel_code="return x",
            eval_result=EvalResult(
                compiled=True,
                correctness=True,
                runtime_us=1.0,
                ref_runtime_us=1.5,
                speedup=1.5,
                metadata={},
            ),
            reward=1.5,
            sampler_path="sampler://seed",
            backend="tinker",
            timestamp=1.0,
            epoch=0,
            is_mutated=True,
        )
    )
    handles = [
        adaptive_train.TaskHandle(
            task=_fake_task(1),
            level=1,
            category_tags=("unknown",),
            category_id="unknown",
        )
    ]
    seed_task, _, seed_problem_id, seed_category, seed_level, source = (
        adaptive_train._select_bootstrap_seed_task(
            replay_buffer=replay,
            ranked_train_tasks=handles,
            replay_recency_window=200,
        )
    )
    assert source == "replay_positive"
    assert seed_problem_id == 77
    assert seed_category == "composite:matmul+activation"
    assert seed_level == 2
    assert seed_task.problem_id == 77


def test_select_bootstrap_seed_task_falls_back_to_l1_anchor(tmp_path: Path):
    replay = adaptive_train.ReplayBuffer(tmp_path / "replay.jsonl")
    handles = [
        adaptive_train.TaskHandle(
            task=_fake_task(5),
            level=2,
            category_tags=("unknown",),
            category_id="unknown",
        ),
        adaptive_train.TaskHandle(
            task=_fake_task(2),
            level=1,
            category_tags=("unknown",),
            category_id="unknown",
        ),
    ]
    seed_task, _, seed_problem_id, _, seed_level, source = adaptive_train._select_bootstrap_seed_task(
        replay_buffer=replay,
        ranked_train_tasks=handles,
        replay_recency_window=200,
    )
    assert source == "l1_anchor"
    assert seed_level == 1
    assert seed_problem_id == 2
    assert seed_task.problem_id == 2


def test_bootstrap_mode_routes_through_mutate_solve_replay(tmp_path: Path, monkeypatch):
    split_train = tmp_path / "split_train.json"
    split_eval = tmp_path / "split_eval.json"
    _write_split(split_train)
    _write_split(split_eval)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)

    def _profile_all_too_hard(
        solver,
        eval_tasks,
        *,
        profile_k,
        temperature,
        max_tokens,
        eval_workers,
    ):
        rows = []
        for handle in eval_tasks:
            rows.append(
                {
                    "task_id": f"{handle.task.problem_id}",
                    "sample_idx": 0,
                    "category_id": handle.category_id,
                    "correctness": True,
                    "speedup": 1.01,
                    "runtime_us": 1000.0,
                    "fast_1": 1.0,
                }
            )
        return rows, 0.01

    counter = {"i": 0}

    def _fake_mutate(
        self,
        seed_task,
        *,
        epoch,
        seed_problem_id,
        target_category,
        level,
        target_speedup_band,
        failure_exemplars,
        solver_trace_summary,
        mutation_instruction,
        decision_mode,
        reason_code,
        teacher_seed_rationale,
    ):
        counter["i"] += 1
        return MutatedTask(
            task_id=f"mut_{epoch}_{counter['i']}",
            parent_task_id=f"seed_{seed_problem_id}",
            seed_problem_id=seed_problem_id,
            name=f"mut_task_{seed_problem_id}",
            reference_code=seed_task.reference_code + f"\n# mut {counter['i']}",
            interface_signature_hash="sig",
            category_tags=("activation",),
            category_id=target_category or "activation",
            mutation_backend="dry_run",
            mutation_model_id="dry_run",
            mutation_prompt_hash=f"prompt_{counter['i']}",
            novelty_hash=f"novel_{counter['i']}",
            epoch_created=epoch,
            mutation_type="logic_restructuring",
            optimization_prompt="focus memory access",
            teacher_decision_mode=decision_mode,
            teacher_reason_code=reason_code,
            teacher_target_speedup_band=target_speedup_band,
            teacher_mutation_instruction=mutation_instruction,
            solver_trace_summary=solver_trace_summary,
            teacher_failure_entry_ids=tuple(
                str(row.get("entry_id", "")) for row in (failure_exemplars or [])
            ),
            teacher_seed_rationale=teacher_seed_rationale or "",
        )

    monkeypatch.setattr(adaptive_train, "_profile_solver", _profile_all_too_hard)
    monkeypatch.setattr(adaptive_train.KernelMutator, "mutate", _fake_mutate)

    run_dir = tmp_path / "run_bootstrap"
    rc = adaptive_train.main(
        [
            "--dry_run",
            "--no-enable_admission_gate",
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
            "--mini_reprofile_every",
            "0",
            "--budget_tier",
            "full",
            "--log_path",
            str(run_dir),
        ]
    )
    assert rc == 0
    summary = [json.loads(line) for line in (run_dir / "epoch_summary.jsonl").read_text().splitlines()]
    assert summary
    assert summary[0]["records_added"] > 0
    assert summary[0]["curriculum"]["bootstrap_mode_entered"] is True
    assert summary[0]["curriculum"]["bootstrap_count"] >= 1
    assert summary[0]["curriculum"]["bootstrap_seed_source_counts"]["l1_anchor"] >= 1
    events = [json.loads(line) for line in (run_dir / "mutation_events.jsonl").read_text().splitlines()]
    assert any(row.get("status") == "mutated" for row in events)
    assert any(row.get("bootstrap_mode") is True for row in events)


def test_admission_gate_rejection_logs_and_skips_solver(tmp_path: Path, monkeypatch):
    split_train = tmp_path / "split_train.json"
    split_eval = tmp_path / "split_eval.json"
    _write_split(split_train)
    _write_split(split_eval)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)
    counter = {"i": 0}

    def _fake_mutate(
        self,
        seed_task,
        *,
        epoch,
        seed_problem_id,
        target_category,
        level,
        target_speedup_band,
        failure_exemplars,
        solver_trace_summary,
        mutation_instruction,
        decision_mode,
        reason_code,
        teacher_seed_rationale,
    ):
        counter["i"] += 1
        return MutatedTask(
            task_id=f"mut_{epoch}_{counter['i']}",
            parent_task_id=f"seed_{seed_problem_id}",
            seed_problem_id=seed_problem_id,
            name=f"mut_task_{seed_problem_id}",
            reference_code=seed_task.reference_code + f"\n# mut {counter['i']}",
            interface_signature_hash="sig",
            category_tags=("activation",),
            category_id=target_category or "activation",
            mutation_backend="dry_run",
            mutation_model_id="dry_run",
            mutation_prompt_hash=f"prompt_{counter['i']}",
            novelty_hash=f"novel_{counter['i']}",
            epoch_created=epoch,
            mutation_type="logic_restructuring",
            optimization_prompt="focus memory access",
            teacher_decision_mode=decision_mode,
            teacher_reason_code=reason_code,
            teacher_target_speedup_band=target_speedup_band,
            teacher_mutation_instruction=mutation_instruction,
            solver_trace_summary=solver_trace_summary,
            teacher_failure_entry_ids=tuple(
                str(row.get("entry_id", "")) for row in (failure_exemplars or [])
            ),
            teacher_seed_rationale=teacher_seed_rationale or "",
        )

    def _reject_all(*args, **kwargs):
        return {
            "admitted": False,
            "success_count": 0,
            "total_count": 8,
            "success_rate": 0.0,
            "wall_clock_s": 0.01,
            "eval_results": [],
        }

    monkeypatch.setattr(adaptive_train.KernelMutator, "mutate", _fake_mutate)
    monkeypatch.setattr(adaptive_train, "_run_admission_gate", _reject_all)

    run_dir = tmp_path / "run_admission_reject"
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
            "--budget_tier",
            "full",
            "--log_path",
            str(run_dir),
        ]
    )
    assert rc == 0
    summary = [json.loads(line) for line in (run_dir / "epoch_summary.jsonl").read_text().splitlines()]
    assert summary
    assert summary[0]["records_added"] == 0
    assert summary[0]["curriculum"]["admission_rejected_tasks"] >= 1
    assert summary[0]["curriculum"]["admission_attempted_tasks"] >= 1
    events = [json.loads(line) for line in (run_dir / "mutation_events.jsonl").read_text().splitlines()]
    assert any(row.get("status") == "admission_rejected" for row in events)
    kpi_rows = [json.loads(line) for line in (run_dir / "kpi_dashboard.jsonl").read_text().splitlines()]
    assert kpi_rows[0]["admission_rejected_tasks"] >= 1
    assert kpi_rows[0]["solver_rollouts_skipped_by_admission"] >= 1


def test_admission_gate_can_be_disabled(tmp_path: Path, monkeypatch):
    split_train = tmp_path / "split_train.json"
    split_eval = tmp_path / "split_eval.json"
    _write_split(split_train)
    _write_split(split_eval)
    monkeypatch.setattr(adaptive_train, "load_task", _fake_task)
    counter = {"i": 0}

    def _fake_mutate(
        self,
        seed_task,
        *,
        epoch,
        seed_problem_id,
        target_category,
        level,
        target_speedup_band,
        failure_exemplars,
        solver_trace_summary,
        mutation_instruction,
        decision_mode,
        reason_code,
        teacher_seed_rationale,
    ):
        counter["i"] += 1
        return MutatedTask(
            task_id=f"mut_{epoch}_{counter['i']}",
            parent_task_id=f"seed_{seed_problem_id}",
            seed_problem_id=seed_problem_id,
            name=f"mut_task_{seed_problem_id}",
            reference_code=seed_task.reference_code + f"\n# mut {counter['i']}",
            interface_signature_hash="sig",
            category_tags=("activation",),
            category_id=target_category or "activation",
            mutation_backend="dry_run",
            mutation_model_id="dry_run",
            mutation_prompt_hash=f"prompt_{counter['i']}",
            novelty_hash=f"novel_{counter['i']}",
            epoch_created=epoch,
            mutation_type="logic_restructuring",
            optimization_prompt="focus memory access",
            teacher_decision_mode=decision_mode,
            teacher_reason_code=reason_code,
            teacher_target_speedup_band=target_speedup_band,
            teacher_mutation_instruction=mutation_instruction,
            solver_trace_summary=solver_trace_summary,
            teacher_failure_entry_ids=tuple(
                str(row.get("entry_id", "")) for row in (failure_exemplars or [])
            ),
            teacher_seed_rationale=teacher_seed_rationale or "",
        )

    monkeypatch.setattr(adaptive_train.KernelMutator, "mutate", _fake_mutate)

    run_dir = tmp_path / "run_admission_disabled"
    rc = adaptive_train.main(
        [
            "--dry_run",
            "--no-enable_admission_gate",
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
            "--budget_tier",
            "full",
            "--log_path",
            str(run_dir),
        ]
    )
    assert rc == 0
    summary = [json.loads(line) for line in (run_dir / "epoch_summary.jsonl").read_text().splitlines()]
    assert summary
    assert summary[0]["records_added"] > 0
    assert summary[0]["curriculum"]["admission_gate_enabled"] is False
    assert summary[0]["curriculum"]["admission_attempted_tasks"] == 0
