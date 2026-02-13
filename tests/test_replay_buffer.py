from pathlib import Path

from src.env.replay_buffer import ReplayBuffer
from src.env.schema import EvalResult, ReplayEntry


def _make_eval(correct: bool, speedup: float, banked: bool = False) -> EvalResult:
    return EvalResult(
        compiled=True,
        correctness=correct,
        runtime_us=1.0,
        ref_runtime_us=max(speedup, 1e-6),
        speedup=speedup,
        metadata={"banked": banked} if banked else {},
    )


def _entry(
    entry_id: str,
    category_id: str,
    timestamp: float,
    speedup: float,
    correct: bool = True,
    banked: bool = False,
) -> ReplayEntry:
    return ReplayEntry(
        entry_id=entry_id,
        task_id=f"task-{entry_id}",
        parent_task_id=None,
        problem_id=1,
        level=1,
        category_id=category_id,
        task_reference_code="ref",
        kernel_code="kernel",
        eval_result=_make_eval(correct=correct, speedup=speedup, banked=banked),
        reward=speedup if correct else 0.0,
        sampler_path="sampler",
        backend="solver",
        timestamp=timestamp,
        epoch=1,
        is_mutated=False,
    )


def test_replay_buffer_append_and_reload(tmp_path: Path):
    path = tmp_path / "replay.jsonl"
    buffer = ReplayBuffer(path)
    buffer.append(_entry("1", "conv", 1.0, 2.0))
    buffer.append(_entry("2", "conv", 2.0, 1.5))
    buffer.append(_entry("3", "matmul", 3.0, 0.9, correct=False))
    assert len(buffer) == 3

    reloaded = ReplayBuffer(path)
    assert len(reloaded) == 3
    ids = [e.entry_id for e in reloaded.entries()]
    assert ids == ["1", "2", "3"]


def test_replay_buffer_query_filters(tmp_path: Path):
    path = tmp_path / "replay.jsonl"
    buffer = ReplayBuffer(path)
    buffer.append(_entry("1", "conv", 1.0, 2.0))
    buffer.append(_entry("2", "conv", 2.0, 1.2))
    buffer.append(_entry("3", "matmul", 3.0, 3.0))
    buffer.append(_entry("4", "conv", 4.0, 0.5, correct=False))

    out = buffer.query(category_id="conv", min_speedup=1.3, correct_only=True)
    assert [e.entry_id for e in out] == ["1"]

    recent = buffer.query(recency_window=2)
    assert [e.entry_id for e in recent] == ["4", "3"]

    limited = buffer.query(limit=2)
    assert [e.entry_id for e in limited] == ["4", "3"]


def test_replay_buffer_select_seed_respects_banked(tmp_path: Path):
    path = tmp_path / "replay.jsonl"
    buffer = ReplayBuffer(path)
    buffer.append(_entry("1", "conv", 1.0, 2.5, banked=True))
    buffer.append(_entry("2", "conv", 2.0, 2.0))

    seed = buffer.select_seed({"category_id": "conv", "correct_only": True})
    assert seed is not None
    assert seed.entry_id == "2"

    seed_allow_banked = buffer.select_seed(
        {"category_id": "conv", "correct_only": True},
        exclude_banked=False,
    )
    assert seed_allow_banked is not None
    assert seed_allow_banked.entry_id == "2"
