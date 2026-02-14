from pathlib import Path

from src.env.training_replay import TrainingReplayBuffer


def _add_outcome(buf: TrainingReplayBuffer, outcome_id: str, epoch: int, zone: str = "learning", k: int = 2):
    buf.add_outcome(
        outcome_id=outcome_id,
        epoch=epoch,
        zone=zone,
        utility_score=0.7,
        category_id="activation",
        problem_id=1,
        level=1,
        prompt_tokens=[100, 200, 300],
        sampled_tokens_list=[[10, 20, 30] for _ in range(k)],
        sampled_logprobs_list=[[-0.5, -0.3, -0.1] for _ in range(k)],
        rewards=[1.0 + i * 0.1 for i in range(k)],
        entry_ids=[f"{outcome_id}_s{i}" for i in range(k)],
        sampler_path="sampler://test",
    )


def test_add_persist_reload_roundtrip(tmp_path: Path):
    path = tmp_path / "artifacts.jsonl"
    buf = TrainingReplayBuffer(path)
    _add_outcome(buf, "o1", epoch=0)
    assert len(buf) == 2

    reloaded = TrainingReplayBuffer(path)
    assert len(reloaded) == 2
    a = reloaded.query_by_epoch(0)
    assert len(a) == 2
    assert a[0].outcome_id == "o1"
    assert a[0].prompt_tokens == [100, 200, 300]
    assert a[0].sampled_tokens == [10, 20, 30]
    assert a[0].sampled_logprobs == [-0.5, -0.3, -0.1]


def test_dedup_on_entry_id(tmp_path: Path):
    path = tmp_path / "artifacts.jsonl"
    buf = TrainingReplayBuffer(path)
    _add_outcome(buf, "o1", epoch=0)
    _add_outcome(buf, "o1", epoch=0)
    assert len(buf) == 2


def test_sample_by_zone_filter(tmp_path: Path):
    path = tmp_path / "artifacts.jsonl"
    buf = TrainingReplayBuffer(path)
    _add_outcome(buf, "o1", epoch=0, zone="learning")
    _add_outcome(buf, "o2", epoch=0, zone="mastered")
    _add_outcome(buf, "o3", epoch=1, zone="learning")

    learning = buf.sample_replay(current_epoch=2, recency_epochs=3, zone_filter="learning")
    assert all(a.zone == "learning" for a in learning)
    assert len(learning) == 4

    mastered = buf.sample_replay(current_epoch=2, recency_epochs=3, zone_filter="mastered")
    assert len(mastered) == 2


def test_sample_by_recency_window(tmp_path: Path):
    path = tmp_path / "artifacts.jsonl"
    buf = TrainingReplayBuffer(path)
    _add_outcome(buf, "o0", epoch=0, zone="learning")
    _add_outcome(buf, "o1", epoch=1, zone="learning")
    _add_outcome(buf, "o2", epoch=2, zone="learning")
    _add_outcome(buf, "o3", epoch=3, zone="learning")

    recent = buf.sample_replay(current_epoch=4, recency_epochs=2, zone_filter="learning")
    epochs = {a.epoch for a in recent}
    assert epochs == {2, 3}


def test_sample_excludes_current_epoch(tmp_path: Path):
    path = tmp_path / "artifacts.jsonl"
    buf = TrainingReplayBuffer(path)
    _add_outcome(buf, "o0", epoch=5, zone="learning")

    result = buf.sample_replay(current_epoch=5, recency_epochs=3, zone_filter="learning")
    assert len(result) == 0


def test_empty_buffer_returns_empty(tmp_path: Path):
    path = tmp_path / "artifacts.jsonl"
    buf = TrainingReplayBuffer(path)
    assert buf.sample_replay(current_epoch=0) == []
    assert buf.query_by_epoch(0) == []


def test_sample_with_limit(tmp_path: Path):
    path = tmp_path / "artifacts.jsonl"
    buf = TrainingReplayBuffer(path)
    for i in range(5):
        _add_outcome(buf, f"o{i}", epoch=i, zone="learning")

    limited = buf.sample_replay(current_epoch=5, recency_epochs=5, zone_filter="learning", limit=3)
    assert len(limited) == 3
    assert limited[0].epoch >= limited[-1].epoch


def test_sample_no_zone_filter(tmp_path: Path):
    path = tmp_path / "artifacts.jsonl"
    buf = TrainingReplayBuffer(path)
    _add_outcome(buf, "o1", epoch=0, zone="learning")
    _add_outcome(buf, "o2", epoch=0, zone="mastered")

    all_zones = buf.sample_replay(current_epoch=1, recency_epochs=1, zone_filter=None)
    assert len(all_zones) == 4
