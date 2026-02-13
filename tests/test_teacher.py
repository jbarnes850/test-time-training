from src.env.schema import CapabilityProfile
from src.env.teacher import (
    CurriculumTeacher,
    HeuristicTeacherBackend,
    TinkerLLMTeacherBackend,
    category_id,
    infer_task_categories,
)


class _DummyTinker:
    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


class _DummyRenderer:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def build_generation_prompt(self, messages):
        return str(messages)

    def get_stop_sequences(self):
        return []

    def parse_response(self, _tokens):
        return {"content": self._response_text}, {}


class _DummySequence:
    def __init__(self):
        self.tokens = []


class _DummyResult:
    def __init__(self, has_sequence: bool = True):
        self.sequences = [_DummySequence()] if has_sequence else []


class _DummyFuture:
    def __init__(self, result_obj):
        self._result_obj = result_obj

    def result(self, timeout=None):
        return self._result_obj


class _DummySamplingClient:
    def __init__(self, result_obj):
        self._result_obj = result_obj

    def sample(self, **_kwargs):
        return _DummyFuture(self._result_obj)


def _build_mock_tinker_backend(response_text: str, *, has_sequence: bool = True):
    backend = object.__new__(TinkerLLMTeacherBackend)
    backend._tinker = _DummyTinker()
    backend._get_text_content = lambda msg: msg.get("content", "")
    backend._model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    backend._renderer_name = "qwen3_instruct"
    backend._temperature = 0.1
    backend._max_tokens = 128
    backend._request_timeout_s = 5.0
    backend._fallback_backend = HeuristicTeacherBackend()
    backend._sampling_client = _DummySamplingClient(_DummyResult(has_sequence=has_sequence))
    backend._renderer = _DummyRenderer(response_text)
    backend._resolved_model_path = "base_model:Qwen/Qwen3-30B-A3B-Instruct-2507"
    return backend


def _profiles_for_backend() -> list[CapabilityProfile]:
    return [
        CapabilityProfile(
            epoch=1,
            split="eval",
            category_id="conv",
            n_tasks=2,
            correctness_rate=1.0,
            mean_speedup=1.2,
            speedup_var=0.0,
            fast_1_rate=1.0,
            failure_rate=0.0,
            sample_count=2,
        ),
        CapabilityProfile(
            epoch=1,
            split="eval",
            category_id="composite:activation+matmul",
            n_tasks=2,
            correctness_rate=0.2,
            mean_speedup=1.05,
            speedup_var=0.1,
            fast_1_rate=0.2,
            failure_rate=0.8,
            sample_count=2,
        ),
    ]


def test_infer_task_categories_composite():
    code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3)
        self.ln = nn.LayerNorm([16, 14, 14])

    def forward(self, x):
        y = self.conv(x)
        y = F.relu(y)
        return self.ln(y)
"""
    tags = infer_task_categories(code)
    assert tags == {"conv", "activation", "normalization"}
    assert category_id(tags) == "composite:activation+conv+normalization"


def test_infer_task_categories_fallback_unknown():
    tags = infer_task_categories("def broken(:")
    assert tags == {"unknown"}
    assert category_id(tags) == "unknown"


def test_update_profile_aggregates_metrics():
    teacher = CurriculumTeacher(seed=7)
    rows = [
        {"task_id": "a", "category_id": "conv", "correctness": True, "speedup": 2.0, "fast_1": 1.0},
        {"task_id": "a", "category_id": "conv", "correctness": False, "speedup": 0.0, "fast_1": 0.0},
        {"task_id": "b", "category_id": "conv", "correctness": True, "speedup": 1.5, "fast_1": 1.0},
        {"task_id": "c", "category_id": "matmul", "correctness": True, "speedup": 3.0, "fast_1": 1.0},
    ]
    profiles = teacher.update_profile(rows, epoch=2, split="eval")
    by_cat = {p.category_id: p for p in profiles}

    conv = by_cat["conv"]
    assert conv.n_tasks == 2
    assert conv.sample_count == 3
    assert round(conv.correctness_rate, 4) == 0.6667
    assert round(conv.mean_speedup, 4) == 1.1667
    assert round(conv.fast_1_rate, 4) == 0.6667

    matmul = by_cat["matmul"]
    assert matmul.correctness_rate == 1.0
    assert matmul.speedup_var == 0.0


def test_rank_tasks_inverse_correctness_prioritizes_hard_categories():
    teacher = CurriculumTeacher(seed=11)
    rows = [
        {"task_id": "a", "category_id": "conv", "correctness": True, "speedup": 1.2},
        {"task_id": "b", "category_id": "conv", "correctness": True, "speedup": 1.1},
        {"task_id": "c", "category_id": "matmul", "correctness": False, "speedup": 0.0},
        {"task_id": "d", "category_id": "matmul", "correctness": False, "speedup": 0.0},
    ]
    teacher.update_profile(rows, epoch=1)
    tasks = [
        {"task_id": "t1", "category_id": "conv"},
        {"task_id": "t2", "category_id": "matmul"},
    ]
    ranked = teacher.rank_tasks(tasks, strategy="inverse_correctness")
    assert ranked[0]["category_id"] == "matmul"


def test_rank_tasks_random_is_seeded():
    teacher_a = CurriculumTeacher(seed=123)
    teacher_b = CurriculumTeacher(seed=123)
    tasks = [
        {"task_id": "t1", "category_id": "conv"},
        {"task_id": "t2", "category_id": "matmul"},
        {"task_id": "t3", "category_id": "activation"},
    ]
    ranked_a = teacher_a.rank_tasks(tasks, strategy="random")
    ranked_b = teacher_b.rank_tasks(tasks, strategy="random")
    assert [t["task_id"] for t in ranked_a] == [t["task_id"] for t in ranked_b]


def test_rank_tasks_easy_to_hard_prefers_high_correctness_first():
    teacher = CurriculumTeacher(seed=5)
    rows = [
        {"task_id": "a", "category_id": "conv", "correctness": True, "speedup": 2.0},
        {"task_id": "b", "category_id": "conv", "correctness": True, "speedup": 2.1},
        {"task_id": "c", "category_id": "matmul", "correctness": False, "speedup": 0.0},
    ]
    teacher.update_profile(rows, epoch=1)
    tasks = [
        {"task_id": "t1", "category_id": "matmul"},
        {"task_id": "t2", "category_id": "conv"},
    ]
    ranked = teacher.rank_tasks(tasks, strategy="easy_to_hard_static")
    assert ranked[0]["category_id"] == "conv"


def test_heuristic_teacher_selects_in_band_frontier_target():
    backend = HeuristicTeacherBackend()
    teacher = CurriculumTeacher(
        seed=9,
        policy_backend=backend,
        target_min_completion=0.25,
        target_max_completion=0.75,
    )
    rows = [
        {"task_id": "a", "category_id": "conv", "correctness": True, "speedup": 1.2},
        {"task_id": "b", "category_id": "conv", "correctness": True, "speedup": 1.1},
        {"task_id": "c", "category_id": "matmul", "correctness": False, "speedup": 0.0},
        {"task_id": "d", "category_id": "matmul", "correctness": True, "speedup": 1.4},
        {"task_id": "e", "category_id": "pooling", "correctness": False, "speedup": 0.0},
    ]
    teacher.update_profile(rows, epoch=1)
    decision = teacher.select_frontier_target()
    assert decision.target_category == "matmul"
    assert decision.backend == "heuristic"
    assert decision.hard_frontier is False


def test_heuristic_teacher_marks_hard_frontier_when_below_band():
    backend = HeuristicTeacherBackend()
    teacher = CurriculumTeacher(
        seed=9,
        policy_backend=backend,
        target_min_completion=0.25,
        target_max_completion=0.75,
    )
    rows = [
        {"task_id": "a", "category_id": "conv", "correctness": True, "speedup": 1.2},
        {"task_id": "b", "category_id": "conv", "correctness": True, "speedup": 1.1},
        {"task_id": "c", "category_id": "matmul", "correctness": False, "speedup": 0.0},
        {"task_id": "d", "category_id": "matmul", "correctness": False, "speedup": 0.0},
    ]
    teacher.update_profile(rows, epoch=1)
    decision = teacher.select_frontier_target()
    assert decision.target_category == "matmul"
    assert decision.hard_frontier is True


def test_tinker_teacher_backend_parses_json_and_infers_hard_frontier():
    response_text = (
        '{"target_category":"composite:activation+matmul",'
        '"hard_frontier":false,'
        '"rationale":"Targeting hard composite frontier."}'
    )
    backend = _build_mock_tinker_backend(response_text)
    decision = backend.decide(
        _profiles_for_backend(),
        target_min_completion=0.25,
        target_max_completion=0.75,
    )
    assert decision.backend == "tinker"
    assert decision.target_category == "composite:activation+matmul"
    # Decision must be inferred from measured profile, not trusted from model payload.
    assert decision.hard_frontier is True


def test_tinker_teacher_backend_falls_back_on_invalid_category():
    backend = _build_mock_tinker_backend('{"target_category":"unknown","hard_frontier":false}')
    decision = backend.decide(
        _profiles_for_backend(),
        target_min_completion=0.25,
        target_max_completion=0.75,
    )
    assert decision.backend == "heuristic"
    assert decision.target_category == "composite:activation+matmul"
    assert decision.hard_frontier is True
