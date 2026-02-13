from src.env.teacher import CurriculumTeacher, category_id, infer_task_categories


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
