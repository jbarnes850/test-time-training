from src.env.schema import KernelTask
from src.env.solver import DryRunSolverBackend


def _task(problem_id: int) -> KernelTask:
    return KernelTask(
        problem_id=problem_id,
        name=f"task_{problem_id}",
        reference_code="""
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return x + 1
""",
    )


def test_dry_run_solver_backend_solves_and_trains():
    solver = DryRunSolverBackend()
    outcome = solver.solve_task(
        _task(7),
        k=3,
        temperature=0.2,
        max_tokens=64,
        level=1,
        eval_workers=1,
    )
    assert len(outcome.eval_results) == 3
    assert all(r.correctness for r in outcome.eval_results)
    assert all(r.speedup >= 1.0 for r in outcome.eval_results)

    new_path = solver.train_on_outcomes([outcome], epoch=2)
    assert new_path.endswith("epoch_2")
    assert solver.sampler_path == new_path


def test_dry_run_solver_metadata_shape():
    solver = DryRunSolverBackend()
    md = solver.metadata()
    assert md["backend"] == "dry_run"
    assert md["provider"] == "local"
    assert "model_id" in md
    assert "sampler_path" in md
    assert md["training_enabled"] is False
