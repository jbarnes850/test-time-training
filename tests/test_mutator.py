from pathlib import Path

import pytest

from src.env.mutator import (
    ALLOWED_MUTATION_TYPES,
    ApiMutatorBackend,
    KernelMutator,
    MutationValidationResult,
    parse_mutation_response,
)
from src.env.replay_buffer import ReplayBuffer
from src.env.schema import EvalResult, KernelTask, ReplayEntry


SEED_CODE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return x + 1
"""

VALID_MUTATION = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def forward(self, x):
        y = torch.relu(x)
        return y + 1
"""

VALID_MUTATION_RESPONSE = f"""```python
{VALID_MUTATION}
```
MUTATION_TYPE: logic_restructuring
OPTIMIZATION_PROMPT: Improve dataflow and fusion opportunities while preserving semantics.
"""


class DummyBackend:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls = 0

    @property
    def backend_name(self) -> str:
        return "dummy"

    @property
    def model_id(self) -> str:
        return "dummy/model"

    def generate_mutation(self, seed_task, prompt, *, system_prompt, temperature, max_tokens) -> str:
        idx = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return self.outputs[idx]


def _seed_task() -> KernelTask:
    return KernelTask(problem_id=4, name="seed", reference_code=SEED_CODE)


def _append_existing_mutation(buffer: ReplayBuffer, code: str) -> None:
    eval_result = EvalResult(
        compiled=True,
        correctness=True,
        runtime_us=1.0,
        ref_runtime_us=2.0,
        speedup=2.0,
        metadata={},
    )
    buffer.append(
        ReplayEntry(
            entry_id="existing",
            task_id="task-existing",
            parent_task_id=None,
            problem_id=4,
            level=1,
            category_id="activation",
            task_reference_code=code,
            kernel_code="kernel",
            eval_result=eval_result,
            reward=2.0,
            sampler_path="sampler",
            backend="solver",
            timestamp=1.0,
            epoch=1,
            is_mutated=True,
        )
    )


def test_validate_mutation_parse_failure(tmp_path: Path):
    replay = ReplayBuffer(tmp_path / "replay.jsonl")
    backend = DummyBackend(["def broken(:"])
    mutator = KernelMutator(backend, replay)

    result = mutator.validate_mutation(_seed_task(), "def broken(:")
    assert isinstance(result, MutationValidationResult)
    assert result.valid is False
    assert result.stage == "parse"


def test_validate_mutation_interface_mismatch(tmp_path: Path):
    replay = ReplayBuffer(tmp_path / "replay.jsonl")
    backend = DummyBackend([""])
    mutator = KernelMutator(backend, replay)

    bad_interface = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, y):
        return x + y
"""
    result = mutator.validate_mutation(_seed_task(), bad_interface)
    assert result.valid is False
    assert result.stage == "interface"


def test_validate_mutation_structural_identity(tmp_path: Path):
    replay = ReplayBuffer(tmp_path / "replay.jsonl")
    backend = DummyBackend([""])
    mutator = KernelMutator(backend, replay)

    result = mutator.validate_mutation(_seed_task(), SEED_CODE)
    assert result.valid is False
    assert result.stage == "structural"


def test_validate_mutation_novelty_collision(tmp_path: Path):
    replay = ReplayBuffer(tmp_path / "replay.jsonl")
    _append_existing_mutation(replay, VALID_MUTATION)
    backend = DummyBackend([""])
    mutator = KernelMutator(backend, replay)

    result = mutator.validate_mutation(_seed_task(), VALID_MUTATION)
    assert result.valid is False
    assert result.stage == "novelty"


def test_mutate_returns_mutated_task_with_lineage(tmp_path: Path):
    replay = ReplayBuffer(tmp_path / "replay.jsonl")
    backend = DummyBackend(["bad response", VALID_MUTATION_RESPONSE])
    mutator = KernelMutator(backend, replay, max_retries=3)

    mutated = mutator.mutate(
        _seed_task(),
        epoch=2,
        decision_mode="learning",
        reason_code="edge_signal",
        target_speedup_band=(1.3, 1.8),
        mutation_instruction="Add one operation while preserving interface.",
        solver_trace_summary="category=activation mean_speedup=1.3",
    )
    assert mutated is not None
    assert mutated.seed_problem_id == 4
    assert mutated.mutation_backend == "dummy"
    assert mutated.mutation_model_id == "dummy/model"
    assert mutated.mutation_type in ALLOWED_MUTATION_TYPES
    assert mutated.optimization_prompt
    assert "activation" in mutated.category_tags
    assert mutated.category_id in {"activation", "composite:activation+reduction"}
    assert mutated.teacher_decision_mode == "learning"
    assert mutated.teacher_reason_code == "edge_signal"
    assert mutated.teacher_target_speedup_band == (1.3, 1.8)
    assert mutated.teacher_mutation_instruction
    assert mutated.solver_trace_summary
    assert mutator.stats.attempts_total == 2
    assert mutator.stats.accepted == 1
    assert mutator.stats.format_failures == 1


def test_mutate_returns_none_when_retries_exhausted(tmp_path: Path):
    replay = ReplayBuffer(tmp_path / "replay.jsonl")
    backend = DummyBackend(["invalid output"])
    mutator = KernelMutator(backend, replay, max_retries=2)

    mutated = mutator.mutate(_seed_task(), epoch=3)
    assert mutated is None
    assert backend.calls == 2
    assert mutator.stats.format_failures == 2


def test_parse_mutation_response_success():
    proposal = parse_mutation_response(VALID_MUTATION_RESPONSE)
    assert proposal is not None
    assert proposal.code.strip().startswith("import torch")
    assert proposal.mutation_type == "logic_restructuring"
    assert proposal.optimization_prompt.startswith("Improve")


def test_parse_mutation_response_missing_fields():
    proposal = parse_mutation_response("```python\nprint('x')\n```")
    assert proposal is None


def test_api_backend_is_stub():
    backend = ApiMutatorBackend(model_id="provider/model")
    with pytest.raises(NotImplementedError):
        backend.generate_mutation(
            _seed_task(),
            "prompt",
            system_prompt="system",
            temperature=0.2,
            max_tokens=64,
        )
