import asyncio
import tinker

from src.tinker_env import KernelEnv
from src.env.schema import EvalResult, KernelTask


class FakeRenderer:
    def build_generation_prompt(self, messages):
        text = "\n".join(m["content"] for m in messages)
        tokens = [ord(c) for c in text][:128]
        return tinker.ModelInput.from_ints(tokens)

    def get_stop_sequences(self):
        return ["\n\n"]

    def parse_response(self, action):
        content = "".join(chr(tok) for tok in action if tok < 256)
        return {"role": "assistant", "content": content}, True


class DummyEval:
    @staticmethod
    def evaluate_kernel(problem_id, kernel_code, level=1):
        return EvalResult(
            compiled=True,
            correctness=True,
            runtime_us=1.0,
            ref_runtime_us=2.0,
            speedup=2.0,
            metadata={},
        )


def test_initial_observation(monkeypatch):
    monkeypatch.setattr(
        "src.tinker_env.load_task",
        lambda problem_id, level=1: KernelTask(problem_id=problem_id, name="x", reference_code="code"),
    )
    env = KernelEnv(problem_id=100, renderer=FakeRenderer(), level=1)
    ob, stop = asyncio.run(env.initial_observation())
    assert isinstance(ob, tinker.ModelInput)
    assert stop == ["\n\n"]


def test_step_reward(monkeypatch):
    monkeypatch.setattr("src.tinker_env.evaluate_kernel", DummyEval.evaluate_kernel)
    monkeypatch.setattr(
        "src.tinker_env.load_task",
        lambda problem_id, level=1: KernelTask(problem_id=problem_id, name="x", reference_code="code"),
    )
    env = KernelEnv(problem_id=100, renderer=FakeRenderer(), level=1)
    result = asyncio.run(env.step([ord('a')]))
    assert result.reward > 0
    assert result.episode_done is True
