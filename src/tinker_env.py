from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from collections import deque
import json
import os
import statistics
import time
import threading

import tinker

from src.env.evaluator import evaluate_kernel, compute_reward
from src.env.tasking import load_task, build_messages
from src.utils.tinker_utils import ensure_tinker_cookbook_on_path
from src.utils.code_utils import extract_python_code, assemble_modelnew_code

ensure_tinker_cookbook_on_path()


# Streaming telemetry for real-time visibility
_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_PATH: str | None = None


def _init_telemetry() -> str | None:
    """Initialize telemetry path from environment."""
    return os.getenv("KERNELBENCH_TELEMETRY_PATH")


def _write_telemetry(record: dict) -> None:
    """Write a telemetry record to the streaming JSONL file."""
    path = _init_telemetry()
    if not path:
        return
    record["timestamp"] = time.time()
    with _TELEMETRY_LOCK:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer, get_text_content
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, StepResult


@dataclass(frozen=True)
class RewardConfig:
    normalize: bool = False
    baseline_window: int = 32
    correct_bonus: float = 0.0
    epsilon: float = 1e-6


_BASELINE_HISTORY: dict[tuple[int, int], deque[float]] = {}


def _load_reward_config() -> RewardConfig:
    mode = os.getenv("KERNELBENCH_REWARD_MODE", "raw").lower()
    normalize = mode in {"normalized", "norm"}
    baseline_window = int(os.getenv("KERNELBENCH_REWARD_BASELINE_WINDOW", "32"))
    correct_bonus = float(os.getenv("KERNELBENCH_CORRECTNESS_BONUS", "0.0"))
    return RewardConfig(
        normalize=normalize,
        baseline_window=baseline_window,
        correct_bonus=correct_bonus,
    )


def _get_baseline_median(key: tuple[int, int], cfg: RewardConfig) -> float:
    history = _BASELINE_HISTORY.get(key)
    if not history:
        return 1.0
    return float(statistics.median(history))


def _update_baseline(key: tuple[int, int], speedup: float, cfg: RewardConfig) -> None:
    history = _BASELINE_HISTORY.get(key)
    if history is None:
        history = deque(maxlen=cfg.baseline_window)
        _BASELINE_HISTORY[key] = history
    history.append(speedup)


@dataclass
class KernelEnv(Env):
    problem_id: int
    renderer: Renderer
    level: int = 1

    def _build_messages(self) -> list[dict[str, str]]:
        task = load_task(self.problem_id, level=self.level)
        return build_messages(task)

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        messages = self._build_messages()
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action: list[int]) -> StepResult:
        try:
            parsed_message, _ = self.renderer.parse_response(action)
            raw_action = extract_python_code(get_text_content(parsed_message))
            task = load_task(self.problem_id, level=self.level)
            kernel_code = assemble_modelnew_code(raw_action, task.reference_code)
        except Exception as exc:
            kernel_code = ""
            raw_action = ""
            eval_result = evaluate_kernel(self.problem_id, kernel_code, level=self.level)
            reward = 0.0
            metrics = {
                "env/reward": reward,
                "env/speedup": 0.0,
                "env/speedup_norm": 0.0,
                "env/baseline_median": 1.0,
                "env/correct": 0,
            }
            logs = {
                "problem_id": self.problem_id,
                "compiled": "False",
                "parse_error": str(exc),
                "raw_action": raw_action,
                "assembled_code": kernel_code,
            }
            # Stream telemetry for parse errors too
            _write_telemetry({
                "problem_id": self.problem_id,
                "level": self.level,
                "compiled": False,
                "correctness": False,
                "speedup": 0.0,
                "speedup_norm": 0.0,
                "reward": 0.0,
                "baseline_median": 1.0,
                "runtime_us": -1.0,
                "ref_runtime_us": -1.0,
                "parse_error": str(exc),
            })
            next_ob, next_stop = await self.initial_observation()
            return StepResult(
                reward=reward,
                episode_done=True,
                next_observation=next_ob,
                next_stop_condition=next_stop,
                metrics=metrics,
                logs=logs,
            )

        eval_result = evaluate_kernel(self.problem_id, kernel_code, level=self.level)
        cfg = _load_reward_config()
        key = (self.level, self.problem_id)
        baseline_median = _get_baseline_median(key, cfg)
        norm_speedup = eval_result.speedup if not cfg.normalize else eval_result.speedup / max(baseline_median, cfg.epsilon)
        reward = compute_reward(norm_speedup, eval_result.correctness, correct_bonus=cfg.correct_bonus)
        if eval_result.correctness:
            _update_baseline(key, eval_result.speedup, cfg)

        metrics = {
            "env/reward": reward,
            "env/speedup": eval_result.speedup,
            "env/speedup_norm": norm_speedup,
            "env/baseline_median": baseline_median,
            "env/correct": 1 if eval_result.correctness else 0,
        }
        logs = {
            "problem_id": self.problem_id,
            "compiled": str(eval_result.compiled),
            "raw_action": raw_action,
            "assembled_code": kernel_code,
        }

        # Stream telemetry for real-time visibility
        _write_telemetry({
            "problem_id": self.problem_id,
            "level": self.level,
            "compiled": eval_result.compiled,
            "correctness": eval_result.correctness,
            "speedup": eval_result.speedup,
            "speedup_norm": norm_speedup,
            "reward": reward,
            "baseline_median": baseline_median,
            "runtime_us": eval_result.runtime_us,
            "ref_runtime_us": eval_result.ref_runtime_us,
        })

        # Single-step episode; return same observation as final state
        next_ob, next_stop = await self.initial_observation()

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=next_ob,
            next_stop_condition=next_stop,
            metrics=metrics,
            logs=logs,
        )


@dataclass
class KernelEnvGroupBuilder(EnvGroupBuilder):
    problem_id: int
    renderer: Renderer
    level: int = 1
    group_size: int = 1

    async def make_envs(self) -> Sequence[Env]:
        return [
            KernelEnv(problem_id=self.problem_id, renderer=self.renderer, level=self.level)
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["kernelbench", f"level_{self.level}"]


@dataclass
class KernelRLDataset(RLDataset):
    problem_ids: list[int]
    renderer: Renderer
    level: int = 1
    batch_size: int = 1
    group_size: int = 1

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        batch_ids = self.problem_ids[start:end]
        return [
            KernelEnvGroupBuilder(
                problem_id=pid,
                renderer=self.renderer,
                level=self.level,
                group_size=self.group_size,
            )
            for pid in batch_ids
        ]

    def __len__(self) -> int:
        return len(self.problem_ids) // self.batch_size
