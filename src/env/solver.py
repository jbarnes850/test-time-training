from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

from src.env.evaluator import EvalResult, compute_reward, evaluate_kernel
from src.env.schema import KernelTask
from src.env.tasking import build_messages
from src.rlvr_utils import build_datums_from_group
from src.utils.code_utils import assemble_modelnew_code, extract_python_code
from src.utils.tinker_utils import ensure_tinker_cookbook_on_path


@dataclass(frozen=True)
class SolverBackendConfig:
    backend: str
    model_id: str
    sampler_path: str
    renderer_name: str = "gpt_oss_no_sysprompt"
    training_enabled: bool = False
    training_state_path: str = ""
    learning_rate: float = 1e-5
    lora_rank: int = 8


@dataclass
class SolveOutcome:
    prompt: Any
    sampled_tokens: list[list[int]]
    sampled_logprobs: list[list[float]]
    raw_actions: list[str]
    kernel_codes: list[str]
    eval_results: list[EvalResult]
    rewards: list[float]
    wall_clock_s: float


class SolverBackend(Protocol):
    backend_name: str
    provider_name: str
    model_id: str
    sampler_path: str

    def solve_task(
        self,
        task: KernelTask,
        *,
        k: int,
        temperature: float,
        max_tokens: int,
        level: int,
        eval_workers: int,
    ) -> SolveOutcome:
        ...

    def train_on_outcomes(
        self,
        outcomes: list[SolveOutcome],
        *,
        epoch: int,
    ) -> str:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


class DryRunSolverBackend:
    def __init__(self, sampler_path: str = "dry_run/sampler"):
        self.backend_name = "dry_run"
        self.provider_name = "local"
        self.model_id = "dry_run/model"
        self.sampler_path = sampler_path
        self.training_enabled = False

    def solve_task(
        self,
        task: KernelTask,
        *,
        k: int,
        temperature: float,
        max_tokens: int,
        level: int,
        eval_workers: int,
    ) -> SolveOutcome:
        start = time.time()
        raw_actions: list[str] = []
        kernel_codes: list[str] = []
        eval_results: list[EvalResult] = []
        rewards: list[float] = []
        for idx in range(k):
            raw_action = "return x"
            speedup = 1.0 + (((task.problem_id + idx) % 4) * 0.1)
            result = EvalResult(
                compiled=True,
                correctness=True,
                runtime_us=1.0,
                ref_runtime_us=speedup,
                speedup=speedup,
                metadata={},
            )
            reward = compute_reward(result.speedup, result.correctness)
            raw_actions.append(raw_action)
            kernel_codes.append(raw_action)
            eval_results.append(result)
            rewards.append(reward)
        return SolveOutcome(
            prompt=None,
            sampled_tokens=[],
            sampled_logprobs=[],
            raw_actions=raw_actions,
            kernel_codes=kernel_codes,
            eval_results=eval_results,
            rewards=rewards,
            wall_clock_s=time.time() - start,
        )

    def train_on_outcomes(self, outcomes: list[SolveOutcome], *, epoch: int) -> str:
        self.sampler_path = f"dry_run/sampler/epoch_{epoch}"
        return self.sampler_path

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "sampler_path": self.sampler_path,
            "training_enabled": self.training_enabled,
            "training_state_path": "",
            "renderer_name": "",
        }


class TinkerSolverBackend:
    def __init__(self, config: SolverBackendConfig):
        import tinker

        ensure_tinker_cookbook_on_path()
        from tinker_cookbook.renderers import get_renderer, get_text_content
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        if not config.sampler_path:
            raise ValueError("Solver sampler path must be provided for Tinker backend.")
        if config.training_enabled and not config.training_state_path:
            raise ValueError(
                "Training requires a Tinker training state path. "
                "Provide --training_state_path or checkpoint_jsonl with state_path."
            )

        self._tinker = tinker
        self._get_text_content = get_text_content
        self.backend_name = "tinker"
        self.provider_name = "tinker"
        self.model_id = config.model_id
        self.sampler_path = config.sampler_path
        self.renderer_name = config.renderer_name
        self.training_enabled = config.training_enabled
        self.training_state_path = config.training_state_path
        self.learning_rate = config.learning_rate
        self.lora_rank = config.lora_rank

        self._service_client = tinker.ServiceClient()
        self._sampling_client = self._service_client.create_sampling_client(model_path=self.sampler_path)
        tokenizer = get_tokenizer(self.model_id)
        self._renderer = get_renderer(self.renderer_name, tokenizer)
        self._training_client = None
        if self.training_enabled:
            self._training_client = self._service_client.create_training_client_from_state(
                self.training_state_path
            )

    def solve_task(
        self,
        task: KernelTask,
        *,
        k: int,
        temperature: float,
        max_tokens: int,
        level: int,
        eval_workers: int,
    ) -> SolveOutcome:
        start = time.time()
        messages = build_messages(task)
        prompt = self._renderer.build_generation_prompt(messages)
        future = self._sampling_client.sample(
            prompt=prompt,
            num_samples=k,
            sampling_params=self._tinker.SamplingParams(
                max_tokens=max_tokens,
                stop=self._renderer.get_stop_sequences(),
                temperature=temperature,
            ),
        )
        result = future.result()
        sampled_tokens = [list(seq.tokens) for seq in result.sequences]
        sampled_logprobs = [
            [float(lp) if lp is not None else 0.0 for lp in seq.logprobs]
            for seq in result.sequences
        ]
        raw_actions: list[str] = []
        kernel_codes: list[str] = []
        for seq in result.sequences:
            parsed_message, _ = self._renderer.parse_response(seq.tokens)
            raw_action = extract_python_code(self._get_text_content(parsed_message))
            kernel_code = assemble_modelnew_code(raw_action, task.reference_code)
            raw_actions.append(raw_action)
            kernel_codes.append(kernel_code)

        eval_results: list[EvalResult] = [None] * len(kernel_codes)  # type: ignore[assignment]
        with ThreadPoolExecutor(max_workers=max(1, eval_workers)) as executor:
            futures = {
                executor.submit(evaluate_kernel, task.problem_id, code, level): idx
                for idx, code in enumerate(kernel_codes)
            }
            for future in as_completed(futures):
                idx = futures[future]
                eval_results[idx] = future.result()

        rewards = [compute_reward(r.speedup, r.correctness) for r in eval_results]
        return SolveOutcome(
            prompt=prompt,
            sampled_tokens=sampled_tokens,
            sampled_logprobs=sampled_logprobs,
            raw_actions=raw_actions,
            kernel_codes=kernel_codes,
            eval_results=eval_results,
            rewards=rewards,
            wall_clock_s=time.time() - start,
        )

    def train_on_outcomes(
        self,
        outcomes: list[SolveOutcome],
        *,
        epoch: int,
    ) -> str:
        if not self.training_enabled or self._training_client is None:
            return self.sampler_path

        all_datums = []
        for outcome in outcomes:
            if outcome.prompt is None:
                continue
            if not outcome.sampled_tokens or not outcome.sampled_logprobs:
                continue
            mean_reward = statistics.mean(outcome.rewards) if outcome.rewards else 0.0
            advantages = [r - mean_reward for r in outcome.rewards]
            datums = build_datums_from_group(
                outcome.prompt,
                outcome.sampled_tokens,
                outcome.sampled_logprobs,
                advantages,
            )
            all_datums.extend(datums)

        if not all_datums:
            return self.sampler_path

        fwd_bwd = self._training_client.forward_backward(all_datums, loss_fn="importance_sampling")
        optim = self._training_client.optim_step(
            self._tinker.AdamParams(learning_rate=self.learning_rate)
        )
        _ = fwd_bwd.result()
        _ = optim.result()
        self.sampler_path = self._training_client.save_weights_for_sampler(
            name=f"adaptive_epoch_{epoch}"
        ).result().path
        self._sampling_client = self._service_client.create_sampling_client(model_path=self.sampler_path)
        return self.sampler_path

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "sampler_path": self.sampler_path,
            "training_enabled": self.training_enabled,
            "training_state_path": self.training_state_path,
            "renderer_name": self.renderer_name,
        }
