from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Protocol

from src.env.replay_buffer import ReplayBuffer
from src.env.schema import KernelTask, MutatedTask
from src.env.teacher import category_id, infer_task_categories
from src.utils.code_utils import extract_python_code
from src.utils.tinker_utils import ensure_tinker_cookbook_on_path


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_model_forward_signature(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return ast.unparse(item.args)
    return None


def build_mutation_prompt(seed_task: KernelTask) -> str:
    return (
        "You are generating a mutated benchmark kernel task.\n"
        "Return only Python code for a complete reference implementation.\n"
        "Requirements:\n"
        "1) Keep the same class name `Model` and same `Model.forward` signature.\n"
        "2) Preserve exact input/output interface and tensor shape semantics.\n"
        "3) Change internal structure and computation pathway meaningfully.\n"
        "4) Do not include markdown, commentary, or backticks.\n\n"
        f"Seed task id: {seed_task.problem_id}\n"
        f"Seed task name: {seed_task.name}\n\n"
        "Seed reference code:\n"
        '"""\n'
        f"{seed_task.reference_code}\n"
        '"""\n'
    )


@dataclass(frozen=True)
class MutationValidationResult:
    valid: bool
    stage: str
    message: str = ""
    interface_signature_hash: str = ""
    novelty_hash: str = ""


class MutatorBackend(Protocol):
    @property
    def backend_name(self) -> str:
        ...

    @property
    def model_id(self) -> str:
        ...

    def generate_mutation(
        self,
        seed_task: KernelTask,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        ...


class TinkerMutatorBackend:
    def __init__(
        self,
        model_id: str = "moonshotai/Kimi-K2.5",
        renderer_name: str = "gpt_oss_no_sysprompt",
    ):
        import tinker

        ensure_tinker_cookbook_on_path()
        from tinker_cookbook.renderers import get_renderer, get_text_content
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self._tinker = tinker
        self._get_text_content = get_text_content
        self._model_id = model_id
        self._renderer_name = renderer_name

        self._service_client = tinker.ServiceClient()
        self._sampling_client = self._service_client.create_sampling_client(model_path=model_id)
        tokenizer = get_tokenizer(model_id)
        self._renderer = get_renderer(renderer_name, tokenizer)

    @property
    def backend_name(self) -> str:
        return "tinker"

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate_mutation(
        self,
        seed_task: KernelTask,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an expert kernel mutator that writes valid Python reference models.",
            },
            {"role": "user", "content": prompt},
        ]
        model_input = self._renderer.build_generation_prompt(messages)
        future = self._sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=self._tinker.SamplingParams(
                max_tokens=max_tokens,
                stop=self._renderer.get_stop_sequences(),
                temperature=temperature,
            ),
        )
        result = future.result()
        if not result.sequences:
            return ""
        parsed_message, _ = self._renderer.parse_response(result.sequences[0].tokens)
        text = self._get_text_content(parsed_message)
        return extract_python_code(text)


class ApiMutatorBackend:
    def __init__(self, model_id: str):
        self._model_id = model_id

    @property
    def backend_name(self) -> str:
        return "api_stub"

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate_mutation(
        self,
        seed_task: KernelTask,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError("ApiMutatorBackend is a stub in Phase 1.")


class KernelMutator:
    def __init__(
        self,
        backend: MutatorBackend,
        replay_buffer: ReplayBuffer,
        *,
        max_retries: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        self.backend = backend
        self.replay_buffer = replay_buffer
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

    def mutate(
        self,
        seed_task: KernelTask,
        *,
        epoch: int,
        seed_problem_id: int | None = None,
    ) -> MutatedTask | None:
        prompt = build_mutation_prompt(seed_task)
        prompt_hash = _sha256_text(prompt)
        for _ in range(self.max_retries):
            mutated_code = self.backend.generate_mutation(
                seed_task,
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            validation = self.validate_mutation(seed_task, mutated_code)
            if not validation.valid:
                continue

            tags = tuple(sorted(infer_task_categories(mutated_code)))
            cat_id = category_id(set(tags))
            parent_id = f"seed_{seed_problem_id if seed_problem_id is not None else seed_task.problem_id}"
            task_id = f"mut_{seed_task.problem_id}_{epoch}_{validation.novelty_hash[:12]}"
            return MutatedTask(
                task_id=task_id,
                parent_task_id=parent_id,
                seed_problem_id=seed_problem_id if seed_problem_id is not None else seed_task.problem_id,
                name=f"mutated_{seed_task.name}_{validation.novelty_hash[:8]}",
                reference_code=mutated_code,
                interface_signature_hash=validation.interface_signature_hash,
                category_tags=tags,
                category_id=cat_id,
                mutation_backend=self.backend.backend_name,
                mutation_model_id=self.backend.model_id,
                mutation_prompt_hash=prompt_hash,
                novelty_hash=validation.novelty_hash,
                epoch_created=epoch,
            )
        return None

    def validate_mutation(
        self,
        seed_task: KernelTask,
        mutated_code: str,
    ) -> MutationValidationResult:
        try:
            ast.parse(mutated_code)
        except SyntaxError as exc:
            return MutationValidationResult(valid=False, stage="parse", message=str(exc))

        try:
            compile(mutated_code, "<mutated_task>", "exec")
        except SyntaxError as exc:
            return MutationValidationResult(valid=False, stage="compile", message=str(exc))

        seed_sig = _extract_model_forward_signature(seed_task.reference_code)
        mut_sig = _extract_model_forward_signature(mutated_code)
        if seed_sig is None or mut_sig is None:
            return MutationValidationResult(
                valid=False,
                stage="interface",
                message="Could not locate Model.forward signature.",
            )
        if seed_sig != mut_sig:
            return MutationValidationResult(
                valid=False,
                stage="interface",
                message="Model.forward signature mismatch.",
            )

        novelty_hash = _sha256_text(mutated_code)
        known_hashes = {
            _sha256_text(entry.task_reference_code)
            for entry in self.replay_buffer.entries()
        }
        if novelty_hash in known_hashes:
            return MutationValidationResult(
                valid=False,
                stage="novelty",
                message="Mutation already exists in replay buffer.",
                interface_signature_hash=_sha256_text(mut_sig),
                novelty_hash=novelty_hash,
            )

        return MutationValidationResult(
            valid=True,
            stage="ok",
            interface_signature_hash=_sha256_text(mut_sig),
            novelty_hash=novelty_hash,
        )
