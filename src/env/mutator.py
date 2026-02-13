from __future__ import annotations

import ast
import hashlib
import re
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Protocol

from src.env.replay_buffer import ReplayBuffer
from src.env.schema import KernelTask, MutatedTask
from src.env.teacher import category_id, infer_task_categories
from src.utils.code_utils import assemble_modelnew_code, extract_python_code
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


def _extract_model_forward_arg_names(code: str) -> list[str] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return [arg.arg for arg in item.args.args if arg.arg != "self"]
    return None


def _target_tags_from_category_id(category: str | None) -> set[str]:
    if not category or category == "unknown":
        return set()
    if category.startswith("composite:"):
        suffix = category.split(":", 1)[1]
        return {tag for tag in suffix.split("+") if tag}
    return {category}


def _matches_target_category(mutated_tags: set[str], target_category: str | None) -> bool:
    target_tags = _target_tags_from_category_id(target_category)
    if not target_tags:
        return True
    return target_tags.issubset(mutated_tags)


def _normalize_model_id(model_id: str) -> str:
    return model_id.split(":", 1)[0]


def _infer_renderer_name(
    model_id: str,
    *,
    explicit_renderer_name: str | None,
    get_recommended_renderer_name,
) -> str:
    if explicit_renderer_name:
        return explicit_renderer_name

    normalized = _normalize_model_id(model_id)
    lname = normalized.lower()
    if lname.startswith("moonshotai/kimi-k2.5"):
        # For constrained generation, disable long-form thinking prefill.
        return "kimi_k25_disable_thinking"

    try:
        return str(get_recommended_renderer_name(normalized))
    except Exception:
        pass

    if "/" in normalized:
        org, model_name = normalized.split("/", 1)
    else:
        org, model_name = "", normalized
    org = org.lower()
    model_name = model_name.lower()

    if org == "moonshotai" or "kimi" in model_name:
        if "k2.5" in model_name:
            return "kimi_k25_disable_thinking"
        return "kimi_k2"
    if org == "openai" or "gpt-oss" in model_name:
        return "gpt_oss_no_sysprompt"
    if org == "qwen" or "qwen" in model_name:
        return "qwen3"
    if org == "meta-llama" or "llama" in model_name:
        return "llama3"
    if org == "deepseek-ai" or "deepseek" in model_name:
        return "deepseekv3"
    return "role_colon"


ALLOWED_MUTATION_TYPES = {
    "architecture_swap",
    "logic_restructuring",
    "complexity_injection",
    "encoding_change",
    "structural_deepening",
    "fusion_rewrite",
}

MUTATION_SYSTEM_PROMPT = (
    "You are a kernel task mutator. Your role is to generate structurally novel benchmark tasks "
    "that preserve strict interface compatibility with the seed task. "
    "Do not optimize for speed. Optimize for valid structural diversity.\n\n"
    "Output format is mandatory:\n"
    "1) A fenced python code block containing the full reference code with class Model.\n"
    "2) A line: MUTATION_TYPE: <one_allowed_type>\n"
    "3) A line: OPTIMIZATION_PROMPT: <one sentence hint for solver optimization focus>\n\n"
    "Allowed MUTATION_TYPE values:\n"
    "- architecture_swap\n"
    "- logic_restructuring\n"
    "- complexity_injection\n"
    "- encoding_change\n"
    "- structural_deepening\n"
    "- fusion_rewrite\n\n"
    "Hard constraints:\n"
    "- Keep the same class name: Model\n"
    "- Keep Model.forward interface exactly identical (same arg names/order/defaults)\n"
    "- Preserve input/output semantics and shape contract\n"
    "- Change internal computational structure materially\n"
    "- No markdown explanation outside required metadata lines\n"
)

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_MUTATION_TYPE_RE = re.compile(r"^\s*MUTATION_TYPE:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_OPT_PROMPT_RE = re.compile(r"^\s*OPTIMIZATION_PROMPT:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def build_mutation_prompt(
    seed_task: KernelTask,
    target_category: str | None = None,
    *,
    target_speedup_band: tuple[float, float] | None = None,
    solver_trace_summary: str | None = None,
    mutation_instruction: str | None = None,
    decision_mode: str | None = None,
    reason_code: str | None = None,
) -> str:
    target_hint = (
        f"Target frontier category: {target_category}\n"
        if target_category and target_category != "unknown"
        else ""
    )
    speedup_hint = ""
    if target_speedup_band is not None:
        speedup_hint = (
            f"Target solver speedup band: [{target_speedup_band[0]:.2f}, {target_speedup_band[1]:.2f}]\n"
        )
    trace_hint = (
        f"Solver trace summary:\n{solver_trace_summary.strip()}\n\n"
        if solver_trace_summary and solver_trace_summary.strip()
        else ""
    )
    instruction_hint = (
        f"Teacher mutation instruction:\n{mutation_instruction.strip()}\n\n"
        if mutation_instruction and mutation_instruction.strip()
        else ""
    )
    mode_hint = f"Decision mode: {decision_mode}\n" if decision_mode else ""
    reason_hint = f"Reason code: {reason_code}\n" if reason_code else ""
    return (
        "Generate one mutated benchmark kernel task.\n"
        "Prioritize difficult but learnable task variation while preserving interface contract.\n\n"
        f"{target_hint}"
        f"{speedup_hint}"
        f"{mode_hint}"
        f"{reason_hint}"
        f"{instruction_hint}"
        f"{trace_hint}"
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


@dataclass(frozen=True)
class MutationProposal:
    raw_response: str
    code: str
    mutation_type: str
    optimization_prompt: str


@dataclass
class MutatorStats:
    attempts_total: int = 0
    accepted: int = 0
    api_failures: int = 0
    format_failures: int = 0
    parse_failures: int = 0
    compile_failures: int = 0
    interface_failures: int = 0
    structural_failures: int = 0
    novelty_failures: int = 0
    target_failures: int = 0
    semantic_failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "attempts_total": self.attempts_total,
            "accepted": self.accepted,
            "api_failures": self.api_failures,
            "format_failures": self.format_failures,
            "parse_failures": self.parse_failures,
            "compile_failures": self.compile_failures,
            "interface_failures": self.interface_failures,
            "structural_failures": self.structural_failures,
            "novelty_failures": self.novelty_failures,
            "target_failures": self.target_failures,
            "semantic_failures": self.semantic_failures,
        }


def parse_mutation_response(raw_response: str) -> MutationProposal | None:
    code_match = _CODE_FENCE_RE.search(raw_response or "")
    type_match = _MUTATION_TYPE_RE.search(raw_response or "")
    opt_match = _OPT_PROMPT_RE.search(raw_response or "")
    if code_match is None or type_match is None or opt_match is None:
        return None

    code = extract_python_code(code_match.group(0))
    mutation_type = type_match.group(1).strip().lower()
    optimization_prompt = opt_match.group(1).strip()
    if mutation_type not in ALLOWED_MUTATION_TYPES:
        return None
    if not code or not optimization_prompt:
        return None

    return MutationProposal(
        raw_response=raw_response,
        code=code,
        mutation_type=mutation_type,
        optimization_prompt=optimization_prompt,
    )


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
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        ...


class TinkerMutatorBackend:
    def __init__(
        self,
        model_id: str = "moonshotai/Kimi-K2.5",
        renderer_name: str | None = None,
        request_timeout_s: float = 180.0,
    ):
        import tinker

        ensure_tinker_cookbook_on_path()
        from tinker_cookbook.model_info import get_recommended_renderer_name
        from tinker_cookbook.renderers import get_renderer, get_text_content
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from transformers.models.auto.tokenization_auto import AutoTokenizer

        self._tinker = tinker
        self._get_text_content = get_text_content
        self._model_id = model_id
        self._request_timeout_s = max(1.0, float(request_timeout_s))
        self._renderer_name = _infer_renderer_name(
            model_id,
            explicit_renderer_name=renderer_name,
            get_recommended_renderer_name=get_recommended_renderer_name,
        )

        self._service_client = tinker.ServiceClient()
        if model_id.startswith("tinker://"):
            self._resolved_model_path = model_id
            self._sampling_client = self._service_client.create_sampling_client(
                model_path=model_id
            )
        else:
            self._resolved_model_path = f"base_model:{model_id}"
            self._sampling_client = self._service_client.create_sampling_client(
                base_model=model_id
            )
        normalized_model_id = _normalize_model_id(model_id)
        try:
            tokenizer = get_tokenizer(normalized_model_id)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    normalized_model_id,
                    use_fast=True,
                    trust_remote_code=True,
                )
            except TypeError:
                tokenizer = AutoTokenizer.from_pretrained(
                    normalized_model_id,
                    trust_remote_code=True,
                )
        self._renderer = get_renderer(self._renderer_name, tokenizer)

    @property
    def backend_name(self) -> str:
        return "tinker"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def resolved_model_path(self) -> str:
        return self._resolved_model_path

    def generate_mutation(
        self,
        seed_task: KernelTask,
        prompt: str,
        *,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
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
        try:
            result = future.result(timeout=self._request_timeout_s)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"Tinker sampling timed out after {self._request_timeout_s:.1f}s"
            ) from exc
        if not result.sequences:
            return ""
        parsed_message, _ = self._renderer.parse_response(result.sequences[0].tokens)
        return self._get_text_content(parsed_message)


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
        system_prompt: str,
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
        semantic_filter: str = "off",
        semantic_correct_trials: int = 1,
        semantic_perf_trials: int = 1,
    ):
        self.backend = backend
        self.replay_buffer = replay_buffer
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        if semantic_filter not in {"off", "fast"}:
            raise ValueError("semantic_filter must be one of {'off', 'fast'}.")
        self.semantic_filter = semantic_filter
        self.semantic_correct_trials = max(1, int(semantic_correct_trials))
        self.semantic_perf_trials = max(1, int(semantic_perf_trials))
        self._stats = MutatorStats()

    @property
    def stats(self) -> MutatorStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = MutatorStats()

    def mutate(
        self,
        seed_task: KernelTask,
        *,
        epoch: int,
        seed_problem_id: int | None = None,
        target_category: str | None = None,
        level: int = 1,
        target_speedup_band: tuple[float, float] | None = None,
        solver_trace_summary: str | None = None,
        mutation_instruction: str | None = None,
        decision_mode: str | None = None,
        reason_code: str | None = None,
    ) -> MutatedTask | None:
        prompt = build_mutation_prompt(
            seed_task,
            target_category=target_category,
            target_speedup_band=target_speedup_band,
            solver_trace_summary=solver_trace_summary,
            mutation_instruction=mutation_instruction,
            decision_mode=decision_mode,
            reason_code=reason_code,
        )
        prompt_hash = _sha256_text(prompt)
        for _ in range(self.max_retries):
            self._stats.attempts_total += 1
            try:
                raw_response = self.backend.generate_mutation(
                    seed_task,
                    prompt,
                    system_prompt=MUTATION_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception:
                self._stats.api_failures += 1
                continue

            proposal = parse_mutation_response(raw_response)
            if proposal is None:
                self._stats.format_failures += 1
                continue

            validation = self.validate_mutation(seed_task, proposal.code)
            if not validation.valid:
                self._record_validation_failure(validation.stage)
                continue

            tags = tuple(sorted(infer_task_categories(proposal.code)))
            if not _matches_target_category(set(tags), target_category):
                self._record_validation_failure("target")
                continue

            if self.semantic_filter != "off":
                semantic_validation = self.semantic_check(seed_task, proposal.code, level=level)
                if not semantic_validation.valid:
                    self._record_validation_failure(semantic_validation.stage)
                    continue

            cat_id = category_id(set(tags))
            parent_id = f"seed_{seed_problem_id if seed_problem_id is not None else seed_task.problem_id}"
            task_id = f"mut_{seed_task.problem_id}_{epoch}_{validation.novelty_hash[:12]}"
            self._stats.accepted += 1
            return MutatedTask(
                task_id=task_id,
                parent_task_id=parent_id,
                seed_problem_id=seed_problem_id if seed_problem_id is not None else seed_task.problem_id,
                name=f"mutated_{seed_task.name}_{validation.novelty_hash[:8]}",
                reference_code=proposal.code,
                interface_signature_hash=validation.interface_signature_hash,
                category_tags=tags,
                category_id=cat_id,
                mutation_backend=self.backend.backend_name,
                mutation_model_id=self.backend.model_id,
                mutation_prompt_hash=prompt_hash,
                novelty_hash=validation.novelty_hash,
                epoch_created=epoch,
                mutation_type=proposal.mutation_type,
                optimization_prompt=proposal.optimization_prompt,
                teacher_decision_mode=decision_mode or "",
                teacher_reason_code=reason_code or "",
                teacher_target_speedup_band=target_speedup_band or (0.0, 0.0),
                teacher_mutation_instruction=mutation_instruction or "",
                solver_trace_summary=solver_trace_summary or "",
            )
        return None

    def _record_validation_failure(self, stage: str) -> None:
        if stage == "parse":
            self._stats.parse_failures += 1
        elif stage == "compile":
            self._stats.compile_failures += 1
        elif stage == "interface":
            self._stats.interface_failures += 1
        elif stage == "structural":
            self._stats.structural_failures += 1
        elif stage == "novelty":
            self._stats.novelty_failures += 1
        elif stage == "target":
            self._stats.target_failures += 1
        elif stage == "semantic":
            self._stats.semantic_failures += 1

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

        try:
            seed_tree = ast.parse(seed_task.reference_code)
            mut_tree = ast.parse(mutated_code)
        except SyntaxError as exc:
            return MutationValidationResult(valid=False, stage="parse", message=str(exc))
        if ast.dump(seed_tree, include_attributes=False) == ast.dump(mut_tree, include_attributes=False):
            return MutationValidationResult(
                valid=False,
                stage="structural",
                message="Mutation is structurally identical to seed.",
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

    def semantic_check(
        self,
        seed_task: KernelTask,
        mutated_code: str,
        *,
        level: int,
    ) -> MutationValidationResult:
        arg_names = _extract_model_forward_arg_names(mutated_code)
        if arg_names is None:
            return MutationValidationResult(
                valid=False,
                stage="semantic",
                message="Could not parse Model.forward args for semantic probe.",
            )

        super_args = ", ".join(arg_names)
        action = f"return super().forward({super_args})" if super_args else "return super().forward()"
        probe_kernel_code = assemble_modelnew_code(action, mutated_code)

        from src.env.evaluator import EvalConfig, evaluate_kernel

        result = evaluate_kernel(
            problem_id=seed_task.problem_id,
            kernel_code=probe_kernel_code,
            level=level,
            config=EvalConfig(
                num_correct_trials=self.semantic_correct_trials,
                num_perf_trials=self.semantic_perf_trials,
                measure_performance=False,
            ),
        )
        if not result.compiled:
            error_text = ""
            if isinstance(result.metadata, dict):
                error_text = str(
                    result.metadata.get("error_message")
                    or result.metadata.get("error")
                    or result.metadata.get("error_trace")
                    or ""
                ).strip()
            detail = f": {error_text}" if error_text else ""
            return MutationValidationResult(
                valid=False,
                stage="semantic",
                message=f"Semantic probe compile failed{detail}",
            )
        if not result.correctness:
            error_text = ""
            if isinstance(result.metadata, dict):
                error_text = str(
                    result.metadata.get("error_message")
                    or result.metadata.get("error")
                    or result.metadata.get("error_trace")
                    or ""
                ).strip()
            detail = f": {error_text}" if error_text else ""
            return MutationValidationResult(
                valid=False,
                stage="semantic",
                message=f"Semantic probe failed correctness check vs original task{detail}",
            )
        return MutationValidationResult(valid=True, stage="ok")
