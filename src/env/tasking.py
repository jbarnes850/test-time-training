from dataclasses import dataclass
from typing import Optional

from src.env.schema import KernelTask, Observation
from src.utils.dataset_utils import load_kernelbench_level


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str = (
        "You are an expert GPU kernel engineer.\n\n"
        "Reasoning effort: high.\n\n"
        "Think step-by-step internally, but do not reveal your reasoning.\n"
        "You must return only valid Python code.\n"
        "- No markdown, no backticks, no analysis, no explanations, no extra text.\n"
        "- Do not call tools or mention tools.\n"
        "- Output only the body of ModelNew.forward (no def line, no class, no imports).\n"
        "- The body must be functionally equivalent to Model.forward but faster.\n"
        "- If you cannot safely improve, output: pass\n"
        "- If you use symbols like torch, nn, or F, assume imports are provided."
        "\n\nFew-shot examples (body-only):\n"
        "Example 1 (Conv2d module):\n"
        "Output:\n"
        "return torch.nn.functional.conv2d(x, self.conv2d.weight, self.conv2d.bias, self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups)\n\n"
        "Example 2 (Linear module):\n"
        "Output:\n"
        "return torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)\n\n"
        "Example 3 (LayerNorm module):\n"
        "Output:\n"
        "return torch.nn.functional.layer_norm(x, self.ln.normalized_shape, self.ln.weight, self.ln.bias, self.ln.eps)\n"
    )
    system_preamble: str = (
        "You are an expert GPU kernel engineer. "
        "Given a PyTorch reference model, write an optimized kernel implementation."
    )
    instruction: str = (
        "Return only the body of ModelNew.forward (no def line, no class, no imports). "
        "Your output must be functionally equivalent to Model.forward but faster."
    )


def load_task(problem_id: int, level: int = 1) -> KernelTask:
    dataset = load_kernelbench_level(level)
    rows = [row for row in dataset if row["problem_id"] == problem_id]
    if not rows:
        raise ValueError(f"Problem id {problem_id} not found in level_{level}")
    row = rows[0]
    return KernelTask(problem_id=row["problem_id"], name=row["name"], reference_code=row["code"])


def render_user_prompt(task: KernelTask, cfg: Optional[PromptConfig] = None) -> str:
    cfg = cfg or PromptConfig()
    return (
        "Task: Optimize the kernel implementation.\n\n"
        f"Problem ID: {task.problem_id}\n"
        f"Task Name: {task.name}\n\n"
        "Reference Model:\n"
        '"""\n'
        f"{task.reference_code}\n"
        '"""\n\n'
        "Requirements:\n"
        "- Same input/output semantics as Model.forward\n"
        "- Return only the *body* of ModelNew.forward (no def line, no class, no imports)\n"
    )


def make_observation(task: KernelTask, cfg: Optional[PromptConfig] = None) -> Observation:
    cfg = cfg or PromptConfig()
    prompt = f"{cfg.system_preamble}\n\n{render_user_prompt(task, cfg=cfg)}"
    return Observation(problem_id=task.problem_id, prompt=prompt)


def build_messages(task: KernelTask, cfg: Optional[PromptConfig] = None) -> list[dict[str, str]]:
    cfg = cfg or PromptConfig()
    return [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": render_user_prompt(task, cfg=cfg)},
    ]
