from __future__ import annotations

import ast
import re
import textwrap


_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)

_QUOTE_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    for src, dst in _QUOTE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    match = _FENCE_RE.search(text)
    if match:
        code = match.group(1)
    else:
        code = text
    code = code.strip()
    code = re.sub(r"^```[a-zA-Z]*", "", code).strip()
    code = re.sub(r"```$", "", code).strip()
    return code


def _extract_forward_args(ref_code: str) -> str:
    try:
        tree = ast.parse(ref_code)
    except SyntaxError:
        return "self, *args, **kwargs"
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return ast.unparse(item.args)
    return "self, *args, **kwargs"


def _extract_forward_body(raw_action: str) -> str:
    raw_action = raw_action.strip()
    if not raw_action:
        return "pass"
    try:
        tree = ast.parse(raw_action)
    except SyntaxError:
        return raw_action
    # If the model emitted a full function or class, extract the forward body.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            lines = [ast.unparse(stmt) for stmt in node.body] or ["pass"]
            return "\n".join(lines)
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    lines = [ast.unparse(stmt) for stmt in item.body] or ["pass"]
                    return "\n".join(lines)
    return raw_action


def assemble_modelnew_code(raw_action: str, ref_code: str) -> str:
    body = _extract_forward_body(raw_action)
    args = _extract_forward_args(ref_code)

    imports = []
    if "import torch" not in ref_code:
        imports.append("import torch")
    if "import torch.nn as nn" not in ref_code and "import torch.nn" not in ref_code:
        imports.append("import torch.nn as nn")
    if "F." in body and "import torch.nn.functional as F" not in ref_code:
        imports.append("import torch.nn.functional as F")

    prefix = "\n".join(imports)
    if prefix:
        prefix += "\n\n"

    indented_body = textwrap.indent(body.rstrip(), " " * 8)
    if not indented_body.strip():
        indented_body = "        pass"

    modelnew = f"class ModelNew(Model):\n    def forward({args}):\n{indented_body}\n"
    return f"{prefix}{ref_code.strip()}\n\n{modelnew}"
