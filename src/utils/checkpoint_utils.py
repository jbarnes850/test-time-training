from __future__ import annotations

import json
from pathlib import Path

from src.utils.path_utils import repo_root


def load_latest_checkpoint(checkpoints_jsonl: str) -> dict | None:
    path = Path(checkpoints_jsonl)
    if not path.is_absolute():
        path = (repo_root() / checkpoints_jsonl).resolve()
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and (payload.get("sampler_path") or payload.get("state_path")):
            return payload
    return None
