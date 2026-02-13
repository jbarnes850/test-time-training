from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.env.schema import EvalResult, ReplayEntry, to_json_dict


def _entry_from_dict(data: dict[str, Any]) -> ReplayEntry:
    eval_data = data["eval_result"]
    eval_result = EvalResult(
        compiled=bool(eval_data["compiled"]),
        correctness=bool(eval_data["correctness"]),
        runtime_us=float(eval_data["runtime_us"]),
        ref_runtime_us=float(eval_data["ref_runtime_us"]),
        speedup=float(eval_data["speedup"]),
        metadata=dict(eval_data.get("metadata", {})),
    )
    return ReplayEntry(
        entry_id=str(data["entry_id"]),
        task_id=str(data["task_id"]),
        parent_task_id=data.get("parent_task_id"),
        problem_id=data.get("problem_id"),
        level=data.get("level"),
        category_id=str(data["category_id"]),
        task_reference_code=str(data["task_reference_code"]),
        kernel_code=str(data["kernel_code"]),
        eval_result=eval_result,
        reward=float(data["reward"]),
        sampler_path=str(data["sampler_path"]),
        backend=str(data["backend"]),
        timestamp=float(data["timestamp"]),
        epoch=int(data["epoch"]),
        is_mutated=bool(data.get("is_mutated", False)),
    )


class ReplayBuffer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[ReplayEntry] = []
        self._entry_ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = _entry_from_dict(json.loads(line))
                if entry.entry_id in self._entry_ids:
                    continue
                self._entries.append(entry)
                self._entry_ids.add(entry.entry_id)

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[ReplayEntry]:
        return list(self._entries)

    def append(self, entry: ReplayEntry) -> None:
        if entry.entry_id in self._entry_ids:
            return
        self._entries.append(entry)
        self._entry_ids.add(entry.entry_id)
        with self.path.open("a") as handle:
            handle.write(json.dumps(to_json_dict(entry)) + "\n")

    def query(
        self,
        category_id: str | None = None,
        min_speedup: float | None = None,
        correct_only: bool | None = None,
        recency_window: int | None = None,
        limit: int | None = None,
    ) -> list[ReplayEntry]:
        rows = self._entries
        if recency_window is not None and recency_window > 0:
            rows = rows[-recency_window:]

        filtered: list[ReplayEntry] = []
        for entry in rows:
            if category_id is not None and entry.category_id != category_id:
                continue
            if min_speedup is not None and entry.eval_result.speedup < min_speedup:
                continue
            if correct_only is True and not entry.eval_result.correctness:
                continue
            if correct_only is False and entry.eval_result.correctness:
                continue
            filtered.append(entry)

        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        if limit is not None and limit > 0:
            filtered = filtered[:limit]
        return filtered

    def select_seed(
        self,
        teacher_signal: dict[str, Any] | None = None,
        exclude_banked: bool = True,
    ) -> ReplayEntry | None:
        signal = teacher_signal or {}
        candidates = self.query(
            category_id=signal.get("category_id"),
            min_speedup=signal.get("min_speedup"),
            correct_only=signal.get("correct_only", True),
            recency_window=signal.get("recency_window"),
            limit=signal.get("limit"),
        )
        if not candidates:
            return None

        for entry in candidates:
            if exclude_banked and bool(entry.eval_result.metadata.get("banked", False)):
                continue
            return entry
        return None
