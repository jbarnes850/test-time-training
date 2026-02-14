from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.env.schema import TrainingArtifact, to_json_dict


def _artifact_from_dict(data: dict[str, Any]) -> TrainingArtifact:
    return TrainingArtifact(
        entry_id=str(data["entry_id"]),
        outcome_id=str(data["outcome_id"]),
        epoch=int(data["epoch"]),
        zone=str(data["zone"]),
        utility_score=float(data["utility_score"]),
        category_id=str(data["category_id"]),
        problem_id=data.get("problem_id"),
        level=data.get("level"),
        prompt_tokens=list(data["prompt_tokens"]),
        sampled_tokens=list(data["sampled_tokens"]),
        sampled_logprobs=[float(lp) for lp in data["sampled_logprobs"]],
        reward=float(data["reward"]),
        sampler_path=str(data["sampler_path"]),
    )


class TrainingReplayBuffer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts: list[TrainingArtifact] = []
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
                artifact = _artifact_from_dict(json.loads(line))
                if artifact.entry_id in self._entry_ids:
                    continue
                self._artifacts.append(artifact)
                self._entry_ids.add(artifact.entry_id)

    def __len__(self) -> int:
        return len(self._artifacts)

    def add_outcome(
        self,
        outcome_id: str,
        epoch: int,
        zone: str,
        utility_score: float,
        category_id: str,
        problem_id: int | None,
        level: int | None,
        prompt_tokens: list[int],
        sampled_tokens_list: list[list[int]],
        sampled_logprobs_list: list[list[float]],
        rewards: list[float],
        entry_ids: list[str],
        sampler_path: str,
    ) -> None:
        for entry_id, sampled_tokens, sampled_logprobs, reward in zip(
            entry_ids, sampled_tokens_list, sampled_logprobs_list, rewards
        ):
            if entry_id in self._entry_ids:
                continue
            artifact = TrainingArtifact(
                entry_id=entry_id,
                outcome_id=outcome_id,
                epoch=epoch,
                zone=zone,
                utility_score=utility_score,
                category_id=category_id,
                problem_id=problem_id,
                level=level,
                prompt_tokens=prompt_tokens,
                sampled_tokens=sampled_tokens,
                sampled_logprobs=sampled_logprobs,
                reward=reward,
                sampler_path=sampler_path,
            )
            self._artifacts.append(artifact)
            self._entry_ids.add(entry_id)
            with self.path.open("a") as handle:
                handle.write(json.dumps(to_json_dict(artifact)) + "\n")

    def sample_replay(
        self,
        current_epoch: int,
        recency_epochs: int = 3,
        zone_filter: str | None = "learning",
        limit: int | None = None,
    ) -> list[TrainingArtifact]:
        min_epoch = max(0, current_epoch - recency_epochs)
        filtered = [
            a
            for a in self._artifacts
            if min_epoch <= a.epoch < current_epoch
            and (zone_filter is None or a.zone == zone_filter)
        ]
        filtered.sort(key=lambda a: a.epoch, reverse=True)
        if limit is not None and limit > 0:
            filtered = filtered[:limit]
        return filtered

    def query_by_epoch(self, epoch: int) -> list[TrainingArtifact]:
        return [a for a in self._artifacts if a.epoch == epoch]
