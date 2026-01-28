from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import chz

from src.tinker_env import KernelRLDataset
from src.utils.path_utils import repo_root
from src.utils.tinker_utils import ensure_tinker_cookbook_on_path

ensure_tinker_cookbook_on_path()

from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


@dataclass(frozen=True)
class SplitConfig:
    split_path: str = "splits/l1_seed42.json"
    shuffle: bool = True
    seed: int = 42


def _load_split(split_path: str) -> dict:
    root = repo_root()
    path = (root / split_path).resolve() if not Path(split_path).is_absolute() else Path(split_path)
    return json.loads(path.read_text())


def _load_problem_ids(split: dict, split_key: str, seed: int, shuffle: bool) -> list[int]:
    ids = list(split["problem_ids"][split_key])
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ids)
    return ids


@chz.chz
class KernelDatasetBuilder(RLDatasetBuilder):
    split_config: SplitConfig = SplitConfig()
    batch_size: int = 8
    group_size: int = 8
    level: int = 1
    model_name_for_tokenizer: str = "openai/gpt-oss-20b"
    renderer_name: str | None = None
    num_epochs: int = 1
    max_batches: int | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        split = _load_split(self.split_config.split_path)
        train_ids = _load_problem_ids(
            split, "train", self.split_config.seed, self.split_config.shuffle
        )
        train_ids = train_ids * self.num_epochs

        if self.max_batches is not None:
            max_ids = self.batch_size * self.max_batches
            train_ids = train_ids[:max_ids]

        renderer_name = self.renderer_name or get_recommended_renderer_name(self.model_name_for_tokenizer)
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(renderer_name, tokenizer)

        train_dataset = KernelRLDataset(
            problem_ids=train_ids,
            renderer=renderer,
            level=self.level,
            batch_size=self.batch_size,
            group_size=self.group_size,
        )

        return train_dataset, None
