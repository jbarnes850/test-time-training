from functools import lru_cache
from typing import Any, Dict

from datasets import load_dataset, load_dataset_builder

DATASET_NAME = "ScalingIntelligence/KernelBench"


@lru_cache(maxsize=1)
def available_kernelbench_splits() -> tuple[str, ...]:
    builder = load_dataset_builder(DATASET_NAME)
    return tuple(sorted(builder.info.splits.keys()))


@lru_cache(maxsize=1)
def available_kernelbench_levels() -> tuple[int, ...]:
    levels: list[int] = []
    for split_name in available_kernelbench_splits():
        if not split_name.startswith("level_"):
            continue
        suffix = split_name.split("_", 1)[1]
        if suffix.isdigit():
            levels.append(int(suffix))
    return tuple(sorted(levels))


def load_kernelbench_level(level: int = 1):
    split = f"level_{level}"
    if split not in available_kernelbench_splits():
        raise ValueError(
            f"Unknown KernelBench split '{split}'. "
            f"Available splits: {list(available_kernelbench_splits())}"
        )
    return load_dataset(DATASET_NAME, split=split)


def dataset_metadata(dataset) -> Dict[str, Any]:
    info = dataset.info
    return {
        "dataset_name": info.builder_name,
        "config_name": info.config_name,
        "version": str(info.version),
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "num_rows": dataset.num_rows,
    }
