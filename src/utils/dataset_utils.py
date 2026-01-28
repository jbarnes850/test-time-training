from typing import Dict, Any

from datasets import load_dataset

DATASET_NAME = "ScalingIntelligence/KernelBench"


def load_kernelbench_level(level: int = 1):
    split = f"level_{level}"
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
