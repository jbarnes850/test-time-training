import argparse
import json
from pathlib import Path

from src.utils.dataset_utils import load_kernelbench_level, dataset_metadata, DATASET_NAME
from src.utils.git_utils import get_git_commit
from src.utils.hash_utils import sha256_json
from src.utils.path_utils import repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="splits/l1_seed42.json")
    parser.add_argument("--out", type=str, default="artifacts/manifest.json")
    args = parser.parse_args()

    root = repo_root()
    split_path = (root / args.split).resolve() if not Path(args.split).is_absolute() else Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split = json.loads(split_path.read_text())

    expected_hash = split.get("split_hash")
    split_copy = dict(split)
    split_copy.pop("split_hash", None)
    computed_hash = sha256_json(split_copy)
    if expected_hash and expected_hash != computed_hash:
        raise ValueError("Split hash mismatch: split file may have been modified")

    level = split.get("level", "level_1")
    level_num = int(level.split("_")[-1])
    dataset = load_kernelbench_level(level_num)
    meta = dataset_metadata(dataset)

    split_dataset = split.get("dataset", {})
    dataset_match = all(
        split_dataset.get(k) == meta.get(k)
        for k in ["config_name", "version", "fingerprint", "num_rows"]
    )
    if not dataset_match:
        print("Warning: dataset metadata does not match split metadata")

    manifest = {
        "dataset": {
            "name": DATASET_NAME,
            "config_name": meta["config_name"],
            "version": meta["version"],
            "fingerprint": meta["fingerprint"],
            "num_rows": meta["num_rows"],
        },
        "split": {
            "path": str(split_path),
            "hash_expected": expected_hash,
            "hash_computed": computed_hash,
            "seed": split.get("seed"),
        },
        "code": {
            "git_commit": get_git_commit(),
        },
        "checks": {
            "dataset_metadata_match": dataset_match,
        },
    }
    manifest["manifest_hash"] = sha256_json(manifest)

    out_path = Path(args.out)
    out_path = (root / out_path).resolve() if not out_path.is_absolute() else out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
