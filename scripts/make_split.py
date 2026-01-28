import argparse
import json
import random
from pathlib import Path

from src.utils.dataset_utils import load_kernelbench_level, dataset_metadata, DATASET_NAME
from src.utils.hash_utils import sha256_json
from src.utils.path_utils import repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    dataset = load_kernelbench_level(args.level)
    if "problem_id" not in dataset.column_names:
        raise ValueError("Dataset missing required column: problem_id")

    problem_ids = sorted(set(dataset["problem_id"]))
    rng = random.Random(args.seed)
    rng.shuffle(problem_ids)

    n_total = len(problem_ids)
    n_train = int(0.8 * n_total)
    train_ids = sorted(problem_ids[:n_train])
    eval_ids = sorted(problem_ids[n_train:])

    meta = dataset_metadata(dataset)
    split = {
        "dataset": {
            "name": DATASET_NAME,
            "config_name": meta["config_name"],
            "version": meta["version"],
            "fingerprint": meta["fingerprint"],
            "num_rows": meta["num_rows"],
        },
        "level": f"level_{args.level}",
        "seed": args.seed,
        "problem_ids": {
            "train": train_ids,
            "eval": eval_ids,
        },
    }
    split_hash = sha256_json(split)
    split["split_hash"] = split_hash

    root = repo_root()
    out_path = Path(args.out) if args.out else root / "splits" / f"l{args.level}_seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2))

    print(f"Loaded {meta['num_rows']} rows from {DATASET_NAME} {meta['config_name']}")
    print(f"Problem IDs: {n_total} (train={len(train_ids)}, eval={len(eval_ids)})")
    print(f"Split hash: {split_hash}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
