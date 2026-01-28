import argparse
import json
from pathlib import Path

from src.utils.path_utils import repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()

    root = repo_root()
    telemetry_path = root / "runs" / args.run / "telemetry.jsonl"
    if not telemetry_path.exists():
        print(f"Telemetry not found: {telemetry_path}")
        return 1

    records = []
    with telemetry_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    n = len(records)
    if n == 0:
        print("No telemetry records found")
        return 1

    correct = [r for r in records if r.get("correctness")]
    speedups = [r.get("speedup", 0.0) for r in correct]
    fast_1 = sum(1 for s in speedups if s > 1.0) / n

    print("Evaluation Summary")
    print(f"Run: {args.run}")
    print(f"Total samples: {n}")
    print(f"Correct samples: {len(correct)}")
    print(f"fast_1: {fast_1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
