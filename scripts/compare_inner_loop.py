import argparse
import json
from pathlib import Path

from src.metrics import fast_1


def _load(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner", type=str, default="runs/inner_loop/inner_loop_summary.json")
    parser.add_argument("--best_of_n", type=str, default="runs/best_of_n/best_of_n_summary.json")
    parser.add_argument("--out", type=str, default="runs/compare_summary.json")
    args = parser.parse_args()

    inner = _load(Path(args.inner))
    best = _load(Path(args.best_of_n))
    best_by_id = {row["problem_id"]: row for row in best}

    rows = []
    for row in inner:
        pid = row["problem_id"]
        best_row = best_by_id.get(pid, {})
        effective_fast_1 = row.get("effective_fast_1", row.get("adapted_fast_1", 0.0))
        rows.append({
            "problem_id": pid,
            "base_fast_1": row.get("base_fast_1", 0.0),
            "adapted_fast_1": row.get("adapted_fast_1", 0.0),
            "effective_fast_1": effective_fast_1,
            "best_of_n_fast_1": best_row.get("fast_1", 0.0),
            "rollback": row.get("rollback", False),
        })

    out_path = Path(args.out)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
