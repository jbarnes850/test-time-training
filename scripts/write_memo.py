import argparse
import json
from pathlib import Path

from src.utils.path_utils import repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="runs/rlvr_smoke/metrics.jsonl")
    parser.add_argument("--compare", type=str, default="runs/compare_summary.json")
    parser.add_argument("--out", type=str, default="docs/memo.md")
    args = parser.parse_args()

    root = repo_root()
    metrics_path = (root / args.metrics).resolve()
    compare_path = (root / args.compare).resolve()
    out_path = (root / args.out).resolve()

    reward_line = "N/A"
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        if lines:
            last = json.loads(lines[-1])
            reward_line = str(last.get("reward/total", last.get("env/all/reward/total", "N/A")))

    compare = []
    if compare_path.exists():
        compare = json.loads(compare_path.read_text())

    avg_base = "N/A"
    avg_adapted = "N/A"
    avg_best = "N/A"
    if compare:
        avg_base = sum(row.get("base_fast_1", 0.0) for row in compare) / len(compare)
        avg_adapted = sum(row.get("effective_fast_1", row.get("adapted_fast_1", 0.0)) for row in compare) / len(compare)
        avg_best = sum(row.get("best_of_n_fast_1", 0.0) for row in compare) / len(compare)

    memo = f"""# KernelBench RL Environment Memo\n\n## Summary\nWe built a KernelBench L1 environment and ran a GRPO/RLVR smoke run using Tinker with `openai/gpt-oss-20b`.\nWe also implemented a per-task test-time inner loop with LoRA updates and a best-of-N baseline for comparison.\n\n## Latest Metrics\n- RLVR reward/total (last step): {reward_line}\n- Comparison rows: {len(compare)}\n- Avg fast_1 (base / inner-loop / best-of-N): {avg_base} / {avg_adapted} / {avg_best}\n\n## Next Steps\n- Scale inner-loop evaluation to more tasks in L1 eval split.\n- Improve prompting and reward signal to increase correctness.\n- Run full comparison table and plot fast_1 uplift across the eval split.\n"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(memo)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
