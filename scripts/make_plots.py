import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.path_utils import repo_root


def plot_reward_curve(metrics_path: Path, out_path: Path):
    df = pd.read_json(metrics_path, lines=True)
    reward_col = None
    for candidate in ["reward/total", "env/all/reward/total"]:
        if candidate in df.columns:
            reward_col = candidate
            break
    if reward_col is None:
        print("reward/total not found in metrics")
        return
    plt.figure(figsize=(6, 4))
    plt.plot(df[reward_col], marker="o")
    plt.title("RLVR Reward Curve")
    plt.xlabel("Step")
    plt.ylabel(reward_col)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_fast1_compare(compare_path: Path, out_path: Path):
    data = json.loads(compare_path.read_text())
    if not data:
        print("No comparison data")
        return
    pids = [row["problem_id"] for row in data]
    base = [row["base_fast_1"] for row in data]
    adapted = [row.get("effective_fast_1", row.get("adapted_fast_1", 0.0)) for row in data]
    bestn = [row["best_of_n_fast_1"] for row in data]

    x = range(len(pids))
    width = 0.25
    plt.figure(figsize=(6, 4))
    plt.bar([i - width for i in x], base, width=width, label="base")
    plt.bar(x, adapted, width=width, label="adapted")
    plt.bar([i + width for i in x], bestn, width=width, label="best-of-n")
    plt.xticks(list(x), [str(pid) for pid in pids])
    plt.xlabel("Problem ID")
    plt.ylabel("fast_1")
    plt.title("fast_1 Comparison")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="runs/rlvr_smoke/metrics.jsonl")
    parser.add_argument("--compare", type=str, default="runs/compare_summary.json")
    parser.add_argument("--out_dir", type=str, default="artifacts/plots")
    args = parser.parse_args()

    root = repo_root()
    metrics_path = (root / args.metrics).resolve()
    compare_path = (root / args.compare).resolve()
    out_dir = (root / args.out_dir).resolve()

    plot_reward_curve(metrics_path, out_dir / "reward_curve.png")
    if compare_path.exists():
        plot_fast1_compare(compare_path, out_dir / "fast1_compare.png")
    else:
        print("Comparison file not found; skipping fast1 plot")

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
