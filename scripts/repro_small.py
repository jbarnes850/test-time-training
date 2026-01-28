import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    py = sys.executable
    run([py, "-m", "scripts.l1_smoke"])
    run([py, "-m", "scripts.make_split", "--level", "1", "--seed", "42"])
    run([py, "-m", "scripts.write_manifest", "--split", "splits/l1_seed42.json"])
    run([py, "-m", "scripts.kb_smoke"])
    run([py, "-m", "scripts.train_smoke", "--max_batches", "2"])
    run([py, "-m", "scripts.best_of_n", "--max_tasks", "1", "--k", "16"])
    run([py, "-m", "scripts.inner_loop_smoke", "--max_tasks", "1", "--k", "16", "--steps", "1"])
    run([py, "-m", "scripts.compare_inner_loop"])
    run([py, "-m", "scripts.make_plots"])
    run([py, "-m", "scripts.write_memo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
