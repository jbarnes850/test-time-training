import json
import subprocess
from pathlib import Path

from src.utils.path_utils import repo_root


def main() -> int:
    root = repo_root()
    config_path = root / "configs" / "settings.json"
    if not config_path.exists():
        print("Missing configs/settings.json")
        return 1
    cfg = json.loads(config_path.read_text())
    kb_root_value = cfg.get("kernelbench_root")
    if not kb_root_value:
        print("kernelbench_root not configured in configs/settings.json")
        return 1
    kb_root = (root / kb_root_value).resolve()
    if not kb_root.exists():
        print(f"KernelBench root not found: {kb_root}")
        return 1
    commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", str(kb_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except Exception:
        pass
    print(f"KernelBench root: {kb_root}")
    print(f"KernelBench commit: {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
