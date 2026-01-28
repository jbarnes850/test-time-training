import argparse
import json
from pathlib import Path

from src.utils.path_utils import repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="artifacts/manifest.json")
    args = parser.parse_args()

    root = repo_root()
    path = (root / args.path).resolve() if not Path(args.path).is_absolute() else Path(args.path)
    if not path.exists():
        print(f"Manifest not found: {path}")
        return 1
    manifest = json.loads(path.read_text())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
