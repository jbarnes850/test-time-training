import sys
from pathlib import Path

from src.utils.path_utils import repo_root


def ensure_tinker_cookbook_on_path() -> Path:
    root = repo_root()
    path = root / "vendor" / "tinker-cookbook"
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
