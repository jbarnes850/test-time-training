import os
import sys
from pathlib import Path

from src.utils.path_utils import repo_root

REQUIRED_VARS = ["TINKER_API_KEY"]
OPTIONAL_VARS = ["HF_TOKEN", "WANDB_API_KEY"]


def _load_dotenv_if_present() -> None:
    env_path = repo_root() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def validate_env(required=None):
    required = REQUIRED_VARS if required is None else required
    missing = [k for k in required if not os.environ.get(k)]
    return missing


def main() -> int:
    _load_dotenv_if_present()
    missing = validate_env()
    if missing:
        print("Missing required environment variables:")
        for key in missing:
            print(f"- {key}")
        return 1
    print("Environment OK.")
    for key in OPTIONAL_VARS:
        if os.environ.get(key):
            print(f"Optional set: {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
