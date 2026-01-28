from src.check_env import validate_env
from src.utils.hash_utils import sha256_json


def test_validate_env_missing(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    missing = validate_env()
    assert "TINKER_API_KEY" in missing


def test_validate_env_present(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    missing = validate_env()
    assert missing == []


def test_sha256_json_stable():
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert sha256_json(a) == sha256_json(b)
