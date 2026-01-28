from src.guardrails import GuardrailConfig, should_rollback


def test_rollback_on_correctness_floor():
    cfg = GuardrailConfig(correctness_floor=0.5, latency_cap_multiplier=2.0)
    assert should_rollback(0.6, 1.0, 0.4, 1.0, cfg)


def test_rollback_on_correctness_drop():
    cfg = GuardrailConfig(correctness_floor=0.0, latency_cap_multiplier=2.0)
    assert should_rollback(0.6, 1.0, 0.5, 1.0, cfg)


def test_rollback_on_latency():
    cfg = GuardrailConfig(correctness_floor=0.0, latency_cap_multiplier=1.2)
    assert should_rollback(0.6, 1.0, 0.6, 1.3, cfg)


def test_no_rollback():
    cfg = GuardrailConfig(correctness_floor=0.0, latency_cap_multiplier=1.5)
    assert not should_rollback(0.6, 1.0, 0.7, 1.2, cfg)
