from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailConfig:
    correctness_floor: float = 0.0
    latency_cap_multiplier: float = 1.5


def should_rollback(base_correctness: float, base_latency: float, adapted_correctness: float, adapted_latency: float, cfg: GuardrailConfig) -> bool:
    if adapted_correctness < cfg.correctness_floor:
        return True
    if adapted_correctness < base_correctness:
        return True
    if base_latency > 0 and adapted_latency > base_latency * cfg.latency_cap_multiplier:
        return True
    return False
