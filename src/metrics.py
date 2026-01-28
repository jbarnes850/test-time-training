from __future__ import annotations

from typing import Iterable


def fast_1(correctness: Iterable[bool], speedups: Iterable[float]) -> float:
    correct_list = list(correctness)
    speed_list = list(speedups)
    n = len(speed_list)
    if n == 0:
        return 0.0
    count = 0
    for is_correct, sp in zip(correct_list, speed_list):
        if is_correct and sp > 1.0:
            count += 1
    return count / n
