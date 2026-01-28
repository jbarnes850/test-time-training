from src.env.evaluator import compute_reward, compute_speedup


def test_incorrect_reward_zero():
    reward = compute_reward(speedup=10.0, correctness=False)
    assert reward == 0.0


def test_reward_monotonic_speedup():
    r1 = compute_reward(speedup=1.1, correctness=True)
    r2 = compute_reward(speedup=1.5, correctness=True)
    assert r2 > r1


def test_compute_speedup():
    assert compute_speedup(2.0, 4.0) == 2.0
    assert compute_speedup(0.0, 4.0) == 0.0
    assert compute_speedup(2.0, 0.0) == 0.0
