"""Tests for pass@k / reliability — the estimator must be provably correct,
since the whole study's reliability axis rests on it."""

import pytest

from da_verify.eval.metrics import TaskScore, aggregate, pass_at_k, reliability


def test_pass_at_k_extremes():
    assert pass_at_k(5, 0, 1) == 0.0           # never correct
    assert pass_at_k(5, 5, 1) == 1.0           # always correct
    assert pass_at_k(5, 1, 5) == 1.0           # n-c=4 < k=5 -> a correct one is guaranteed


def test_pass_at_k_known_value():
    # n=4, c=2, k=2: 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)
    assert pass_at_k(2, 1, 1) == pytest.approx(0.5)


def test_pass_at_k_monotonic_in_k():
    # more attempts can only help
    assert pass_at_k(10, 3, 1) <= pass_at_k(10, 3, 5)


def test_pass_at_k_requires_enough_samples():
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def test_reliability_is_strict():
    assert reliability(5, 5) == 1.0
    assert reliability(5, 4) == 0.0
    assert reliability(0, 0) == 0.0


def test_aggregate_separates_capability_and_reliability():
    # one task always right, one task right 1/4 of the time
    scores = [
        TaskScore(id=1, level="easy", n_samples=4, n_correct=4, n_format_ok=4, n_candidate=4),
        TaskScore(id=2, level="hard", n_samples=4, n_correct=1, n_format_ok=4, n_candidate=4),
    ]
    agg = aggregate(scores, k=4)
    # pass@1 = mean(1.0, 0.25) = 0.625
    assert agg["pass@1"] == pytest.approx(0.625)
    # reliability: only task 1 is all-correct -> 0.5
    assert agg["reliability(pass^k)"] == pytest.approx(0.5)
    # pass@4 >= pass@1 (capability >= single-shot)
    assert agg["pass@4"] >= agg["pass@1"]
