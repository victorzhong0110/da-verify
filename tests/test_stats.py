"""Tests for the paired-comparison statistics — these decide whether the
headline 'verification helps' claim is real, so the math must be pinned."""

import pytest

from da_verify.eval.stats import compare_paired, mcnemar_exact_p, wilson_interval


def test_wilson_brackets_the_estimate():
    lo, hi = wilson_interval(8, 10)
    assert lo < 0.8 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_clamps_at_one():
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and lo > 0.0  # not the degenerate [1,1] the normal approx gives


def test_wilson_empty():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_mcnemar_no_discordant_is_ns():
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_all_one_direction_is_significant():
    # 10 fixes, 0 breaks -> strong evidence of a difference
    assert mcnemar_exact_p(0, 10) < 0.01


def test_mcnemar_symmetric_is_not_significant():
    # equal fixes and breaks -> no evidence
    assert mcnemar_exact_p(5, 5) > 0.5


def test_mcnemar_is_symmetric_in_args():
    assert mcnemar_exact_p(2, 8) == mcnemar_exact_p(8, 2)


def test_compare_paired_counts_and_direction():
    # A: task0,1 correct ; B: task1,2 correct
    a = [True, True, False, False]
    b = [False, True, True, False]
    cmp = compare_paired(a, b)
    assert cmp.n == 4
    assert cmp.both == 1            # task1
    assert cmp.a_only == 1          # task0: A right, B wrong (B broke it)
    assert cmp.b_only == 1          # task2: A wrong, B right (B fixed it)
    assert cmp.neither == 1         # task3
    assert cmp.net == 0             # 1 fixed - 1 broke
    assert cmp.acc_a == 0.5 and cmp.acc_b == 0.5


def test_compare_paired_requires_aligned_lengths():
    with pytest.raises(ValueError):
        compare_paired([True], [True, False])
