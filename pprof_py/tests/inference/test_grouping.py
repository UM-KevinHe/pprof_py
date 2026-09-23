"""Size-based grouping for groupwise empirical-null calibration."""
import numpy as np
import pytest

from pprof_py.inference import assign_groups


def test_rank_rule_blocks_are_ceiling_sized():
    assert np.bincount(assign_groups(np.arange(10.0), 4, rule="rank"))[1:].tolist() == [3, 3, 3, 1]
    assert np.bincount(assign_groups(np.arange(8.0), 4, rule="rank"))[1:].tolist() == [2, 2, 2, 2]
    assert assign_groups(np.arange(8.0)[::-1], 4, rule="rank").tolist() == [4, 4, 3, 3, 2, 2, 1, 1]


def test_rank_rule_ties_follow_input_order_or_key():
    size = np.array([5, 5, 5, 5, 1, 9])                  # the cut after 3 providers falls inside the tie
    assert assign_groups(size, 2, rule="rank").tolist() == [1, 1, 2, 2, 1, 2]
    key = np.array([40, 30, 20, 10, 50, 60])            # tie-break by key reverses the tied providers
    assert assign_groups(size, 2, rule="rank", order=key).tolist() == [2, 2, 1, 1, 1, 2]
    assert assign_groups(size, 2, rule="rank", order=key.astype(str)).tolist() == [2, 2, 1, 1, 1, 2]


def test_rank_rule_missing_sizes_sort_last():
    size = np.array([np.nan, 3.0, 1.0, np.nan, 2.0, 4.0])
    assert assign_groups(size, 2, rule="rank").tolist() == [2, 1, 1, 2, 1, 2]
    assert assign_groups(size, 2, rule="rank", order=np.arange(6)).tolist() == [2, 1, 1, 2, 1, 2]


def test_quantile_rule_is_type7_breaks_with_ties_to_the_lower_group():
    rng = np.random.default_rng(0)
    for _ in range(20):
        size = np.round(rng.gamma(2.0, 30.0, int(rng.integers(20, 400))))
        breaks = np.quantile(size, [0.25, 0.5, 0.75], method="linear")
        if np.unique(breaks).size < 3:
            continue
        assert (assign_groups(size, 4) == np.searchsorted(breaks, size, side="left") + 1).all()   # quantile is the default


def test_quantile_rule_rejects_ambiguous_input():
    with pytest.raises(ValueError):
        assign_groups([1.0, np.nan, 3.0, 4.0], 2, rule="quantile")
    with pytest.raises(ValueError):
        assign_groups(np.r_[np.ones(20), 2.0], 4, rule="quantile")


def test_single_group_and_invalid_arguments():
    assert (assign_groups([3.0, 1.0, np.nan], 1) == 1).all()
    for bad in (dict(n_groups=0), dict(n_groups=2.5), dict(rule="kmeans")):
        with pytest.raises(ValueError):
            assign_groups(np.arange(5.0), **{"n_groups": 2, **bad})
    with pytest.raises(ValueError):
        assign_groups(np.arange(5.0), 2, rule="rank", order=np.arange(4))
