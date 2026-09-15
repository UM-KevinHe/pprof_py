"""Engine self-consistency: the efficient sweep-based BreslowTies engine
must agree with a deliberately naive, independently-written brute-force
implementation (see brute_force_reference.py) at random beta values --
not just at the fitted optimum -- across every combination of left
truncation, strata, weights, and offsets.

This is deliberately independent of whether R is installed: it validates
that algorithms/risk_sets.py + algorithms/ties.py correctly implement
*the formulas this package defines*, which is a precondition for (but not
a substitute for) matching R -- that correspondence is what
test_r_comparison.py checks separately.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from pprof_py.algorithms.survival.cox_likelihood import cox_partial_likelihood

sys.path.insert(0, os.path.dirname(__file__))
from brute_force_reference import brute_force_partial_likelihood  # noqa: E402


def _make_dataset(n, p, seed, left_trunc=False, n_strata=1, use_weights=False, use_offset=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta_true = rng.normal(size=p) * 0.5
    offset = rng.normal(size=n) * 0.3 if use_offset else np.zeros(n)
    risk = np.exp(X @ beta_true + offset)
    U = rng.uniform(size=n)
    T_duration = -np.log(U) / risk
    C_duration = rng.exponential(scale=np.median(T_duration) * 1.5, size=n)
    obs_duration = np.maximum(np.round(np.minimum(T_duration, C_duration), 1), 0.1)
    event = (T_duration <= C_duration).astype(float)

    if left_trunc:
        start = np.round(rng.uniform(0, T_duration.mean() * 0.5, size=n), 1)
    else:
        start = np.zeros(n)
    stop = start + obs_duration  # exact by construction: stop > start always

    strata = rng.integers(0, n_strata, size=n)
    weight = rng.uniform(0.5, 2.0, size=n) if use_weights else np.ones(n)
    return X, start, stop, event, offset, weight, strata


SCENARIOS = [
    dict(n=40, p=2, seed=1, left_trunc=False, n_strata=1, use_weights=False, use_offset=False),
    dict(n=40, p=2, seed=2, left_trunc=True, n_strata=1, use_weights=False, use_offset=False),
    dict(n=60, p=3, seed=3, left_trunc=False, n_strata=3, use_weights=False, use_offset=False),
    dict(n=60, p=2, seed=4, left_trunc=False, n_strata=1, use_weights=True, use_offset=False),
    dict(n=60, p=2, seed=5, left_trunc=False, n_strata=1, use_weights=False, use_offset=True),
    dict(n=80, p=3, seed=6, left_trunc=True, n_strata=4, use_weights=True, use_offset=True),
    dict(n=25, p=1, seed=7, left_trunc=True, n_strata=2, use_weights=True, use_offset=True),
    dict(n=15, p=1, seed=8, left_trunc=False, n_strata=1, use_weights=False, use_offset=False),
]


@pytest.mark.parametrize("cfg", SCENARIOS, ids=[f"n{c['n']}_trunc{c['left_trunc']}_strata{c['n_strata']}_w{c['use_weights']}_off{c['use_offset']}" for c in SCENARIOS])
def test_matches_brute_force_at_random_beta(cfg):
    X, start, stop, event, offset, weight, strata = _make_dataset(**cfg)
    p = X.shape[1]
    rng = np.random.default_rng(999)

    for _ in range(3):
        beta = rng.normal(size=p) * 0.7
        ll1, sc1, inf1 = cox_partial_likelihood(
            X, start, stop, event, beta, offset=offset, weight=weight, strata=strata, ties="breslow"
        )
        ll2, sc2, inf2 = brute_force_partial_likelihood(X, start, stop, event, beta, offset, weight, strata)

        assert ll1 == pytest.approx(ll2, rel=1e-8, abs=1e-8)
        np.testing.assert_allclose(sc1, sc2, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(inf1, inf2, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("cfg", SCENARIOS, ids=[f"n{c['n']}_trunc{c['left_trunc']}_strata{c['n_strata']}_w{c['use_weights']}_off{c['use_offset']}" for c in SCENARIOS])
def test_martingale_residuals_numba_matches_python_fallback(cfg):
    """Same cross-check as the two above, for the martingale-residuals
    kernel (algorithms/statistics/residuals.py) -- a literal C port is
    exactly the kind of code where a compiled and an interpreted version
    could subtly diverge, so this is checked directly rather than
    assumed from the R-comparison tests passing alone."""
    from pprof_py.inference.survival.residuals import (
        _martingale_residuals_one_stratum_numba,
        _martingale_residuals_one_stratum_python,
    )
    from pprof_py.utils.numerical import safe_exp

    X, start, stop, event, offset, weight, strata = _make_dataset(**cfg)
    p = X.shape[1]
    rng = np.random.default_rng(555)

    for efron in (False, True):
        for _ in range(3):
            beta = rng.normal(size=p) * 0.7
            eta = X @ beta + offset
            score = safe_exp(eta)

            for s in np.unique(strata):
                mask = strata == s
                numba_resid = _martingale_residuals_one_stratum_numba(
                    start[mask], stop[mask], event[mask].astype(bool), weight[mask], score[mask], efron
                )
                python_resid = _martingale_residuals_one_stratum_python(
                    start[mask], stop[mask], event[mask].astype(bool), weight[mask], score[mask], efron
                )
                np.testing.assert_allclose(numba_resid, python_resid, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("cfg", SCENARIOS, ids=[f"n{c['n']}_trunc{c['left_trunc']}_strata{c['n_strata']}_w{c['use_weights']}_off{c['use_offset']}" for c in SCENARIOS])
def test_breslow_numba_matches_python_fallback_at_random_beta(cfg):
    """Same check as the Efron version below, for BreslowTies -- its
    stratum_contribution also has an independent numba kernel and Python
    fallback (`_stratum_contribution_python`) that must agree exactly.
    """
    from pprof_py.algorithms.survival.ties import BreslowTies

    X, start, stop, event, offset, weight, strata = _make_dataset(**cfg)
    p = X.shape[1]
    rng = np.random.default_rng(1234)
    tie_method = BreslowTies()

    for s in np.unique(strata):
        mask = strata == s
        if not event[mask].any():
            continue
        for _ in range(2):
            beta = rng.normal(size=p) * 0.7
            eta = X[mask] @ beta + offset[mask]

            numba_contrib = tie_method.stratum_contribution(X[mask], start[mask], stop[mask], event[mask], weight[mask], eta)
            python_contrib = tie_method._stratum_contribution_python(X[mask], start[mask], stop[mask], event[mask], weight[mask], eta)

            assert numba_contrib.log_likelihood == pytest.approx(python_contrib.log_likelihood, rel=1e-8, abs=1e-8)
            np.testing.assert_allclose(numba_contrib.score, python_contrib.score, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(numba_contrib.information, python_contrib.information, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("cfg", SCENARIOS, ids=[f"n{c['n']}_trunc{c['left_trunc']}_strata{c['n_strata']}_w{c['use_weights']}_off{c['use_offset']}" for c in SCENARIOS])
def test_efron_numba_matches_python_fallback_at_random_beta(cfg):
    """EfronTies has two independent implementations of the same formula
    -- a numba-compiled kernel (used whenever numba is installed) and a
    pure-Python, callback-based fallback (`_stratum_contribution_python`
    / `_baseline_hazard_increments_python`, what test_r_comparison.py's
    efron tests originally validated against R before the numba kernel
    existed). They must agree exactly, at more than just the fitted
    optimum, or the two paths would silently give different answers
    depending on whether numba happens to be installed in a given
    environment.
    """
    from pprof_py.algorithms.survival.ties import EfronTies

    X, start, stop, event, offset, weight, strata = _make_dataset(**cfg)
    p = X.shape[1]
    rng = np.random.default_rng(2024)
    tie_method = EfronTies()

    for s in np.unique(strata):
        mask = strata == s
        if not event[mask].any():
            continue
        for _ in range(2):
            beta = rng.normal(size=p) * 0.7
            eta = X[mask] @ beta + offset[mask]

            numba_contrib = tie_method.stratum_contribution(X[mask], start[mask], stop[mask], event[mask], weight[mask], eta)
            python_contrib = tie_method._stratum_contribution_python(X[mask], start[mask], stop[mask], event[mask], weight[mask], eta)

            assert numba_contrib.log_likelihood == pytest.approx(python_contrib.log_likelihood, rel=1e-8, abs=1e-8)
            np.testing.assert_allclose(numba_contrib.score, python_contrib.score, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(numba_contrib.information, python_contrib.information, rtol=1e-8, atol=1e-8)

            numba_times, numba_dH0 = tie_method.baseline_hazard_increments(X[mask], start[mask], stop[mask], event[mask], weight[mask], eta)
            python_times, python_dH0 = tie_method._baseline_hazard_increments_python(X[mask], start[mask], stop[mask], event[mask], weight[mask], eta)
            np.testing.assert_allclose(numba_times, python_times, rtol=1e-10)
            np.testing.assert_allclose(numba_dH0, python_dH0, rtol=1e-8, atol=1e-8)
