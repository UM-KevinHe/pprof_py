"""The single robust location-scale core: R parity, equivalence to the survival
implementation it will replace, and basic properties. (Equivalence to the former
general implementation was verified bit-for-bit before that implementation was removed.)"""
import json
from pathlib import Path

import numpy as np
import pytest

from pprof_py.inference import BISQUARE_RLM, HUBER_RLM, MM_RLM, MEstimator, robust_location_scale
from pprof_py.inference.survival.empirical_null import fit_robust_location_scale

DATA = Path(__file__).parent / "data"
CASES = {k: np.array([np.nan if v is None else v for v in vals], dtype=float)
         for k, vals in json.loads((DATA / "rlm_cases.json").read_text()).items()}
REFERENCE = json.loads((DATA / "rlm_reference.json").read_text())


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("preset, key", [(HUBER_RLM, "huber"), (BISQUARE_RLM, "bisquare"), (MM_RLM, "mm")])
def test_matches_mass_rlm(case, preset, key):
    ref = REFERENCE[case][key]
    res = preset(CASES[case])
    assert abs(res.location - ref["location"]) <= 1e-12
    assert abs(res.scale - ref["scale"]) <= 1e-12
    assert res.n_iter == ref["n_iter"]
    assert res.converged == ref["converged"]
    assert res.n_used == ref["n_used"]


def _vectors(seed=0, count=60):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = int(rng.integers(5, 2500))
        z = rng.normal(rng.normal(0, .3), rng.uniform(.8, 2), n)
        k = int(rng.integers(0, n // 5 + 1))
        z[:k] += rng.choice([-5, 5], k)
        yield z


def test_reproduces_survival_implementation_exactly():
    survival_settings = MEstimator(psi="bisquare", tuning=4.685, init="median", maxiter=1000, tol=1e-8)
    for z in _vectors(2):
        res = survival_settings(z)
        assert (res.location, res.scale) == fit_robust_location_scale(z)


@pytest.mark.parametrize("preset", [HUBER_RLM, BISQUARE_RLM, MM_RLM])
def test_affine_equivariance(preset):
    z = next(_vectors(3))
    a, b = preset(z), preset(2.5 + 3.0 * z)
    assert b.location == pytest.approx(2.5 + 3.0 * a.location, rel=1e-10)
    assert b.scale == pytest.approx(3.0 * a.scale, rel=1e-10)


def test_non_finite_values_are_ignored():
    z = next(_vectors(4))
    with_missing = np.r_[z, np.nan, np.inf, -np.inf]
    assert HUBER_RLM(with_missing) == HUBER_RLM(z)


def test_zero_scale_is_reported_not_hidden():
    res = robust_location_scale([2.0, 2.0, 2.0, 1.0, 3.0])   # MAD about the mean is 0 (as in MASS::rlm)
    assert (res.location, res.scale, res.converged, res.n_iter) == (2.0, 0.0, True, 0)


def test_iteration_limit_reports_non_convergence():
    z = next(_vectors(5))
    res = robust_location_scale(z, maxiter=1, tol=1e-12)
    assert res.n_iter == 1 and not res.converged


@pytest.mark.parametrize("kwargs", [dict(psi="cauchy"), dict(maxiter=0), dict(tol=0.0), dict(tuning=-1.0),
                                    dict(init="mode"), dict(method="S"), dict(method="MM"),
                                    dict(method="MM", psi="bisquare", tuning=1.5)])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        robust_location_scale(np.arange(10.0), **kwargs)


def test_needs_a_finite_value():
    with pytest.raises(ValueError):
        robust_location_scale([np.nan, np.inf])


def test_mm_needs_two_values_and_holds_the_s_scale():
    with pytest.raises(ValueError):
        MM_RLM([1.0])
    z = next(_vectors(6))
    a, b = MM_RLM(z), MEstimator(psi="bisquare", method="MM", maxiter=200, tol=1e-10)(z)
    assert a.scale == b.scale                      # the scale is the S-estimate, not re-estimated by the M-step
