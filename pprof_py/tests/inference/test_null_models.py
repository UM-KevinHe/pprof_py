"""Null models: parameters, groupwise fitting, fit masks, common means, and visible fallbacks."""
import numpy as np
import pandas as pd
import pytest

from pprof_py.inference import (DEFAULT_ESTIMATOR, HUBER_RLM, EmpiricalNull, EmpiricalNullWarning, FixedNull,
                                MEstimator, TheoreticalNull, ZFrame, assign_groups)


def _z(n=400, seed=0):
    return np.random.default_rng(seed).normal(0.2, 1.4, n)


def test_theoretical_and_fixed_parameters():
    z = _z(5)
    m, s, g = TheoreticalNull().parameters(z)
    assert (m == 0).all() and (s == 1).all() and g is None
    m, s, _ = FixedNull(mean=0.5, sd=1.81).parameters(z)
    assert (m == 0.5).all() and (s == 1.81).all()
    for bad in (dict(sd=0.0), dict(sd=-1.0), dict(mean=np.nan)):
        with pytest.raises(ValueError):
            FixedNull(**bad)


def test_overall_fit_uses_the_default_estimator_on_finite_values():
    z = _z(); z[:5] = np.nan
    en = EmpiricalNull.fit(z)
    ref = DEFAULT_ESTIMATOR(z)
    assert (en.mean == ref.location).all() and (en.sd == ref.scale).all()
    assert en.diagnostics.loc[0, "n_fitted"] == z.size - 5


def test_groups_from_size_match_explicit_groups():
    z = _z(); size = np.random.default_rng(1).integers(10, 300, z.size).astype(float)
    a = EmpiricalNull.fit(z, size=size, n_groups=4, grouping="rank", order=np.arange(z.size))
    b = EmpiricalNull.fit(z, groups=assign_groups(size, 4, rule="rank", order=np.arange(z.size)))
    assert (a.mean == b.mean).all() and (a.group == b.group).all()


def test_groupwise_fit_and_fit_mask():
    z = _z(); groups = np.repeat([1, 2, 3, 4], 100)
    mask = np.ones(z.size, bool); mask[:10] = False; z[:10] = 50.0          # wild values, left out of the fit
    en = EmpiricalNull.fit(z, groups=groups, fit_mask=mask, estimator=HUBER_RLM)
    for g in (1, 2, 3, 4):
        members = groups == g
        ref = HUBER_RLM(z[members & mask])
        assert (en.mean[members] == ref.location).all() and (en.sd[members] == ref.scale).all()
    assert en.mean[0] == en.mean[50]                                          # masked providers keep their group's values
    assert en.diagnostics.set_index("group").loc[1, "n_fitted"] == 90


def test_common_mean_pools_eligible_values_and_keeps_group_sds():
    z = _z(); groups = np.repeat([1, 2], 200); mask = np.ones(z.size, bool); mask[:5] = False
    fitted = EmpiricalNull.fit(z, groups=groups, fit_mask=mask)
    pooled = EmpiricalNull.fit(z, groups=groups, fit_mask=mask, common_mean=True)
    assert np.allclose(pooled.mean, z[mask].mean()) and (pooled.sd == fitted.sd).all()
    assert (pooled.diagnostics.fitted_mean == fitted.diagnostics.null_mean).all()
    assert (EmpiricalNull.fit(z, groups=groups, common_mean=0.0).mean == 0.0).all()


def test_small_groups_raise_by_default_or_fall_back_visibly():
    z = _z(); groups = np.r_[np.ones(398, int), 2, 2]
    with pytest.raises(ValueError, match="at least 3"):
        EmpiricalNull.fit(z, groups=groups)
    with pytest.warns(EmpiricalNullWarning, match="fewer than 3"):
        en = EmpiricalNull.fit(z, groups=groups, small_group="theoretical")
    d = en.diagnostics.set_index("group")
    assert d.loc[2, "fallback"] and not d.loc[1, "fallback"]
    assert (en.mean[-2:] == 0).all() and (en.sd[-2:] == 1).all()
    with pytest.raises(ValueError):
        EmpiricalNull.fit(z, min_group_size=1)


def test_degenerate_scale_raises():
    with pytest.raises(ValueError, match="nonpositive"):
        EmpiricalNull.fit(np.r_[np.full(10, 2.0), 1.0, 3.0])


def test_non_convergence_is_reported():
    with pytest.warns(EmpiricalNullWarning, match="did not converge"):
        en = EmpiricalNull.fit(_z(), estimator=MEstimator(maxiter=1, tol=1e-12))
    assert not en.diagnostics.loc[0, "converged"]


def test_custom_estimator_and_missing_labels():
    z = _z()
    assert (EmpiricalNull.fit(z, estimator=lambda v: (float(np.median(v)), 1.5)).sd == 1.5).all()
    with pytest.warns(EmpiricalNullWarning, match="no group label"):
        en = EmpiricalNull.fit(z, groups=np.r_[np.ones(399), np.nan])
    assert np.isnan(en.mean[-1]) and np.isfinite(en.mean[0])


def test_alignment_is_checked():
    zf = ZFrame.from_arrays(_z(10), provider_id=list("abcdefghij"))
    en = EmpiricalNull.fit(zf)
    with pytest.raises(ValueError):
        en.parameters(_z(11))
    with pytest.raises(ValueError):
        en.parameters(ZFrame.from_arrays(_z(10), provider_id=list("abcdefghiz")))


def test_from_parameters():
    en = EmpiricalNull.from_parameters([0.1, 0.2], [1.1, 1.2])
    m, s, _ = en.parameters(np.zeros(2))
    assert m.tolist() == [0.1, 0.2] and s.tolist() == [1.1, 1.2]
    with pytest.raises(ValueError):
        EmpiricalNull.from_parameters([0.0], [0.0])
