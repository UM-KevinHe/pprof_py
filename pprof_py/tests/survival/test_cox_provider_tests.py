"""CoxPH.test: the SMR tutorial's Section 4 inference for the indirect standardized ratios."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pprof_py import CoxPH
from pprof_py.inference.empirical_null import EmpiricalNull
from pprof_py.inference.empirical_null.estimators import HUBER_RLM
from pprof_py.inference.survival import poisson_confidence_bounds, poisson_exact_test

Z975 = stats.norm.ppf(0.975)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(29)
    K = 120
    size = rng.integers(20, 120, K)
    g = rng.normal(0, 0.25, K)
    g[rng.choice(K, 6, replace=False)] += 0.7
    prov = np.repeat(np.arange(K), size)
    X = rng.normal(size=(prov.size, 2))
    t = rng.exponential(1.0 / (0.3 * np.exp(X @ [0.5, -0.3] + g[prov])))
    c = rng.uniform(0.5, 3.0, prov.size)
    stop, event = np.minimum(t, c), (t <= c).astype(float)
    m = CoxPH(ties="breslow").fit(X, duration=stop, event=event, strata=prov)
    data = dict(duration=stop, event=event, provider_id=prov)
    return m, X, data


def _excludes_one(t):
    return ((t["ci_lower"] > 1) | (t["ci_upper"] < 1)).to_numpy()


def test_midp_theoretical(fitted):
    m, X, data = fitted
    t = m.test(X, **data)
    o, e = t["observed"].to_numpy(), t["expected"].to_numpy()
    # Eq. 10 and 14: the smaller one-sided mid-p tail, floored at 1e-6, as a signed normal quantile
    q_lo = 2 * stats.poisson.cdf(o, e) - stats.poisson.pmf(o, e)
    q_hi = 2 * stats.poisson.sf(o - 1, e) - stats.poisson.pmf(o, e)
    p = np.maximum(1e-6, np.minimum(q_lo, q_hi) / 2)
    z = np.where(q_lo <= q_hi, stats.norm.ppf(p), -stats.norm.ppf(p))
    assert np.allclose(t["z_raw"], z, rtol=0, atol=1e-10)
    assert np.allclose(t["p_value"], 2 * stats.norm.sf(np.abs(z)), rtol=1e-12)
    assert np.allclose(t["estimate"], o / e)
    assert np.array_equal((t["flag"] != 0).to_numpy(), _excludes_one(t))              # limits invert the test
    lo, hi = poisson_confidence_bounds(o, e, t["p_value"], np.zeros(len(t)), np.ones(len(t)))
    assert np.allclose(t["ci_lower"], lo / e, rtol=0, atol=1e-8)
    assert np.allclose(t["ci_upper"], hi / e, rtol=0, atol=1e-8)


def test_midp_empirical_null_grouped_by_person_time(fitted):
    m, X, data = fitted
    pt = m.calculate_standardized_measures(X, **data)["indirect"]["person_time"].to_numpy()
    t = m.test(X, **data, null_model=EmpiricalNull.fitter(size=pt, n_groups=4, estimator=HUBER_RLM))
    null = EmpiricalNull.fit(t["z_raw"].to_numpy(), size=pt, n_groups=4, estimator=HUBER_RLM)
    assert np.allclose(t["null_mean"], null.mean) and np.allclose(t["null_sd"], null.sd)
    za = (t["z_raw"] - t["null_mean"]) / t["null_sd"]
    assert np.allclose(t["z_adjusted"], za)
    assert np.allclose(t["p_value"], 2 * stats.norm.sf(np.abs(za)))
    assert np.array_equal((t["flag"] != 0).to_numpy(), _excludes_one(t))
    # each finite limit is where the calibrated p-value equals 0.05 (Section 4.2.2)
    from pprof_py.inference.survival import poisson_midp_zscore
    for side in ("ci_lower", "ci_upper"):
        rows = t[np.isfinite(t[side]) & (t[side] > 0)]
        zz = poisson_midp_zscore(rows["observed"], rows[side] * rows["expected"])
        p_at = 2 * stats.norm.sf(np.abs(zz - rows["null_mean"]) / rows["null_sd"])
        assert np.allclose(p_at, 0.05, atol=1e-6)


def test_exact_method(fitted):
    m, X, data = fitted
    t = m.test(X, **data, test_method="exact")
    o, e = t["observed"].to_numpy(), t["expected"].to_numpy()
    p, lo, hi = poisson_exact_test(o, e)
    assert np.allclose(t["p_value"], p, rtol=1e-10)
    naive = np.where(o > e, np.minimum(0.999, 2 * stats.poisson.sf(o - 1, e)), np.minimum(0.999, 2 * stats.poisson.cdf(o, e)))
    assert np.allclose(t["p_value"], naive, rtol=1e-10)                                  # Eq. 9
    assert np.allclose(t["ci_lower"], lo) and np.allclose(t["ci_upper"], hi)


def test_limits_against_the_tutorial():
    # Eq. 12, exact chi-square limits (E < 100), on the count scale: Remark 4.1's "Exact" rows.
    for o, ref in ((100, (81.36, 121.63)), (1000, (938.97, 1063.95))):
        _, lo, hi = poisson_exact_test(np.array([o]), np.array([50.0]))
        assert np.round(lo[0] * 50, 2) == ref[0] and np.round(hi[0] * 50, 2) == ref[1]
    # Eq. 11, Byar's approximation (E >= 100): it matches the exact limits to 0.01 here.  Remark 4.1's "Byar"
    # rows, (81.81, 121.08) and (939.46, 1063.44), are Eq. 11 evaluated at O + 0.5 in both limits.
    byar = lambda o, sgn: o * (1 - 1 / (9 * o) + sgn * Z975 / (3 * np.sqrt(o))) ** 3            # noqa: E731
    for o, remark in ((100, (81.81, 121.08)), (1000, (939.46, 1063.44))):
        _, lo, hi = poisson_exact_test(np.array([o]), np.array([200.0]))
        assert lo[0] * 200 == pytest.approx(byar(o, -1)) and hi[0] * 200 == pytest.approx(byar(o + 1, +1))
        assert (np.round(byar(o + 0.5, -1), 2), np.round(byar(o + 0.5, +1), 2)) == remark


def test_arguments(fitted):
    m, X, data = fitted
    sub = m.test(X, **data, providers=[3, 7])
    assert list(sub.index) == [3, 7]
    with pytest.raises(ValueError, match="test_method"):
        m.test(X, **data, test_method="wald")
    with pytest.raises(ValueError, match="theoretical null"):
        m.test(X, **data, test_method="exact", null_model=EmpiricalNull.fitter())
