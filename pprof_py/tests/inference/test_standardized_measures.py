"""Model-backed standardized measures and the test_standardized composition."""
import warnings

import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel
from pprof_py.inference import (PROVIDER_TEST_COLUMNS, EmpiricalNull, FixedNull, StandardPopulation, at_bound,
                                provider_test, standardized_measure, z_statistic)

sig = lambda x: 1.0 / (1.0 + np.exp(-x))


@pytest.fixture(scope="module")
def fit():
    rng = np.random.default_rng(42)
    m = 60
    sizes = rng.integers(15, 90, m)
    prov = np.repeat(np.arange(m), sizes)
    n = prov.size
    df = pd.DataFrame(rng.normal(size=(n, 2)), columns=["x1", "x2"])
    df["provider"] = prov
    df["patient"] = [f"{p}-{k // 3}" for p, k in zip(prov, range(n))]          # clustered observations
    g = rng.normal(0, 0.45, m)
    df["y"] = rng.binomial(1, sig(-1.8 + df[["x1", "x2"]].to_numpy() @ [0.5, -0.4] + g[prov])).astype(float)
    df.loc[df.provider.isin([3, 7]), "y"] = 0.0                                    # two zero-event providers
    model = LogisticFixedEffectModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df, x_vars=["x1", "x2"], y_var="y", group_var="provider", obs_id_var="patient")
    return model


def _parts(model):
    gamma = model.coefficients_["gamma"].ravel()
    return gamma, np.median(gamma), model.xbeta_.ravel(), model.outcome_.ravel(), np.asarray(model.group_indices_)


@pytest.mark.parametrize("variance", ["model", "robust"])
def test_direct_measures_match_their_definition(fit, variance):
    gamma, g0, xb, y, _ = _parts(fit)
    se_g = np.sqrt((fit.variances_ if variance == "model" else fit.robust_variances_)["gamma"].ravel())
    P = sig(gamma[:, None] + xb[None, :])
    rate = standardized_measure(fit, "direct_rate", variance=variance)
    assert np.allclose(rate.estimate, P.mean(1), rtol=1e-13)
    assert np.allclose(rate.se, (P * (1 - P)).mean(1) * se_g, rtol=1e-13)
    assert rate.reference_value == pytest.approx(sig(g0 + xb).mean(), rel=1e-13)
    ratio = standardized_measure(fit, "direct_ratio", variance=variance)
    assert np.allclose(ratio.estimate, P.sum(1) / y.sum(), rtol=1e-13)


@pytest.mark.parametrize("indirect_variance", ["null", "fitted"])
def test_indirect_measures_match_their_definition(fit, indirect_variance):
    gamma, g0, xb, y, idx = _parts(fit)
    p0 = sig(g0 + xb)
    pv = p0 if indirect_variance == "null" else sig(gamma[idx] + xb)
    E = np.bincount(idx, p0); O = np.bincount(idx, y); V = np.bincount(idx, pv * (1 - pv))
    ratio = standardized_measure(fit, "indirect_ratio", indirect_variance=indirect_variance)
    assert np.allclose(ratio.estimate, O / E, rtol=1e-13) and np.allclose(ratio.se, np.sqrt(V) / E, rtol=1e-13)
    assert ratio.reference_value == 1.0
    rate = standardized_measure(fit, "indirect_rate", indirect_variance=indirect_variance)
    assert np.allclose(rate.estimate, O / E * y.mean(), rtol=1e-13)
    assert rate.reference_value == pytest.approx(y.mean())


def test_reference_options(fit):
    gamma = fit.coefficients_["gamma"].ravel()
    assert standardized_measure(fit, "gamma").reference_value == pytest.approx(np.median(gamma))
    assert standardized_measure(fit, "gamma", reference="mean").reference_value == pytest.approx(
        np.average(gamma, weights=fit.group_sizes_))
    assert standardized_measure(fit, "gamma", reference=-1.5).reference_value == -1.5
    a = standardized_measure(fit, "indirect_ratio", reference=-1.5)
    b = standardized_measure(fit, "indirect_ratio")
    assert not np.allclose(a.estimate, b.estimate)            # gamma_0 sets the expected counts
    with pytest.raises(ValueError):
        standardized_measure(fit, "gamma", reference="mode")


def test_population_extension(fit):
    gamma, _, xb, y, _ = _parts(fit)
    pop = StandardPopulation.from_model(fit).extend(xbeta=0.0, weight=150.0, events=120.0)
    est = standardized_measure(fit, "direct_rate", population=pop).estimate
    assert np.allclose(est, (sig(gamma[:, None] + xb[None, :]).sum(1) + 150 * sig(gamma)) / (xb.size + 150), rtol=1e-13)
    rate = standardized_measure(fit, "indirect_rate", population=pop)
    assert rate.reference_value == pytest.approx((y.sum() + 120) / (xb.size + 150))
    unknown = StandardPopulation.from_model(fit).extend(xbeta=0.0, weight=150.0)
    standardized_measure(fit, "direct_rate", population=unknown)          # needs no event total
    for measure in ("direct_ratio", "indirect_rate"):
        with pytest.raises(ValueError, match="events"):
            standardized_measure(fit, measure, population=unknown)


def test_invalid_combinations(fit):
    with pytest.raises(ValueError, match="robust"):
        standardized_measure(fit, "indirect_ratio", variance="robust")
    with pytest.raises(ValueError):
        standardized_measure(fit, "indirect_ratio", indirect_variance="wald")
    with pytest.raises(ValueError):
        standardized_measure(fit, "median_rate")
    with pytest.raises(ValueError, match="fitted"):
        standardized_measure(LogisticFixedEffectModel(), "direct_rate")


def test_agrees_with_the_reporting_function(fit):
    sm = fit.calculate_standardized_measures(stdz=["indirect", "direct"], null="median")
    ind, dr = sm["indirect"].set_index("group_id"), sm["direct"].set_index("group_id")
    for measure, ref, unit in (("indirect_ratio", ind.indirect_ratio, 1), ("indirect_rate", ind.indirect_rate, 100),
                               ("direct_ratio", dr.direct_ratio, 1), ("direct_rate", dr.direct_rate, 100)):
        est = standardized_measure(fit, measure).to_frame().estimate.loc[ref.index]
        assert np.allclose(unit * est, ref, rtol=1e-12)              # the reporting function gives rates in percent


def test_indirect_default_is_the_score_test(fit):
    score = fit.test(test_method="score")
    res = fit.test_standardized("indirect_ratio").loc[score.index]
    assert np.allclose(res.z_raw, score.z_raw, rtol=0, atol=1e-12)
    assert (res.flag.to_numpy(dtype=float) == score.flag.to_numpy(dtype=float)).all()
    assert res.z_raw.notna().all()                                    # zero-event providers are tested


def test_test_standardized_defaults_and_null_models(fit):
    res = fit.test_standardized("direct_rate")
    assert tuple(res.columns) == PROVIDER_TEST_COLUMNS
    assert res.attrs["null_model"]["kind"] == "theoretical" and (res.null_sd == 1.0).all()
    assert res.null_value.iloc[0] == pytest.approx(standardized_measure(fit, "direct_rate").reference_value)
    fixed = fit.test_standardized("direct_rate", null_model=FixedNull(sd=1.81))
    assert (fixed.null_sd == 1.81).all()
    sizes = fit.group_sizes_
    via_method = fit.test_standardized("indirect_ratio", null_model=EmpiricalNull.fitter(size=sizes, n_groups=4))
    z = z_statistic(standardized_measure(fit, "indirect_ratio"), null_value="reference", transform="identity")
    by_hand = provider_test(z, EmpiricalNull.fit(z, size=sizes, n_groups=4), bounds=(0.0, np.inf))
    pd.testing.assert_frame_equal(via_method, by_hand)
    with pytest.raises(TypeError):
        fit.test_standardized("direct_rate", null_model="empirical")


def test_providers_subset_and_bounds(fit):
    full = fit.test_standardized("direct_rate", null_value="mean")
    ids = list(full.index[:5])
    sub = fit.test_standardized("direct_rate", null_value="mean", providers=ids)
    pd.testing.assert_frame_equal(sub, full.loc[ids])               # null value still from all providers
    ratio = fit.test_standardized("indirect_ratio")
    assert (ratio.ci_lower >= 0).all()
    raw = fit.test_standardized("indirect_ratio", bounds=None)
    assert (raw.ci_lower < 0).any()                                   # identity-scale limits can go negative unclipped
    rate = fit.test_standardized("direct_rate", transform="identity")
    assert ((rate.ci_lower >= 0) & (rate.ci_upper <= 1)).all()
    with pytest.raises(ValueError):
        fit.test_standardized("direct_rate", bounds="natural")


def test_at_bound_marks_zero_event_providers(fit):
    flagged = set(np.asarray(fit.groups_)[at_bound(fit)].tolist())
    assert {3, 7} <= flagged


def test_indirect_variance_defaults_to_gamma_0(fit):
    default = standardized_measure(fit, "indirect_ratio")
    at_null = standardized_measure(fit, "indirect_ratio", indirect_variance="null")
    assert np.array_equal(default.se, at_null.se)
