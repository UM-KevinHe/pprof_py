"""z-statistics and decisions: schema, conventions, and statistical properties."""
import numpy as np
import pandas as pd
import pytest

from pprof_py.inference import (IDENTITY, LOG, LOGIT, PROVIDER_TEST_COLUMNS, EmpiricalNull, FixedNull,
                                MeasureFrame, TheoreticalNull, ZFrame, assign_groups, flags, intervals,
                                p_values, provider_test, z_statistic)


def _rates(n=300, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.7, n)
    return MeasureFrame.from_arrays(p, rng.uniform(0.01, 0.06, n), measure="direct_rate")


# --------------------------------------------------------------------------- z_statistic
def test_z_statistic_transforms_and_null_values():
    m = _rates()
    z = z_statistic(m, null_value=0.4, transform="logit")
    logit = np.log(m.estimate / (1 - m.estimate))
    assert np.allclose(z.z, (logit - np.log(0.4 / 0.6)) / (m.se / (m.estimate * (1 - m.estimate))), rtol=1e-14)
    assert z_statistic(m, null_value="mean").null_value == pytest.approx(m.estimate.mean())   # 'auto' -> logit
    assert z_statistic(m, null_value="median", transform="log").null_value == pytest.approx(np.median(m.estimate))
    assert z_statistic(m, null_value=lambda e: 0.25, transform=IDENTITY).null_value == 0.25
    ref = MeasureFrame.from_arrays(m.estimate, m.se, measure="direct_rate", reference_value=0.3)
    assert z_statistic(ref).null_value == 0.3
    with pytest.raises(ValueError, match="reference"):
        z_statistic(m)
    with pytest.raises(ValueError, match="auto"):
        z_statistic(MeasureFrame.from_arrays(m.estimate, m.se), null_value="mean")
    with pytest.raises(ValueError, match="domain"):
        z_statistic(m, null_value=1.5, transform="logit")


def test_untestable_providers_get_missing_results():
    m = MeasureFrame.from_arrays([0.3, np.nan, 0.3, 0.3, 1.0, 0.2], [0.02, 0.02, np.nan, 0.0, 0.02, 0.03],
                                 measure="direct_rate")
    res = provider_test(z_statistic(m, null_value=0.25, transform="logit"))
    untested = [False, True, True, True, True, False]
    assert res.z_raw.isna().tolist() == untested and res.p_value.isna().tolist() == untested
    assert res.flag.isna().tolist() == untested and res.ci_lower.isna().tolist() == untested


# --------------------------------------------------------------------------- schema and conventions
def test_schema_and_flag_sign():
    res = provider_test(z_statistic(_rates(), null_value="mean"), FixedNull(mean=0.1, sd=1.3))
    assert tuple(res.columns) == PROVIDER_TEST_COLUMNS and res.index.name == "provider_id"
    assert str(res.flag.dtype) == "Int8"
    assert (res.z_adjusted[res.flag == 1] > 0).all() and (res.z_adjusted[res.flag == -1] < 0).all()
    assert {"transform", "alternative", "level", "critical", "interval", "null_model"} <= set(res.attrs)
    assert res.attrs["null_model"] == {"kind": "fixed", "null_mean": 0.1, "null_sd": 1.3}


def test_one_sided_flags_and_p_values():
    z = z_statistic(_rates(), null_value="mean")
    assert set(flags(z, alternative="greater").dropna().unique()) <= {0, 1}
    assert set(flags(z, alternative="less").dropna().unique()) <= {-1, 0}
    assert np.allclose(p_values(z, alternative="greater") + p_values(z, alternative="less"), 1.0)


def test_critical_value_option():
    z = ZFrame.from_arrays([1.959, 1.9599, 1.96, 1.961, -1.97])
    assert flags(z).tolist() == [0, 0, 1, 1, -1]            # exact quantile 1.959964
    assert flags(z, critical=1.96).tolist() == [0, 0, 0, 1, -1]


def test_bring_your_own_z_has_no_intervals():
    zf = ZFrame.from_arrays(np.random.default_rng(1).normal(size=20))
    res = provider_test(zf)
    assert res.ci_lower.isna().all() and res.flag.notna().all()
    with pytest.raises(ValueError):
        intervals(zf)


# --------------------------------------------------------------------------- statistical properties
def test_type_one_error_under_theoretical_and_fixed_nulls():
    rng = np.random.default_rng(7)
    n = 200_000
    tol = 4 * np.sqrt(0.05 * 0.95 / n)
    assert abs(np.mean(flags(rng.normal(0, 1, n)) != 0) - 0.05) < tol
    assert abs(np.mean(flags(rng.normal(0, 1.7, n), FixedNull(sd=1.7)) != 0) - 0.05) < tol


def _overdispersed(seed, contamination):
    rng = np.random.default_rng(seed)
    size = rng.integers(10, 200, 40_000).astype(float)
    groups = assign_groups(size, 4)
    centre, spread = np.array([0.3, 0.1, -0.1, -0.2]), np.array([2.2, 1.7, 1.4, 1.2])
    z = rng.normal(centre[groups - 1], spread[groups - 1])
    outlier = rng.random(z.size) < contamination
    z[outlier] += np.sign(rng.normal(size=outlier.sum())) * 6 * spread[groups[outlier] - 1]
    return z, groups, outlier


def test_empirical_null_restores_nominal_error_on_overdispersed_data():
    z, groups, outlier = _overdispersed(3, 0.0)
    assert np.mean(flags(z) != 0) > 0.15
    assert abs(np.mean(flags(z, EmpiricalNull.fit(z, groups=groups)) != 0) - 0.05) < 0.006


def test_empirical_null_with_outliers_is_close_to_nominal_and_conservative():
    # 5% true outliers inflate the MAD-based scale by about 6%, so the calibrated test is
    # slightly conservative (about 3.7%), yet far closer to nominal than the uncalibrated one.
    z, groups, outlier = _overdispersed(4, 0.05)
    uncalibrated = np.mean(flags(z)[~outlier] != 0)
    calibrated = np.mean(flags(z, EmpiricalNull.fit(z, groups=groups))[~outlier] != 0)
    assert uncalibrated > 0.15
    assert 0.03 < calibrated <= 0.055


def _duality_cases():
    rng = np.random.default_rng(11)
    for transform, est in ((IDENTITY, rng.normal(0.5, 0.2, 400)), (LOGIT, rng.uniform(0.1, 0.8, 400)),
                           (LOG, rng.lognormal(0.0, 0.3, 400))):
        m = MeasureFrame.from_arrays(est, rng.uniform(0.02, 0.2, 400) * np.abs(est))
        z = z_statistic(m, null_value="median", transform=transform)
        en = EmpiricalNull.fit(z, groups=assign_groups(rng.integers(5, 50, 400), 4))
        for null in (TheoreticalNull(), FixedNull(mean=0.4, sd=1.5), en):
            yield z, null


@pytest.mark.parametrize("alternative", ["two_sided", "greater", "less"])
@pytest.mark.parametrize("critical", [None, 1.96])
def test_inversion_intervals_agree_with_flags(alternative, critical):
    for z, null in _duality_cases():
        f = flags(z, null, alternative=alternative, critical=critical).to_numpy(dtype=float)
        lo, hi = intervals(z, null, alternative=alternative, critical=critical)
        excludes = (lo > z.null_value) | (hi < z.null_value)
        assert ((f != 0) == excludes).all()


def test_scale_only_intervals():
    z = z_statistic(_rates(), null_value="mean")
    for null in (TheoreticalNull(), FixedNull(sd=1.81)):     # location 0: the two forms coincide
        assert np.allclose(intervals(z, null, form="scale_only"), intervals(z, null), rtol=1e-14)
    zf = z_statistic(MeasureFrame.from_arrays([0.35, 0.8], [0.1, 0.1]), null_value=0.5, transform="identity")
    shifted = FixedNull(mean=1.0, sd=1.0)                      # z = -1.5 is flagged (calibrated -2.5) ...
    lo, hi = intervals(zf, shifted, form="scale_only")
    assert flags(zf, shifted).tolist()[0] == -1 and lo[0] < 0.5 < hi[0]   # ... but its scale-only interval covers 0.5


def test_bounds_clip_limits():
    zf = z_statistic(MeasureFrame.from_arrays([0.02, 0.97], [0.05, 0.05]), null_value=0.5, transform="identity")
    lo, hi = intervals(zf, bounds=(0.0, 1.0))
    assert lo[0] == 0.0 and hi[1] == 1.0
