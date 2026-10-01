"""Phase 6 adversarial review: regressions for the defects hostile data exposed (ticks, limits, degenerate sigma,
extreme statistics, label placement)."""
import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel, LogisticRandomEffectModel
from pprof_py.inference import EmpiricalNull
from pprof_py.presentation import ProviderProfile, funnel, null_calibration, observed_expected, provider_variation
from pprof_py.presentation.figures._observed import sqrt_ticks


def _logistic(sizes, effects, seed):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(len(sizes)), sizes)
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1.3 + np.asarray(effects)[pid] + 0.4 * x))))
    return pd.DataFrame({"y": y, "x1": x, "provider_id": [f"P{j:03d}" for j in pid]})


@pytest.fixture(scope="module")
def heavy():
    rng = np.random.default_rng(1)
    d = _logistic(np.r_[20000, rng.integers(30, 150, 59)], np.r_[1.5, rng.normal(0, 0.3, 59)], 4)
    m = LogisticFixedEffectModel()
    m.fit(d, y_var="y", x_vars=["x1"], provider_var="provider_id")
    return m


@pytest.fixture(scope="module")
def ties():
    d = _logistic(np.full(30, 40), np.zeros(30), 5)
    d["y"] = np.tile(np.r_[np.ones(8, int), np.zeros(32, int)], 30)
    fe, re = LogisticFixedEffectModel(), LogisticRandomEffectModel(verbose=False)
    for m in (fe, re):
        m.fit(d, y_var="y", x_vars=["x1"], provider_var="provider_id")
    return fe, re


def _labels(axis):
    return [t.get_text() for t in axis.get_ticklabels() if t.get_text()]


def test_log_axes_over_many_decades_label_powers_of_ten(heavy):
    r = funnel(heavy)
    r.to_bytes("png")                                                      # labels are set at draw time
    labels = _labels(r.axes.xaxis)
    exponents = [np.log10(float(lab.replace(",", ""))) for lab in labels]
    assert labels and np.allclose(exponents, np.round(exponents)), labels   # powers of ten only


def test_square_root_ticks_are_even_and_separated(heavy):
    ticks = sqrt_ticks(10000.0)
    gaps = np.diff(np.sqrt(ticks))
    assert ticks[0] == 0 and ticks[-1] <= 10000 and gaps.max() / gaps.min() < 2.5
    r = observed_expected(ProviderProfile.from_model(heavy, test_method="poibin_exact", limits=True))
    r.to_bytes("png")
    assert any("," in lab for lab in _labels(r.axes.xaxis))                 # thousands separators


def test_null_calibration_keeps_the_bulk_in_view(heavy):
    r = null_calibration(heavy, test_method="score", null_model=EmpiricalNull.fitter())
    z = heavy.test(test_method="score")["z_raw"].to_numpy()
    lo, hi = r.axes.get_xlim()
    assert z.max() > hi and hi <= 10.0                                     # the extreme statistic is off the axis
    assert "counted at its edge" in r.long_description
    assert any(a.get_gid() and a.get_gid().startswith("calibration-beyond-right") for a in r.figure.findobj())


def test_funnel_keeps_its_limits_in_view(ties):
    fe, _ = ties
    prof = ProviderProfile.from_model(fe, limits=True)
    r = funnel(prof)
    f = prof.data
    assert r.axes.get_ylim()[1] >= np.nanmax(f["funnel_upper"].to_numpy()) - 1e-12


def test_variation_with_sigma_at_zero(ties):
    _, re = ties
    r = provider_variation(re)
    gids = {a.get_gid() for a in r.figure.findobj() if a.get_gid()}
    assert "variation-fitted" not in gids and "variation-range" not in gids and "variation-degenerate" in gids
    assert r.axes.get_ylim()[1] < 1e4 and "no between-provider variation" in r.long_description
    assert "Bracket" not in r.long_description
