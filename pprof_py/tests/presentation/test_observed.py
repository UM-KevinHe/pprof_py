"""Observed versus expected: points and count limits equal the profile; limits agree with flags in counts (S4)."""
import numpy as np
import pandas as pd
import pytest

from pprof_py import CoxPH, LogisticFixedEffectModel
from pprof_py.presentation import CapabilityError, ProviderProfile, observed_expected


def _binary(seed=4, n=40):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n), rng.integers(25, 120, n))
    x = rng.normal(size=pid.size)
    gamma = rng.normal(-1.0, 0.5, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(gamma[pid] + 0.5 * x))))
    y[pid == 3] = 0                                                  # zero events
    return pd.DataFrame({"y": y, "x1": x, "provider_id": [f"P{j:02d}" for j in pid]})


@pytest.fixture(scope="module")
def fe():
    m = LogisticFixedEffectModel()
    m.fit(_binary(), y_var="y", x_vars=["x1"], provider_var="provider_id")
    return m


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def test_points_and_count_limits_are_the_profile(fe):
    prof = ProviderProfile.from_model(fe, test_method="poibin_exact", limits=True)
    r = observed_expected(prof)
    f = prof.data
    for key in ("above", "below", "not_different"):
        sel = (f["status"] == key).to_numpy()
        if sel.any():
            (pts,) = _gid(r.figure, f"status-{key}")
            np.testing.assert_array_equal(np.asarray(pts.get_offsets()), f.loc[sel, ["expected", "observed"]].to_numpy())
    e = f["expected"].to_numpy()
    for side, col in (("lower", "funnel_lower"), ("upper", "funnel_upper")):
        lim = f[col].to_numpy() * e
        m = np.isfinite(lim) & (lim >= 0)
        (marks,) = _gid(r.figure, f"count-marks-{side}")
        np.testing.assert_allclose(np.asarray(marks.get_offsets()), np.c_[e[m], lim[m]], rtol=1e-12)
        np.testing.assert_allclose(lim[np.isfinite(lim)] % 1.0, 0.5, atol=1e-8)       # half-integer counts
    lo, hi, o, flag = f["funnel_lower"] * e, f["funnel_upper"] * e, f["observed"], f["flag"]
    np.testing.assert_array_equal(((o > hi) | (o < lo)).to_numpy(), (flag != 0).to_numpy())   # S4 in counts
    assert np.any(f["observed"].to_numpy() == 0) and not _gid(r.figure, "count-limit-upper-")  # references not drawn
    (identity,) = _gid(r.figure, "identity")
    np.testing.assert_array_equal(identity.get_xdata(), identity.get_ydata())
    assert r.axes.get_xscale() == "function" and "square-root" in r.axes.get_xlabel()


def test_score_and_coxph(fe):
    r = observed_expected(fe)                                        # score test: marks only (precision is not E)
    assert r.provenance["test_method"] == "score" and _gid(r.figure, "count-marks-upper")
    rng = np.random.default_rng(29)
    prov = np.repeat(np.arange(25), 60)
    X = rng.normal(size=(prov.size, 2))
    t = rng.exponential(1 / (0.3 * np.exp(X @ [0.5, -0.3])))
    c = rng.uniform(0.5, 3, prov.size)
    data = dict(duration=np.minimum(t, c), event=(t <= c).astype(float), provider_id=prov)
    cox = CoxPH(ties="breslow").fit(X, duration=data["duration"], event=data["event"], strata=prov)
    rc = observed_expected(cox, X, **data)
    (curve,) = _gid(rc.figure, "count-limit-upper-")
    prof = ProviderProfile.from_model(cox, X, limits=True, **data)
    cg = prof.funnel_curves[prof.funnel_curves["test_level"]]
    np.testing.assert_allclose(curve.get_ydata(), (cg["upper"] * cg["precision"]).to_numpy(), rtol=1e-12)


def test_needs_observed_and_expected(fe):
    with pytest.raises(CapabilityError, match="expected"):
        observed_expected(ProviderProfile.from_model(fe))            # no funnel limits: no expected counts


def test_export_is_deterministic(fe):
    a, b = observed_expected(fe), observed_expected(fe)
    for fmt in ("svg", "pdf", "png"):
        assert a.to_bytes(fmt) == b.to_bytes(fmt) == a.to_bytes(fmt)
