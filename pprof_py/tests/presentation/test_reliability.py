"""Reliability display and table: values equal the fitted IUR objects; IUR shown as a property of the measure."""
import numpy as np
import pytest

from pprof_py.measures.iur import BootstrapIUR, DirectIUR, SplitHalfIUR
from pprof_py.presentation import CapabilityError, reliability, reliability_table


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = 60
    size = rng.integers(10, 250, n)
    groups = np.repeat(np.arange(n), size)
    p = 1 / (1 + np.exp(-(-1.5 + rng.normal(0, 0.3, n)[groups])))
    obs = rng.binomial(1, p).astype(float)
    return obs, np.full(obs.size, obs.mean()), groups, size, rng


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


def test_figure_draws_the_fitted_values(data):
    obs, exp_, groups, _, _ = data
    b = BootstrapIUR(n_boot=100, seed=3).fit(obs, exp_, groups)
    r = reliability(b)
    (pts,) = _gid(r.figure, "reliability-providers")
    np.testing.assert_array_equal(np.asarray(pts.get_offsets()), np.c_[b.group_sizes_, b.iur_groups_])
    (curve,) = _gid(r.figure, "reliability-curve")
    order = np.argsort(b.group_sizes_, kind="stable")
    np.testing.assert_array_equal(curve.get_xdata(), np.asarray(b.group_sizes_, dtype=float)[order])
    np.testing.assert_array_equal(curve.get_ydata(), np.asarray(b.iur_groups_)[order])
    assert np.all(np.asarray(_gid(r.figure, "overall-iur")[0].get_ydata()) == b.iur_)
    assert np.all(np.asarray(_gid(r.figure, "n-prime")[0].get_xdata()) == b.n_prime_)
    assert "not a score for any provider" in r.long_description and r.kind == "reliability"
    a, c = reliability(b), reliability(b)
    for fmt in ("svg", "pdf", "png"):
        assert a.to_bytes(fmt) == c.to_bytes(fmt)


def test_tables_equal_the_sources(data):
    obs, exp_, groups, size, rng = data
    b = BootstrapIUR(n_boot=100, seed=3).fit(obs, exp_, groups)
    v = reliability_table(b).to_frame()["value"]
    assert v["Overall IUR"] == b.iur_ and v["Effective size n\u2032"] == b.n_prime_
    assert v["Between-provider variance"] == b.s2_between_ and v["Providers"] == b.n_groups_
    deciles = b.decile_table()
    np.testing.assert_array_equal(v.iloc[5:].to_numpy(), deciles.iloc[0].to_numpy())
    d = DirectIUR().fit(size.astype(float), rng.normal(1, 0.2, size.size), 0.5 / np.sqrt(size))
    assert reliability_table(d).to_frame()["value"]["Overall IUR"] == d.iur_
    with pytest.raises(CapabilityError, match="iur_groups_"):
        reliability(d)
    s = SplitHalfIUR().fit(obs, exp_, groups)
    vs = reliability_table(s).to_frame()["value"]
    assert vs["Kappa (split-half agreement of flags)"] == s.summary()["iur_kappa"].iloc[0]
    with pytest.raises(CapabilityError):
        reliability_table(object())
