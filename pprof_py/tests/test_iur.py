"""Inter-unit reliability: R parity of the decomposition (B11), per-provider reliabilities, split-half exclusions (C29)."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pprof_py.measures.iur import BootstrapIUR, SplitHalfIUR
from pprof_py.measures.iur._core import _iur_decomposition

DATA = Path(__file__).resolve().parent / "data" / "iur_r"


def _goldens():
    inp = pd.read_csv(DATA / "inputs.csv")
    boot = pd.read_csv(DATA / "boot.csv").to_numpy()
    r = pd.read_csv(DATA / "r_iur.csv").iloc[0]
    return inp["size"].to_numpy(), boot, inp["measure"].to_numpy(), r, pd.read_csv(DATA / "r_iur_fac.csv")["IUR.fac"].to_numpy()


def test_decomposition_matches_r_iur_bootdata():
    size, boot, measure, r, _ = _goldens()
    d = _iur_decomposition(size, boot.var(axis=1, ddof=1), measure)
    assert d.n_groups == r.nF
    np.testing.assert_allclose([d.iur, d.s2_between, d.s2_within, d.n_prime],
                               [r.IUR, r.s2_b, r.s2_w, r.n_prime], rtol=1e-13, atol=0)


def test_per_group_reliability_is_the_curve_at_each_size():
    size, boot, measure, r, r_fac = _goldens()
    d = _iur_decomposition(size, boot.var(axis=1, ddof=1), measure)
    np.testing.assert_allclose(d.iur_groups, r.s2_b / (r.s2_b + r.s2_w / size), rtol=1e-13, atol=0)
    # Deliberate deviation from R's IUR.fac, which divides the measure-level variance by the size twice.
    assert np.max(np.abs(d.iur_groups - r_fac)) > 0.5


def _smr_cohort(seed=3, K=300, tau=0.25):
    rng = np.random.default_rng(seed)
    n = rng.integers(20, 401, K)
    prov = np.repeat(np.arange(K), n)
    e = rng.gamma(2.0, 0.05, prov.size)
    O = rng.poisson(e * np.exp(rng.normal(0, tau, K))[prov])
    var_R, mean_R = np.exp(tau**2) * (np.exp(tau**2) - 1), np.exp(tau**2 / 2)
    rho = var_R / (var_R + mean_R / np.bincount(prov, weights=e))
    return O, e, prov, rho


def test_iur_groups_tracks_true_reliability_and_decile_table():
    O, e, prov, rho = _smr_cohort()
    b = BootstrapIUR(n_boot=50, seed=1).fit(O, e, prov)
    np.testing.assert_allclose(b.iur_groups_, b.s2_between_ / (b.s2_between_ + b.s2_within_ / b.group_sizes_),
                               rtol=1e-14, atol=0)
    table = b.decile_table()
    np.testing.assert_allclose(table["min"].iloc[0], b.iur_groups_[np.argmin(b.group_sizes_)], rtol=1e-14)
    np.testing.assert_allclose(table["max"].iloc[0], b.iur_groups_[np.argmax(b.group_sizes_)], rtol=1e-14)
    assert np.mean(np.abs(b.iur_groups_ - rho)) < 0.08          # the old form: about 0.45


def test_split_half_excludes_groups_that_cannot_be_split():
    O, e, prov, _ = _smr_cohort(K=60)
    K = prov.max() + 1
    O1, e1, p1 = np.r_[O, 1.0], np.r_[e, 0.2], np.r_[prov, K]          # one group with a single record
    with pytest.warns(UserWarning, match="cannot be split"):
        m = SplitHalfIUR(n_iter=3, seed=5).fit(O1, e1, p1)
    ref = SplitHalfIUR(n_iter=3, seed=5).fit(O, e, prov)
    assert list(m.excluded_groups_) == [K] and m.n_groups_ == K
    np.testing.assert_array_equal(m.iur_all_, ref.iur_all_)


def test_split_half_without_small_groups_is_silent():
    O, e, prov, _ = _smr_cohort(K=60)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = SplitHalfIUR(n_iter=2, seed=5).fit(O, e, prov)
    assert m.excluded_groups_.size == 0 and m.n_groups_ == 60
