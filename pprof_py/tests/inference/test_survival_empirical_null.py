"""Survival empirical-null functions (Poisson mid-p z-scores and R-compatible empirical-null wrappers).

Goldens come from EmpiNull's exported cal_Z_htaz and empirical_null_groupwise, and from MASS::rlm with
the settings of the team's empirical_null.R (bisquare, least-squares start, maxit 1000, acc 1e-8).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pprof_py.inference.survival import (adjust_empirical_null, fit_empirical_null, fit_grouped_empirical_null,
                                         poisson_midp_zscore)
from pprof_py.inference.survival import empirical_null as survival_en

DATA = Path(__file__).parent / "data"
X = pd.read_csv(DATA / "smr_input.csv")
Z = poisson_midp_zscore(X.obs, X.exp)


def test_midp_z_matches_cal_z_htaz():
    np.testing.assert_allclose(Z, pd.read_csv(DATA / "smr_calz.csv").z, rtol=0, atol=1e-12)


@pytest.mark.parametrize("method, key", [("M", "overall"), ("MM", "overall_mm")])
def test_overall_fit_matches_rlm(method, key):
    ref = json.loads((DATA / "smr_overall.json").read_text())[key]
    res = fit_empirical_null(Z, method=method)
    assert abs(res["intercept"] - ref["intercept"]) <= 1e-12 and abs(res["scale"] - ref["scale"]) <= 1e-12


def test_grouped_fit_matches_empinull():
    ref = pd.read_csv(DATA / "smr_groupwise.csv")
    res = fit_grouped_empirical_null(Z, X.yar, n_groups=4)
    assert (res["group"] == ref.group).all()
    g = res["group"].astype(int) - 1
    np.testing.assert_allclose(res["intercept"][g], ref.intercept, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res["scale"][g], ref.scale, rtol=0, atol=1e-12)


@pytest.mark.parametrize("common_mean, tag", [(False, "false"), (True, "true"), (0.0, "zero")])
def test_adjust_matches_empirical_null_adjust(common_mean, tag):
    ref = pd.read_csv(DATA / f"smr_adjust_{tag}.csv")
    res = adjust_empirical_null(Z, size=X.yar, n_groups=4, common_mean=common_mean)
    np.testing.assert_allclose(res["z_adj"], ref.z_adj, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res["p_value"], ref.p_value, rtol=0, atol=1e-12)


def test_missing_sizes_are_ungrouped_as_in_empirical_null_r():
    size = X.yar.to_numpy().copy(); size[:6] = np.nan
    res = adjust_empirical_null(Z, size=size, n_groups=4)
    assert np.isnan(res["group"][:6]).all() and np.isnan(res["z_adj"][:6]).all()
    finite = fit_grouped_empirical_null(Z[6:], size[6:], n_groups=4)
    np.testing.assert_array_equal(fit_grouped_empirical_null(Z, size, n_groups=4)["intercept"], finite["intercept"])


def test_small_groups_raise_and_r_names_exist():
    with pytest.raises(ValueError):
        fit_grouped_empirical_null(Z[:6], X.yar[:6], n_groups=4)
    assert survival_en.cal_Z_htaz is poisson_midp_zscore
    assert survival_en.empirical_null_groupwise is fit_grouped_empirical_null
    assert survival_en.empirical_null_adjust is adjust_empirical_null
