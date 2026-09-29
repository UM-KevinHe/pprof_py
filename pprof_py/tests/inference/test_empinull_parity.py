"""Parity with EmpiNull::fit_pci (R), the reference for empirical-null calibration.

Golden files were produced by EmpiNull 1.2.0 on synthetic provider-level inputs.
Each configuration below is the pprof_py equivalent of one fit_pci call.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pprof_py.inference import (EmpiricalNull, FixedNull, MEstimator, MeasureFrame, ZFrame, provider_test,
                                z_statistic)

DATA = Path(__file__).parent / "data"
X = pd.read_csv(DATA / "empinull_input.csv")


def _z_only():
    return ZFrame.from_arrays(X.z, provider_id=X.provider_id)


def _wald():
    m = MeasureFrame.from_arrays(X.est, X.se, provider_id=X.provider_id, measure="direct_rate")
    return z_statistic(m, null_value="mean", transform="logit")


CONFIGS = {
    "groupwise_bisquare_quantile": lambda: (_z_only(), lambda z: EmpiricalNull.fit(z, size=X["size"], n_groups=4), {}),
    "groupwise_quantile_ties": lambda: (_z_only(), lambda z: EmpiricalNull.fit(z, size=X["size_tied"], n_groups=4), {}),
    "groupwise_huber_rank": lambda: (_z_only(), lambda z: EmpiricalNull.fit(
        z, size=X["size"], n_groups=4, grouping="rank", estimator=MEstimator(psi="huber", maxiter=1000, tol=1e-8)), {}),
    "overall_common_mean_mask": lambda: (_z_only(), lambda z: EmpiricalNull.fit(
        z, fit_mask=X.fit_mask.to_numpy(), common_mean=True), {}),
    "groupwise_common_zero": lambda: (_z_only(), lambda z: EmpiricalNull.fit(z, size=X["size"], common_mean=0.0), {}),
    "wald_logit_groupwise": lambda: (_wald(), lambda z: EmpiricalNull.fit(z, size=X["size"]), {}),
    "wald_logit_groupwise_greater": lambda: (_wald(), lambda z: EmpiricalNull.fit(z, size=X["size"]),
                                             {"alternative": "greater"}),
    "wald_logit_fixed": lambda: (_wald(), lambda z: FixedNull(sd=1.81), {}),
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_matches_empinull(name):
    ref = pd.read_csv(DATA / f"empinull_{name}.csv").set_index("provider_id")
    z, make_null, options = CONFIGS[name]()
    ours = provider_test(z, make_null(z), **options).loc[ref.index]
    for col_ours, col_ref in (("z_raw", "z_raw"), ("null_mean", "null_mean"), ("null_sd", "null_sd"),
                              ("z_adjusted", "z_adjusted"), ("p_value", "p_value")):
        np.testing.assert_allclose(ours[col_ours], ref[col_ref], rtol=0, atol=1e-12)
    assert (ours.flag.to_numpy(dtype=float) == ref.direction.to_numpy(dtype=float)).all()
    if ours.attrs["null_model"]["kind"] == "empirical":        # EmpiNull labels fixed/theoretical nulls group 1; we report none
        assert (ours.null_group.astype(float).to_numpy() == ref.group.to_numpy(dtype=float)).all()
    if ref.ci_lower.notna().any():
        np.testing.assert_allclose(ours.ci_lower, ref.ci_lower, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ours.ci_upper, ref.ci_upper, rtol=0, atol=1e-12)
