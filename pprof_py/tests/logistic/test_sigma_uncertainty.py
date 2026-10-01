"""B2: the profile-likelihood interval for the random-effect SDs (LogisticRandomEffectModel.profile_sigma) and
Stage 3's flags across it (LogisticThreeStageModel.sigma_sensitivity).

The reference profile is computed in R from lme4's conditional modes, with the Laplace log-determinant evaluated
at the mode.  lme4's own deviance function reports a log-determinant that is not evaluated at its returned mode
away from the optimum, so its confint(method = "profile") limits sit about 1e-4 away (checked loosely).
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticRandomEffectModel, LogisticThreeStageModel

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "data" / "re_profile"

# Environmental: without nlopt, Stage 1 of the random-effects fit falls back from bobyqa to Powell with a warning.
pytestmark = pytest.mark.filterwarnings("ignore:optimizer_stage1='bobyqa' requires the nlopt package:RuntimeWarning")


@pytest.fixture(scope="module")
def fitted():
    d = pd.read_csv(GOLDEN / "raw.csv")
    m = LogisticRandomEffectModel(verbose=False).fit(d, y_var="readmit", x_vars=None, provider_var="facility_id",
                                                     cluster_vars=["hospital_id"], offset_var="off", verbose=False)
    return m, json.loads((GOLDEN / "r_profile.json").read_text())


def test_profile_interval_matches_r(fitted):
    m, r = fitted
    prof = m.profile_sigma()
    assert list(prof.index) == ["facility_id", "hospital_id"]
    for g in prof.index:
        assert prof.loc[g, "sigma"] == pytest.approx(r["sigma"][g], abs=1e-5)
        assert np.allclose(prof.loc[g, ["lower", "upper"]].to_numpy(float), r["exact"][g], atol=1e-6, rtol=0)
        assert np.allclose(prof.loc[g, ["lower", "upper"]].to_numpy(float), r["lme4"][g], atol=3e-4, rtol=0)


def test_profile_arguments(fitted):
    m, _ = fitted
    one = m.profile_sigma("hospital_id", level=0.5)
    assert list(one.index) == ["hospital_id"]
    assert one.loc["hospital_id", "lower"] < one.loc["hospital_id", "sigma"] < one.loc["hospital_id", "upper"]
    with pytest.raises(ValueError, match="Unknown grouping"):
        m.profile_sigma("clinic")
    with pytest.raises(ValueError, match="level"):
        m.profile_sigma(level=1.5)


def _crossed(seed=11, n_fac=40, n_hosp=10):
    rng = np.random.default_rng(seed)
    fq, he = rng.normal(0, 0.3, n_fac), rng.normal(0, 0.4, n_hosp)
    rows = []
    for f in range(n_fac):
        hosp = rng.choice(n_hosp, rng.integers(1, 4), replace=False)
        rows.append(pd.DataFrame({"fac": f + 1, "hosp": rng.choice(hosp, rng.integers(40, 160)) + 1}))
    df = pd.concat(rows, ignore_index=True)
    df["x1"], df["x2"] = rng.normal(size=len(df)), rng.binomial(1, 0.4, len(df))
    lo = -1.2 + 0.4 * df["x1"] + 0.3 * df["x2"] + fq[df["fac"] - 1] + he[df["hosp"] - 1]
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lo)))
    return df


def test_sigma_sensitivity():
    m = LogisticThreeStageModel().fit(_crossed(), "y", ["x1", "x2"], "fac", "hosp")
    sens = m.sigma_sensitivity(test_method="poibin_exact")
    s = sens["sigma"]
    assert s["lower"] < s["estimate"] < s["upper"]
    assert s["estimate"] == pytest.approx(m.stage2_.sigma_["hosp"])
    flags = sens["flags"]
    assert list(flags.columns) == ["lower", "estimate", "upper", "stable"]
    assert np.array_equal(flags["estimate"].to_numpy(), m.test(test_method="poibin_exact")["flag"].to_numpy())
    assert flags["stable"].equals(flags[["lower", "estimate", "upper"]].nunique(axis=1) == 1)
    # the refits at the interval's ends are real refits: their test statistics differ
    lower, upper = (sens["tests"][k].select_dtypes("number") for k in ("lower", "upper"))
    assert not np.allclose(lower.to_numpy(float, na_value=np.nan), upper.to_numpy(float, na_value=np.nan), equal_nan=True)
