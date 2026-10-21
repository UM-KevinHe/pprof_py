"""pprofR's patient-clustered provider tests: the R reference values and their Python mirror.

``r_reference/run_pprofr_clustered.R`` computed the reference values with pprofR f23c2889 (R and package
versions are in ``results/pprofr_clustered_provenance.csv``), with fits that reach the maximum to rounding
error (pprofR's ``backtrack = FALSE``). These tests check the reference files (checksums, exact doubles,
the fits, the cohorts' features), that pprofR keeps a patient's rows in input order, and
that the mirror in ``clustered_reference.py`` reproduces every value. The negative controls break one choice
the mirror makes at a time and show that the reference values catch it.

Cohorts: pprofR's own test cohorts (``sim_fe()`` with seeds 31, 13, 14 and 16; seed 31 also binomial) and
two stress cohorts: ``s1_spanning`` (patients treated at more than one provider, rows interleaved across
providers) and ``s2_long`` (binomial rows, patients with up to 50 rows).
"""
import hashlib
import itertools
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from pprof_py.tests.logistic import clustered_reference as mirror

HERE = os.path.dirname(os.path.abspath(__file__))
R_REFERENCE = os.path.normpath(os.path.join(HERE, "..", "..", "r_reference"))
GENERATOR = "run_pprofr_clustered.R"
INPUTS = "data/pprofr_clustered_inputs.csv"
PROVIDERS = "results/pprofr_clustered_providers.csv"
RHAT = "results/pprofr_clustered_rhat.csv"
COHORTS = "results/pprofr_clustered_cohorts.csv"
PROVENANCE = "results/pprofr_clustered_provenance.csv"
PPROFR_COMMIT = "f23c2889b59108f9506da3f91327c3f239945a63"
COHORT_NAMES = ["t31_binary", "t31_binomial", "t13", "t14", "t16", "s1_spanning", "s2_long"]

# The parity bar, as in pprofR's own tests. z-statistics, p-values and R-hat entries have no units, so the
# absolute part covers values at or near 0 (the median provider's statistics are 0 or rounding noise).
RTOL = ATOL = 1e-10
CLUSTERED = ["robust_wald_stat", "stabilized_wald_stat", "robust_score_stat", "stabilized_score_stat"]
STABILIZED = ["stabilized_wald_stat", "stabilized_score_stat"]


def _path(name):
    return os.path.join(R_REFERENCE, name)


def _read(name):
    path = _path(name)
    header = pd.read_csv(path, nrows=0).columns
    text = {col: str for col in ("cohort", "provider", "patient", "basis") if col in header}
    # R writes missing values as NA; pandas' other default markers would turn the basis "null" into NaN
    return pd.read_csv(path, float_precision="round_trip", dtype=text, keep_default_na=False, na_values=["NA"])


def _provenance():
    table = pd.read_csv(_path(PROVENANCE), dtype=str, keep_default_na=False)
    return dict(zip(table["key"], table["value"]))


def _file_md5(path):
    with open(path, "rb") as fh:  # as written on Linux, whatever a checkout did to line endings
        return hashlib.md5(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def _bytes_md5(values, dtype):
    return hashlib.md5(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def excess(got, want):
    """The largest error in units of the tolerance; at most 1 passes."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    return float(np.max(np.abs(got - want) / (ATOL + RTOL * np.abs(want))))


@dataclass
class Cohort:
    rows: pd.DataFrame       # in input order
    providers: pd.DataFrame  # R's per-provider values, providers in R's (sorted) order
    rhat: pd.DataFrame
    meta: pd.Series

    @property
    def beta(self):
        return self.meta[["beta_x1", "beta_x2", "beta_x3"]].to_numpy(dtype=float)

    @property
    def gamma(self):
        return self.providers["gamma"].to_numpy(dtype=float)

    def args(self, order=None):
        d = self.rows if order is None else self.rows.iloc[order]
        return (d["Y"].to_numpy(dtype=float), d["N"].to_numpy(dtype=float),
                d[["x1", "x2", "x3"]].to_numpy(dtype=float), d["provider"].to_numpy(dtype=str),
                d["patient"].to_numpy(dtype=str))

    def mirror(self, order=None, **kwargs):
        return mirror.clustered_tests(*self.args(order), self.gamma, self.beta, **kwargs)

    def r_order(self):
        """pprofR's data order: the input rows sorted, stably, by provider and then patient."""
        return np.lexsort((self.rows["patient"].to_numpy(dtype=str), self.rows["provider"].to_numpy(dtype=str)))


@pytest.fixture(scope="module")
def cohorts():
    inputs, providers, rhat, meta = _read(INPUTS), _read(PROVIDERS), _read(RHAT), _read(COHORTS)
    meta = meta.set_index("cohort")
    return {name: Cohort(inputs[inputs["cohort"] == name].reset_index(drop=True),
                         providers[providers["cohort"] == name].reset_index(drop=True),
                         rhat[rhat["cohort"] == name].reset_index(drop=True), meta.loc[name])
            for name in meta.index}


# ---- the reference files ---------------------------------------------------------------------------

def test_provenance_and_checksums():
    prov = _provenance()
    assert prov["pprofR_commit"] == PPROFR_COMMIT and prov["pprofR_version"] == "0.2.0"
    for key in ("R", "platform", "os", "blas", "lapack", "rng", "Rcpp", "RcppArmadillo", "poibin", "generated_utc"):
        assert prov[key], key
    assert prov["collate"] == "C"
    assert float(prov["fit_tol"]) == 1e-12 and prov["fit_backtrack"] == "FALSE" and float(prov["check_tol"]) == 1e-10
    listed = {key[len("md5:"):]: value for key, value in prov.items() if key.startswith("md5:")}
    assert sorted(listed) == sorted([INPUTS, PROVIDERS, RHAT, COHORTS])
    for name, md5 in listed.items():
        assert _file_md5(_path(name)) == md5, name
    assert _file_md5(_path(GENERATOR)) == prov["generator_md5"], "the generator changed after the files were written"


def test_an_edited_reference_file_fails_its_checksum(tmp_path):
    data = open(_path(PROVIDERS), "rb").read()
    edited = tmp_path / "providers.csv"
    edited.write_bytes(data[:-2] + bytes([data[-2] ^ 1]) + data[-1:])  # one bit of the last value
    assert _file_md5(_path(PROVIDERS)) == _provenance()["md5:" + PROVIDERS]
    assert _file_md5(str(edited)) != _provenance()["md5:" + PROVIDERS]


def test_every_cohort_is_present(cohorts):
    assert list(cohorts) == COHORT_NAMES


@pytest.mark.parametrize("name", COHORT_NAMES)
def test_values_read_back_as_rs_doubles(cohorts, name):
    c = cohorts[name]
    assert _bytes_md5(c.rows["x1"], "<f8") == c.meta["md5_x1"]
    assert _bytes_md5(c.rows["x3"], "<f8") == c.meta["md5_x3"]
    assert _bytes_md5(c.gamma, "<f8") == c.meta["md5_gamma"]
    assert _bytes_md5(c.beta, "<f8") == c.meta["md5_beta"]


@pytest.mark.parametrize("name", COHORT_NAMES)
def test_reference_fits_solve_their_score_equations(cohorts, name):
    """R's gamma-hat and beta-hat are the maximum to rounding error, so any converged fit can be compared with
    them directly."""
    c = cohorts[name]
    y, n, Z, provider, _ = c.args()
    code = np.searchsorted(c.providers["provider"].to_numpy(dtype=str), provider)
    r = y - n * expit(c.gamma[code] + mirror.linear_predictor(Z, c.beta))
    assert np.max(np.abs(np.bincount(code, weights=r))) <= ATOL
    assert np.max(np.abs(Z.T @ r)) <= ATOL


@pytest.mark.parametrize("name", COHORT_NAMES)
def test_pprofr_keeps_input_order_within_patients(cohorts, name):
    """pprofR's data order is a stable sort of the input by provider and patient, so each patient's rows keep
    their input order (the premise of using input position as the order key, D4)."""
    c = cohorts[name]
    assert _bytes_md5(c.rows["row_id"].to_numpy()[c.r_order()], "<i4") == c.meta["md5_r_order"]
    assert list(c.providers["provider"]) == sorted(c.providers["provider"])


def test_cohorts_have_their_features(cohorts):
    def returns(providers):  # does the patient come back to a provider after visiting another?
        runs = [key for key, _ in itertools.groupby(providers)]
        return len(runs) != len(set(runs))

    for name, c in cohorts.items():
        d, meta = c.rows, c.meta
        clusters = d.groupby(["provider", "patient"], sort=False).size()
        by_patient = d.groupby("patient", sort=False)["provider"]
        assert (meta["n_rows"], meta["n_providers"], meta["n_patients"]) == (len(d), d["provider"].nunique(),
                                                                         d["patient"].nunique()), name
        assert (meta["n_clusters"], meta["max_cluster_rows"]) == (len(clusters), clusters.max()), name
        assert meta["n_spanning_patients"] == (by_patient.nunique() > 1).sum(), name
        assert meta["n_interleaved_patients"] == sum(returns(list(g)) for _, g in by_patient), name
        assert d.groupby("provider")["patient"].nunique().min() >= 10, name  # above pprofR's patient cutoff
        assert meta["fit_tol"] == 1e-12 and meta["fit_iter"] > 0, name
    s1, s2 = cohorts["s1_spanning"].meta, cohorts["s2_long"].meta
    assert s1["n_spanning_patients"] >= 100 and s1["n_interleaved_patients"] >= 50
    assert s2["max_cluster_rows"] == 50 and bool(s2["binomial"])
    corner = cohorts["s2_long"].rhat.query("s == 50 and t == 50")
    assert list(corner["m"]) == [2, 2]  # R-hat's last entry averages over the two 50-row patients


# ---- the mirror reproduces pprofR -----------------------------------------------------------------------

@pytest.mark.parametrize("order", ["input", "pprofR"])
@pytest.mark.parametrize("name", COHORT_NAMES)
def test_mirror_reproduces_pprofr(cohorts, name, order):
    c = cohorts[name]
    out = c.mirror(None if order == "input" else c.r_order())
    P = c.providers
    # both are the correctly rounded midpoint of the middle values (or the middle value itself)
    assert out["gamma0"] == c.meta["gamma0"]
    for col in [k for k in P.columns if k.endswith(("_stat", "_se", "_p", "_ci_lower", "_ci_upper"))]:
        assert excess(out[col], P[col]) <= 1, col
    for col in [k for k in P.columns if k.endswith("_flag")]:
        np.testing.assert_array_equal(out[col], P[col].to_numpy(dtype=float), err_msg=col)
    for basis in ("null", "fit"):
        for mine, theirs in (("U", "U"), ("V", "V"), ("Vs", "Vstab"), ("I", "I")):
            assert excess(out[basis][mine], P[f"{theirs}_{basis}"]) <= 1, f"{theirs}_{basis}"
        R = c.rhat[c.rhat["basis"] == basis]
        s, t = R["s"].to_numpy() - 1, R["t"].to_numpy() - 1
        assert out[basis]["R"].shape == (s.max() + 1, s.max() + 1)
        assert excess(out[basis]["R"][s, t], R["R"]) <= 1, f"R_{basis}"
        np.testing.assert_array_equal(out[basis]["m"][np.maximum(s, t)], R["m"])


@pytest.mark.parametrize("name", COHORT_NAMES)
def test_identity_rhat_gives_the_beta_fixed_model_based_tests(cohorts, name):
    """With R-hat the identity the stabilized variance is the information: the stabilized score test becomes
    pprofR's (beta-fixed) score test, and the stabilized Wald SE becomes 1 / sqrt(I)."""
    c = cohorts[name]
    eye = np.eye(int(c.meta["max_cluster_rows"]))
    out = c.mirror(rhat={"null": eye, "fit": eye})
    assert excess(out["stabilized_score_stat"], c.providers["score_stat"]) <= 1
    assert excess(out["stabilized_wald_se"], 1 / np.sqrt(c.providers["I_fit"])) <= 1


# ---- negative controls --------------------------------------------------------------------------------

def _patient_only_layout(provider, patient):
    """Clusters by patient alone, each charged to the provider of its first row."""
    lay = mirror.layout(provider, patient)
    by_patient = mirror.layout(np.full(len(patient), "-"), patient)
    first = np.array([rows[0] for rows in by_patient.members])
    return mirror.Layout(lay.providers, lay.provider, by_patient.cluster, lay.provider[first],
                         by_patient.position, by_patient.members)


def _reversed_within_patients(patient):
    order = np.arange(len(patient))
    for p in np.unique(patient):
        rows = np.flatnonzero(patient == p)
        order[rows] = rows[::-1]
    return order


def _residuals(c):
    """Standardized residuals at the fit and at the null, with the layout."""
    y, n, Z, provider, patient = c.args()
    lay = mirror.layout(provider, patient)
    xbeta = mirror.linear_predictor(Z, c.beta)
    p = {"fit": expit(c.gamma[lay.provider] + xbeta), "null": expit(np.median(c.gamma) + xbeta)}
    return {basis: mirror.standardized(y - n * q, n * q * (1 - q))[0] for basis, q in p.items()}, lay


def _over_all_clusters(e, lay):
    R, m = mirror.pooled_rhat(e, lay)
    q = np.maximum.outer(np.arange(len(m)), np.arange(len(m)))
    return R * m[q] / len(lay.members)


def _per_provider(c):
    """Stabilized statistics with each provider's R-hat built from its own patients only."""
    y, n, Z, provider, patient = c.args()
    out = c.mirror()
    xbeta = mirror.linear_predictor(Z, c.beta)
    lay = mirror.layout(provider, patient)
    g0 = out["gamma0"]
    for basis, p in (("fit", expit(c.gamma[lay.provider] + xbeta)), ("null", expit(g0 + xbeta))):
        r, v = y - n * p, n * p * (1 - p)
        out[basis]["Vs"] = np.array([mirror.meats(r[rows], v[rows], mirror.layout(provider[rows], patient[rows]))["Vs"][0]
                                     for rows in (np.flatnonzero(lay.provider == i) for i in range(len(lay.providers)))])
    out["stabilized_wald_stat"] = (c.gamma - g0) / (np.sqrt(out["fit"]["Vs"]) / out["fit"]["I"])
    out["stabilized_score_stat"] = out["null"]["U"] / np.sqrt(out["null"]["Vs"])
    return out


def _variant(c, control):
    if control == "patient_only_clusters":
        _, _, _, provider, patient = c.args()
        return c.mirror(lay=_patient_only_layout(provider, patient))
    if control == "reversed_within_patients":
        return c.mirror(_reversed_within_patients(c.rows["patient"].to_numpy(dtype=str)))
    if control == "rhat_per_provider":
        return _per_provider(c)
    e, lay = _residuals(c)
    if control == "centred_rhat":
        return c.mirror(rhat={b: mirror.pooled_rhat(e[b] - e[b].mean(), lay)[0] for b in e})
    if control == "rhat_over_all_clusters":
        return c.mirror(rhat={b: _over_all_clusters(e[b], lay) for b in e})
    if control == "score_rhat_at_fit":  # D5's alternative: the score tests' R-hat from fitted residuals
        return c.mirror(rhat={"null": mirror.pooled_rhat(e["fit"], lay)[0]})
    raise ValueError(control)


# control: (cohorts where it must be caught, statistics it changes)
CONTROLS = {
    "patient_only_clusters": (["s1_spanning"], CLUSTERED),
    "reversed_within_patients": (COHORT_NAMES, STABILIZED),
    "centred_rhat": (COHORT_NAMES, STABILIZED),
    "rhat_over_all_clusters": (COHORT_NAMES, STABILIZED),
    "rhat_per_provider": (COHORT_NAMES, STABILIZED),
    "score_rhat_at_fit": (COHORT_NAMES, ["stabilized_score_stat"]),
}


@pytest.mark.parametrize("name", COHORT_NAMES)
@pytest.mark.parametrize("control", list(CONTROLS))
def test_negative_controls(cohorts, control, name):
    """The comparison with R must fail clearly (error over 1,000 times the tolerance) for every statistic the
    control changes and still pass for the others. Clustering by patient alone only matters when patients are
    treated at more than one provider, and the order of rows within a patient only for the stabilized tests."""
    c = cohorts[name]
    out = _variant(c, control)
    caught, changed = CONTROLS[control]
    for stat in CLUSTERED:
        err = excess(out[stat], c.providers[stat])
        if name in caught and stat in changed:
            assert err > 1e3, (stat, err)
        else:
            assert err <= 1, (stat, err)


def test_a_corrupted_reference_value_is_caught(cohorts):
    c = cohorts["s2_long"]
    got = c.mirror()["stabilized_score_stat"]
    want = c.providers["stabilized_score_stat"].to_numpy(dtype=float).copy()
    assert excess(got, want) <= 1
    want[np.argmax(np.abs(want))] *= 1 + 1e-8
    assert excess(got, want) > 1
