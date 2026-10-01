"""The family-support matrix of docs/source/presentation/decision_guide.md, checked on fitted models.

If a display gains or loses a family, this test fails until the guide's matrix is updated with it.
"""
import numpy as np
import pytest

from pprof_py import (CoxPH, LinearFixedEffectModel, LinearRandomEffectModel, LogisticFixedEffectModel,
                      LogisticRandomEffectModel, LogisticThreeStageModel)
from pprof_py.inference import EmpiricalNull
from pprof_py import presentation as P
from pprof_py.presentation._synthetic import provider_data

FAMILIES = ("LogFE", "LogRE", "LinFE", "LinRE", "3stage", "CoxPH")
# the decision guide's matrix: True supported, False refused
MATRIX = {
    "funnel":            (True, True, True, False, True, True),
    "interval plot":     (True, True, True, True, True, True),
    "observed_expected": (True, True, False, False, True, True),
    "forest":            (True, True, True, True, True, True),
    "data_quality":      (True, True, True, True, True, True),
    "null_calibration":  (True, True, True, True, True, True),
    "variation":         (False, True, False, True, False, False),
    "flag_stability":    (True, True, True, True, True, True),
    "several measures":  (True, True, True, True, True, True),
}


@pytest.fixture(scope="module")
def fits():
    d = provider_data(60)
    rng = np.random.default_rng(3)
    lfe, lre = LogisticFixedEffectModel(), LogisticRandomEffectModel(verbose=False)
    for m in (lfe, lre):
        m.fit(d, y_var="y", x_vars=["x1"], provider_var="provider_id")
    nfe, nre = LinearFixedEffectModel(), LinearRandomEffectModel()
    for m in (nfe, nre):
        m.fit(d, y_var="y_cont", x_vars=["x1"], provider_var="provider_id")
    crossed = d.assign(hosp=rng.integers(0, 3, len(d)))
    ts = LogisticThreeStageModel().fit(crossed, "y", ["x1"], "provider_id", "hosp")
    prov = np.repeat(np.arange(30), 60)
    X = rng.normal(size=(prov.size, 2))
    t = rng.exponential(1 / (0.3 * np.exp(X @ [0.5, -0.3])))
    c = rng.uniform(0.5, 3, prov.size)
    data = dict(duration=np.minimum(t, c), event=(t <= c).astype(float), provider_id=prov)
    cox = CoxPH(ties="breslow").fit(X, duration=data["duration"], event=data["event"], strata=prov)
    return {"LogFE": (lfe, (), {}), "LogRE": (lre, (), {}), "LinFE": (nfe, (), {}), "LinRE": (nre, (), {}),
            "3stage": (ts, (), {}), "CoxPH": (cox, (X,), data)}, (lfe, lre, nfe, nre)


CALLS = {
    "funnel": lambda m, a, k: P.funnel(m, *a, **k),
    "interval plot": lambda m, a, k: P.caterpillar(m, *a, **k),
    "observed_expected": lambda m, a, k: P.observed_expected(m, *a, **k),
    "forest": lambda m, a, k: P.forest(m),
    "data_quality": lambda m, a, k: P.data_quality(m, *a, **k),
    "null_calibration": lambda m, a, k: P.null_calibration(m, *a, null_model=EmpiricalNull.fitter(), **k),
    "variation": lambda m, a, k: P.provider_variation(m),
    "flag_stability": lambda m, a, k: P.flag_stability(m, *a, **k),
    "several measures": lambda m, a, k: P.multi_measure(P.ProfileCollection(
        {"A": P.ProviderProfile.from_model(m, *a, **k), "B": P.ProviderProfile.from_model(m, *a, **k)})),
}


@pytest.mark.parametrize("display", list(MATRIX))
def test_matrix_row(fits, display):
    models, _ = fits
    for family, supported in zip(FAMILIES, MATRIX[display]):
        m, a, k = models[family]
        if supported:
            assert isinstance(CALLS[display](m, *[a, k]), P.FigureResult), (display, family)
        else:
            with pytest.raises(ValueError):                       # CapabilityError, or the statistical layer's refusal
                CALLS[display](m, a, k)


def test_shrinkage_pairs_and_logistic_re_funnel_tests(fits):
    _, (lfe, lre, nfe, nre) = fits
    assert P.shrinkage(lfe, lre).kind == P.shrinkage(nfe, nre).kind == "shrinkage"
    assert P.funnel(lre).provenance["test_method"] == "poibin_exact"   # the guide's footnote 1
    with pytest.raises(ValueError, match="shrunken estimates"):
        P.funnel(lre, test_method="wald")


def test_flag_stability_reports_a_failing_sigma_sensitivity(fits):
    """On these data the three-stage sigma_sensitivity() raises (sigma estimated at 0; audit §9); the display omits
    only the sigma scenarios, warns, and says so in its footnote and table note."""
    models, _ = fits
    ts = models["3stage"][0]
    try:
        ts.sigma_sensitivity()
        pytest.skip("sigma_sensitivity() succeeds on these data")
    except Exception:
        pass
    with pytest.warns(UserWarning, match="sigma_sensitivity\\(\\) failed"):
        r = P.flag_stability(ts)
    assert "\u03c3 at lower bound" not in r.provenance["scenarios"] and "are omitted" in r.long_description
    with pytest.warns(UserWarning):
        t = P.flag_stability_table(ts)
    assert "are omitted" in dict(t.spec.notes)["a"]
