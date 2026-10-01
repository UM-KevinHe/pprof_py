"""Between-provider variation: values from the random-effect model per family; the derived range is exact."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pprof_py import LinearRandomEffectModel, LogisticFixedEffectModel, LogisticRandomEffectModel
from pprof_py.presentation import CapabilityError, provider_variation, provider_variation_table


def _data(seed=4, n=40, binary=True):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n), rng.integers(20, 90, n))
    x = rng.normal(size=pid.size)
    eta = rng.normal(0.0, 0.5, n)[pid] + 0.5 * x
    y = rng.binomial(1, 1 / (1 + np.exp(-(eta - 1.0)))) if binary else eta + rng.normal(size=pid.size)
    return pd.DataFrame({"y": y, "x1": x, "provider_id": pid})


@pytest.fixture(scope="module")
def models():
    re, lre = LogisticRandomEffectModel(verbose=False), LinearRandomEffectModel()
    re.fit(_data(), y_var="y", x_vars=["x1"], provider_var="provider_id")
    lre.fit(_data(binary=False), y_var="y", x_vars=["x1"], provider_var="provider_id")
    return re, lre


def _gid(fig, gid):
    return [a for a in fig.findobj() if getattr(a, "get_gid", lambda: None)() == gid]


@pytest.mark.parametrize("level", [0.95, 0.9])
def test_logistic_values_and_drawing(models, level):
    re, _ = models
    r = provider_variation(re, level=level)
    sigma = re.sigma_["provider_id"]
    ps = re.profile_sigma(level=level).loc["provider_id"]
    p = r.provenance
    assert p["sigma"] == sigma and p["lower"] == ps["lower"] and p["upper"] == ps["upper"]
    blups = re.get_random_effects().to_numpy()
    (hist,) = _gid(r.figure, "variation-histogram")
    values, edges = hist.get_data()[:2]
    np.testing.assert_array_equal(values, np.histogram(blups, bins=edges)[0])
    width = edges[1] - edges[0]
    for gid, s in (("variation-fitted", sigma), ("variation-bound-lower", ps["lower"]),
                   ("variation-bound-upper", ps["upper"])):
        (line,) = _gid(r.figure, gid)
        np.testing.assert_allclose(line.get_ydata(), blups.size * width * norm.pdf(line.get_xdata(), 0.0, s),
                                   rtol=1e-12)
    z = norm.ppf(1 - (1 - level) / 2)
    (rng_,) = _gid(r.figure, "variation-range")
    np.testing.assert_array_equal(rng_.get_xdata(), [-z * sigma, z * sigma])
    t = provider_variation_table(re, level=level).to_frame()["value"]
    assert t["As odds ratios: upper"] == np.exp(z * sigma) and t["SD of the BLUPs (descriptive)"] == np.std(blups, ddof=1)
    assert "would lie" in r.long_description and "pulled toward the average" in r.long_description


def test_linear_uses_the_random_effect_sd_not_the_residual_sd(models):
    _, lre = models
    r = provider_variation(lre)
    assert r.provenance["sigma"] == lre.random_effect_sd_["provider_id"] != lre.sigma_
    assert r.provenance["lower"] is None and not _gid(r.figure, "variation-bound-lower")
    assert "no interval" in r.long_description
    t = provider_variation_table(lre)
    assert t.spec.cells.loc["\u03c3, 95% interval: lower", "value"] == "not available"


def test_fixed_effect_models_and_determinism(models):
    fe = LogisticFixedEffectModel()
    fe.fit(_data(), y_var="y", x_vars=["x1"], provider_var="provider_id")
    with pytest.raises(CapabilityError, match="reliability"):
        provider_variation(fe)
    a, b = provider_variation(models[0]), provider_variation(models[0])
    for fmt in ("svg", "pdf", "png"):
        assert a.to_bytes(fmt) == b.to_bytes(fmt)
