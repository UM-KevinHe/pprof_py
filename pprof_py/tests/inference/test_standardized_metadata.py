"""test_standardized() records its test method, variance and reference effect (approved addition to R7)."""
import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel


@pytest.fixture(scope="module")
def fe():
    rng = np.random.default_rng(8)
    pid = np.repeat(np.arange(30), rng.integers(30, 90, 30))
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(rng.normal(-1.0, 0.4, 30)[pid] + 0.5 * x))))
    m = LogisticFixedEffectModel()
    m.fit(pd.DataFrame({"y": y, "x1": x, "provider_id": pid}), y_var="y", x_vars=["x1"], provider_var="provider_id")
    return m


@pytest.mark.parametrize("kw,method,variance", [
    ({"measure": "indirect_ratio"}, "score", "null"),
    ({"measure": "indirect_rate"}, "score", "null"),
    ({"measure": "indirect_ratio", "indirect_variance": "fitted"}, "wald", "fitted"),
    ({"measure": "indirect_ratio", "transform": "log"}, "wald", "null"),
    ({"measure": "direct_rate"}, "wald", "model"),
    ({"measure": "gamma"}, "wald", "model"),
])
def test_method_and_variance(fe, kw, method, variance):
    a = fe.test_standardized(**kw).attrs
    assert a["test_method"] == method and a["variance"] == variance


@pytest.mark.parametrize("reference", ["median", "mean", -1.0])
def test_reference_is_the_reference_effect_of_test(fe, reference):
    a = fe.test_standardized(measure="indirect_ratio", reference=reference).attrs
    assert a["reference"] == fe.test(test_method="score", reference=reference).attrs["reference"]


def test_score_variant_flags_like_the_score_test(fe):
    std = fe.test_standardized(measure="indirect_ratio")
    score = fe.test(test_method="score")
    np.testing.assert_allclose(std["z_raw"].to_numpy(), score["z_raw"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert std["flag"].equals(score["flag"])
