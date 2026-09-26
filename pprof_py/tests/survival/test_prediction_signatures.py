"""Aligned discrete-survival prediction methods (C11): one signature, long and wide forms."""
import inspect
import warnings

import numpy as np
import pytest

from pprof_py import DiscreteSurvival, ProviderPenalizedDiscreteSurvival


@pytest.fixture(scope="module")
def fits():
    rng = np.random.default_rng(21)
    n = 240
    X = rng.normal(size=(n, 3))
    prov = rng.integers(0, 5, n)
    t = np.clip(np.ceil(rng.exponential(np.exp(-X @ [0.5, -0.3, 0.2])) * 3), 1, 5)
    ev = rng.binomial(1, 0.7, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return X, t, prov, DiscreteSurvival(n_lambda=5).fit(X, t, ev), ProviderPenalizedDiscreteSurvival(n_lambda=5).fit(X, t, ev, prov)


@pytest.mark.parametrize("name", ["predict_hazard", "predict_survival"])
def test_the_signatures_match_after_provider_id(fits, name):
    _, _, _, ds, pds = fits
    a = list(inspect.signature(getattr(ds, name)).parameters)
    b = [p for p in inspect.signature(getattr(pds, name)).parameters if p != "provider_id"]
    assert a == b == ["X", "time", "lambda_value", "which"]


def test_wide_and_long_forms_agree(fits):
    X, t, prov, ds, pds = fits
    k = len(ds.timepoint_map_)
    wide = ds.predict_hazard(X)
    assert wide.shape == (len(X), k)
    assert np.array_equal(wide.ravel(), ds.predict_hazard(X, time=np.full(len(X), ds.timepoint_map_[-1])))
    pw = pds.predict_hazard(X, prov)
    cut = np.clip(np.searchsorted(pds.time_points_, t) + 1, 1, pw.shape[1])
    assert np.array_equal(pds.predict_hazard(X, prov, time=t), np.concatenate([pw[i, :cut[i]] for i in range(len(X))]))
    assert np.array_equal(pds.predict_survival(X, prov), np.cumprod(1 - pw, axis=1))
    assert np.array_equal(pds.predict_hazard(X, prov, lambda_value=pds.lambda_path_[2]), pds.predict_hazard(X, prov, which=2))
