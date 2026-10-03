"""Model plot methods delegate to pprof_py.presentation; the rest warn and keep their old drawing (D12, spec §13)."""
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from pprof_py import (LinearFixedEffectModel, LinearRandomEffectModel, LogisticFixedEffectModel,
                      LogisticRandomEffectModel)
from pprof_py.plotting import plot_caterpillar
from pprof_py.presentation import FigureResult


def _binary(seed=4, n=30):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n), rng.integers(30, 90, n))
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(rng.normal(-1.0, 0.5, n)[pid] + 0.5 * x))))
    return pd.DataFrame({"y": y, "x1": x, "provider_id": [f"P{j:02d}" for j in pid]})


def _continuous(seed=5, n=25):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n), rng.integers(20, 60, n))
    x = rng.normal(size=pid.size)
    return pd.DataFrame({"y": rng.normal(0, 0.5, n)[pid] + 0.4 * x + rng.normal(size=pid.size), "x1": x,
                         "provider_id": pid})


@pytest.fixture(scope="module")
def models():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fe, re = LogisticFixedEffectModel(), LogisticRandomEffectModel(verbose=False)
        lfe, lre = LinearFixedEffectModel(), LinearRandomEffectModel()
        for m in (fe, re):
            m.fit(_binary(), y_var="y", x_vars=["x1"], provider_var="provider_id")
        for m in (lfe, lre):
            m.fit(_continuous(), y_var="y", x_vars=["x1"], provider_var="provider_id")
    return {"fe": fe, "re": re, "lfe": lfe, "lre": lre}


def _call(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]


@pytest.mark.parametrize("key,test_method", [("fe", "score"), ("lfe", "wald")])
def test_funnel_delegates(models, key, test_method, tmp_path):
    m = models[key]
    r, dep = _call(lambda: m.plot_funnel(alpha=[0.05, 0.002], save_path=str(tmp_path / "f.svg")))
    assert isinstance(r, FigureResult) and r.kind == "funnel" and not dep
    assert r.provenance["test_method"] == test_method and r.provenance["funnel"]["levels"] == (0.95, 0.998)
    fig, ax = r
    assert fig is r.figure and (tmp_path / "f.svg").read_bytes() == r.to_bytes("svg")


def test_removed_keywords_raise(models):
    # styling keywords, target=, use_flags=False and stdz= were deprecated in 0.6.0 and removed in 0.7.0 (D91)
    fe = models["fe"]
    for call, word in ((lambda: fe.plot_funnel(point_colors=["r", "g", "b"]), "point_colors"),
                       (lambda: fe.plot_funnel(target=1.0), "target"),
                       (lambda: fe.plot_provider_effects(figsize=(4, 4)), "figsize"),
                       (lambda: fe.plot_provider_effects(use_flags=False), "use_flags=False"),
                       (lambda: models["lfe"].plot_funnel(stdz="direct"), "stdz")):
        with pytest.raises(TypeError, match=word):
            call()


def test_random_effect_funnel_is_the_count_test(models):
    r, dep = _call(lambda: models["re"].plot_funnel())
    assert r.provenance["test_method"] == "poibin_exact" and not dep and r.kind == "funnel"
    with pytest.raises(ValueError, match="test_method='wald' was removed in 0.7.0"):
        models["re"].plot_funnel(test_method="wald")


@pytest.mark.parametrize("key", ["fe", "re", "lfe", "lre"])
def test_provider_effects_delegate(models, key):
    m = models[key]
    ids = list(m.test().index[:6])
    r, dep = _call(lambda: m.plot_provider_effects(group_ids=ids))
    assert isinstance(r, FigureResult) and r.kind == "caterpillar" and not dep
    assert len(r.figure.axes) >= 2 and r.counts["above"] + r.counts["below"] + r.counts["not_different"] == 6


def test_standardized_measures(models):
    r, dep = _call(lambda: models["fe"].plot_standardized_measures())
    assert isinstance(r, FigureResult) and not dep
    assert r.provenance["measure"] == "indirect_ratio" and r.provenance["test_method"] == "score"
    for key in ("re", "lfe", "lre"):                    # removed in 0.7.0 (D44, D91): no test-consistent display
        assert not hasattr(models[key], "plot_standardized_measures")


def test_linear_random_effect_has_no_funnel(models):
    assert not hasattr(models["lre"], "plot_funnel")    # removed in 0.7.0: no funnel agrees with its test (ADR-004)


def test_plot_caterpillar_reads_test_columns(models, tmp_path):
    t = models["fe"].test().reset_index()
    _, dep = _call(lambda: plot_caterpillar(t, group_col="provider_id", flag_col="flag", save_path=str(tmp_path / "a.png")))
    assert not dep
    old = t.rename(columns={"ci_lower": "lower", "ci_upper": "upper"})
    with pytest.raises(ValueError, match="ci_lower_col='lower'"):                      # the fallback was removed
        plot_caterpillar(old, group_col="provider_id")
    _, dep = _call(lambda: plot_caterpillar(old, ci_lower_col="lower", ci_upper_col="upper", group_col="provider_id",
                                            save_path=str(tmp_path / "b.png")))
    assert not dep


def test_delegated_methods_import_no_pyplot():
    code = ("import sys, warnings, numpy as np, pandas as pd; warnings.simplefilter('ignore')\n"
            "from pprof_py import LogisticFixedEffectModel\n"
            "rng = np.random.default_rng(1); pid = np.repeat(np.arange(15), 40); x = rng.normal(size=pid.size)\n"
            "y = rng.binomial(1, 1 / (1 + np.exp(1.0 - 0.5 * x)))\n"
            "m = LogisticFixedEffectModel(); m.fit(pd.DataFrame({'y': y, 'x1': x, 'provider_id': pid}), y_var='y', "
            "x_vars=['x1'], provider_var='provider_id')\n"
            "m.plot_funnel().to_bytes('png'); m.plot_provider_effects().to_bytes('png')\n"
            "print('matplotlib.pyplot' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
