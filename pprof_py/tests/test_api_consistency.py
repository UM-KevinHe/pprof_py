"""API consistency: random-effects attribute names, profiling guards, defaults and warnings."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from pprof_py import (GroupLassoCoxPHCV, LinearRandomEffectModel, LogisticFERandomClusterModel, LogisticFixedEffectModel,
                      LogisticRandomEffectModel, LogisticThreeStageModel, PenalizedCoxPHCV)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def crossed():
    rng = np.random.default_rng(8)
    m, k = 20, 5
    prov = np.repeat(np.arange(m), 40)
    clus = rng.integers(0, k, prov.size)
    x = rng.normal(size=prov.size)
    y = rng.binomial(1, expit(-1 + 0.5 * x + rng.normal(0, 0.4, m)[prov] + rng.normal(0, 0.4, k)[clus])).astype(float)
    yl = x + rng.normal(0, 0.5, m)[prov] + rng.normal(0, 0.5, k)[clus] + rng.normal(size=prov.size)
    return pd.DataFrame({"x": x, "y": y, "yl": yl, "provider": prov, "cluster": clus})


def _fit_re(cls, data, y, **kwargs):
    model = cls(verbose=False)
    extra = {"verbose": False} if cls is LogisticRandomEffectModel else {}
    return _quiet(model.fit, data, y_var=y, x_vars=["x"], provider_var="provider", **kwargs, **extra)


@pytest.mark.parametrize("cls, y", [(LogisticRandomEffectModel, "y"), (LinearRandomEffectModel, "yl")])
def test_random_effect_attributes_follow_the_fixed_effect_names(crossed, cls, y):
    m = _fit_re(cls, crossed, y, cluster_vars=["cluster"])
    assert np.array_equal(np.asarray(m.provider_ids_), np.arange(20))
    assert np.array_equal(np.bincount(m.provider_indices_), m.provider_sizes_) and m.provider_sizes_.sum() == len(crossed)
    assert list(m.cluster_ids_) == ["cluster"] and m.cluster_sizes_["cluster"].sum() == len(crossed)
    assert np.array_equal(np.bincount(m.cluster_indices_["cluster"]), m.cluster_sizes_["cluster"])
    assert not any(hasattr(m, old) for old in ("groups_", "group_sizes_", "group_indices_"))


def test_linear_profiling_needs_a_fit_without_cluster_factors(crossed):
    m = _fit_re(LinearRandomEffectModel, crossed, "yl", cluster_vars=["cluster"])
    for method in (m.calculate_standardized_measures, m.calculate_confidence_intervals, m.test):
        with pytest.raises(ValueError, match="without cluster_vars"):
            method()
    single = _fit_re(LinearRandomEffectModel, crossed, "yl")
    assert "indirect" in single.calculate_standardized_measures()


def test_fixed_effect_wald_warns_about_providers_at_the_bound(crossed):
    d = crossed.copy()
    d.loc[d["provider"] == 0, "y"] = 0.0                    # no events: the effect runs to the bound
    fe = _quiet(LogisticFixedEffectModel(use_dataprep=False, screen_providers=False).fit, d, y_var="y", x_vars=["x"],
                provider_var="provider")
    with pytest.warns(UserWarning, match="at the bound"):
        fe.test(test_method="wald")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fe.test(test_method="wald", providers=[1, 2])        # silent when no tested provider is at the bound


def test_defaults():
    assert LogisticFERandomClusterModel().convergence_criterion == "max_delta_gamma"
    assert LogisticThreeStageModel().convergence_criterion == "max_delta_gamma"
    assert PenalizedCoxPHCV().se_rule == "1se"
    assert GroupLassoCoxPHCV(groups=[0, 1]).se_rule == "1se"


def _legend_texts(ax):
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend is not None else []


def test_untested_providers_are_drawn_as_their_own_category(crossed):
    import inspect
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pprof_py.plotting.coefficients import plot_caterpillar

    df = pd.DataFrame({"estimate": [0.1, -0.2, 0.3, 0.0], "flag": [1.0, -1.0, np.nan, 0.0]}, index=list("abcd"))
    params = inspect.signature(plot_caterpillar).parameters
    kwargs = {k: v for k, v in dict(estimate_col="estimate", flag_col="flag").items() if k in params}
    plt.close("all")
    plot_caterpillar(df, **kwargs)
    texts = _legend_texts(plt.gca())
    assert "Not tested (1)" in texts and not any(t.startswith("Expected (2)") for t in texts)
    plt.close("all")

    fe = _quiet(LogisticFixedEffectModel(use_dataprep=False, screen_providers=False).fit, crossed, y_var="y",
                x_vars=["x"], provider_var="provider")
    real = fe.test(test_method="poibin_exact")
    patched = real.copy()
    patched.loc[patched.index[0], "flag"] = np.nan              # one provider without a test result
    fe.test = lambda *args, **kwargs: patched
    out = _quiet(fe.plot_funnel)
    ax = out[1] if isinstance(out, tuple) else plt.gca()
    assert "Not tested (1)" in _legend_texts(ax)
    plt.close("all")
