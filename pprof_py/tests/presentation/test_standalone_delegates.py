"""The standalone plot_funnel and plot_caterpillar draw through the presentation layer (D45, D60)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pprof_py import plot_caterpillar
from pprof_py.plotting import plot_funnel
from pprof_py.presentation import FigureResult


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(5)
    E = rng.uniform(5, 200, 40)
    est = rng.poisson(E * rng.lognormal(0, 0.15, 40)) / E
    df = pd.DataFrame({"estimate": est, "precision": E}, index=[f"P{i:02d}" for i in range(40)])
    grid = np.linspace(4, 210, 80)
    lim = pd.concat([pd.DataFrame({"precision": grid, "control_lower": 1 - norm.ppf(1 - a / 2) / np.sqrt(grid),
                                   "control_upper": 1 + norm.ppf(1 - a / 2) / np.sqrt(grid), "alpha": a})
                     for a in (0.05, 0.002)], ignore_index=True)
    z = norm.ppf(0.975)
    df["flag"] = np.where(est > 1 + z / np.sqrt(E), 1, np.where(est < 1 - z / np.sqrt(E), -1, 0))
    return df, lim


def _gid(fig, prefix):
    return [a for a in fig.findobj() if str(getattr(a, "get_gid", lambda: None)() or "").startswith(prefix)]


def _quiet(f, *a, **k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = f(*a, **k)
    # warnings from installed libraries (e.g. pyparsing deprecations inside Matplotlib 3.5) are not ours
    return out, [x for x in w if not issubclass(x.category, RuntimeWarning) and "site-packages" not in x.filename]


def test_funnel_draws_the_supplied_limits(frames, tmp_path):
    df, lim = frames
    r, w = _quiet(plot_funnel, df, lim, save_path=str(tmp_path / "f.png"))
    assert isinstance(r, FigureResult) and not w and (tmp_path / "f.png").stat().st_size > 0
    fig, ax = r
    for alpha in (0.05, 0.002):
        level = f"{1 - alpha:g}"
        part = lim[lim["alpha"] == alpha]
        for side, col in (("upper", "control_upper"), ("lower", "control_lower")):
            (line,) = [a for a in _gid(r.figure, f"limit-curve-{level}-") if a.get_gid().endswith(side)]
            np.testing.assert_array_equal(line.get_xdata(), part["precision"].to_numpy())
            np.testing.assert_array_equal(line.get_ydata(), part[col].to_numpy())
    at = lim[lim["alpha"] == 0.05].sort_values("precision")
    for side, col in (("lower", "control_lower"), ("upper", "control_upper")):
        (marks,) = _gid(r.figure, f"limit-marks-{side}")
        expected = np.interp(df["precision"], at["precision"], at[col])               # independent of the code
        np.testing.assert_allclose(np.asarray(marks.get_offsets())[:, 1], expected, rtol=1e-12)
    assert "as supplied" in r.long_description


def test_funnel_warnings_and_legacy_paths(frames):
    df, lim = frames
    bad = df.copy()
    bad.iloc[0, bad.columns.get_loc("flag")] = 1 - abs(int(bad["flag"].iloc[0]))
    _, w = _quiet(plot_funnel, bad, lim, point_size=99)
    kinds = {x.category for x in w}
    assert DeprecationWarning in kinds and UserWarning in kinds                    # styling keyword; S4 contradiction
    with pytest.raises(TypeError, match="unexpected keyword"):
        plot_funnel(df, lim, no_such_option=1)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out, w = _quiet(plot_funnel, df, lim, ax=ax)
    assert out == (fig, ax) and any(issubclass(x.category, DeprecationWarning) for x in w)
    plt.close(fig)


def test_caterpillar_draws_the_frame(frames):
    df, _ = frames
    half = norm.ppf(0.975) / np.sqrt(df["precision"])          # intervals consistent with the flags (S3)
    t = df.assign(ci_lower=df["estimate"] - half, ci_upper=df["estimate"] + half)
    r, w = _quiet(plot_caterpillar, t, flag_col="flag", refline_value=1.0)
    assert isinstance(r, FigureResult) and not w
    # bars keep their bounds in pprof_bounds; their round ends are drawn onto them (D79)
    segs = np.concatenate([np.asarray(a.pprof_bounds if hasattr(a, "pprof_bounds") else a.get_segments())
                           for a in _gid(r.figure, "interval-")])
    np.testing.assert_allclose(np.sort(segs[:, 0, 0]), np.sort(t["ci_lower"].to_numpy()))
    np.testing.assert_allclose(np.sort(segs[:, 1, 0]), np.sort(t["ci_upper"].to_numpy()))
    (ref,) = _gid(r.figure, "reference")
    assert np.all(np.asarray(ref.get_xdata()) == 1.0)
    old = df.assign(lower=df["estimate"] - half, upper=df["estimate"] + half)
    r2, w = _quiet(plot_caterpillar, old, flag_col="flag", refline_value=1.0)
    assert isinstance(r2, FigureResult) and any("ci_lower_col='lower'" in str(x.message) for x in w)
    for kwargs in ({"sort_by_estimate": False}, {"orientation": "horizontal"}, {"refline_value": None}):
        out, w = _quiet(plot_caterpillar, t, flag_col="flag", save_path="/dev/null", **{"refline_value": 1.0, **kwargs})
        assert out is None and any("keeps the earlier drawing" in str(x.message) for x in w)


def test_model_mixins_keep_the_legacy_functions():
    from pprof_py.plotting import coefficients, funnel
    from pprof_py.plotting.linear import random_effect as lre
    from pprof_py.plotting.logistic import fixed_effect as lfe

    assert lfe.plot_caterpillar is coefficients._legacy_plot_caterpillar and lfe._render_funnel is funnel._legacy_plot_funnel
    assert lre.plot_caterpillar is coefficients._legacy_plot_caterpillar
