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


def test_funnel_warnings_and_removed_options(frames):
    df, lim = frames
    bad = df.copy()
    bad.iloc[0, bad.columns.get_loc("flag")] = 1 - abs(int(bad["flag"].iloc[0]))
    _, w = _quiet(plot_funnel, bad, lim)
    assert UserWarning in {x.category for x in w}                                 # S4: flags contradict the limits
    for removed in ({"point_size": 99}, {"ax": object()}, {"no_such_option": 1}):  # styling keywords and ax= (0.7.0)
        with pytest.raises(TypeError, match="unexpected keyword"):
            plot_funnel(df, lim, **removed)


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
    with pytest.raises(ValueError, match="ci_lower_col='lower'"):           # the lower/upper fallback (0.7.0)
        plot_caterpillar(old, flag_col="flag", refline_value=1.0)
    r2, w = _quiet(plot_caterpillar, old, ci_lower_col="lower", ci_upper_col="upper", flag_col="flag", refline_value=1.0)
    assert isinstance(r2, FigureResult) and not w
    for removed in ({"sort_by_estimate": False}, {"orientation": "horizontal"}, {"point_color": "red"}):
        with pytest.raises(TypeError, match="unexpected keyword"):         # options removed in 0.7.0 (D91)
            plot_caterpillar(t, flag_col="flag", refline_value=1.0, **removed)
    with pytest.raises(ValueError, match="refline_value"):
        plot_caterpillar(t, flag_col="flag", refline_value=None)
    with pytest.raises(ValueError, match="interval columns"):
        plot_caterpillar(df, flag_col="flag", refline_value=1.0)


def test_the_earlier_renderers_are_removed():
    from pprof_py import LinearFixedEffectModel, LinearRandomEffectModel, LogisticFixedEffectModel, \
        LogisticRandomEffectModel
    from pprof_py.plotting import coefficients, funnel

    assert not hasattr(coefficients, "_legacy_plot_caterpillar") and not hasattr(funnel, "_legacy_plot_funnel")
    for cls in (LogisticFixedEffectModel, LogisticRandomEffectModel, LinearFixedEffectModel, LinearRandomEffectModel):
        assert not [n for n in dir(cls) if n.startswith("_legacy")]
