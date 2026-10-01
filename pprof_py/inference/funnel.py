"""Funnel-plot coordinates and control limits that agree with ``test()`` by construction.

:func:`funnel_limits` runs a model's ``test()`` once and reuses the null distributions, the calibration
(``null_mean``, ``null_sd``) and the decision rule that produced the flags (ADR-003; decisions D1, D10, D11):

* **count tests** (exact Poisson-binomial and cluster-mixture nulls): for each provider, the smallest count the test
  flags high and the largest it flags low. Limits sit half-way between counts, ``(o_hi - 1/2) / E`` and
  ``(o_lo + 1/2) / E`` on the O/E scale, so no provider lies on a line. These nulls are not functions of the
  expected count E alone, so the limits are per provider; curves are a Poisson(E) reference with the same
  calibration and are labelled as such;
* **CoxPH** (``midp`` and ``exact``): the same search under the test's own Poisson null; curves are exact;
* **score test**: ``1 + (null_mean +/- c * null_sd) * sqrt(V0) / E`` on the O/E scale, exact curves in ``E^2/V0``;
* **Wald tests**: ``reference + q(null_mean +/- c * null_sd) * SE`` on the test's working scale, where ``q`` is the
  Student-t conversion when the test uses one; exact curves in ``1/SE^2``.

Every result is checked before it is returned: a tested provider lies outside its own limits exactly when
``test()`` flagged it. Flags exist only at the test's level; other levels are reference curves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from ._recording import recording
from .count_tests import ClusterMixture, MonteCarlo, PlugIn, RowMixture
from .decision import flags
from .effect_tests import EXACT_P_FLOOR, _clustered_pmf, _integrated_probs, _pmf_tails, poibin_tails_all, z_from_tails
from .empirical_null.models import NullModel, _as_z
from .zstat import z_to_t

__all__ = ["FunnelLimits", "funnel_limits"]

_GRID_POINTS = 200
_TIE = 1e-9


@dataclass(frozen=True, eq=False)
class FunnelLimits:
    """Funnel coordinates and control limits from one provider test.

    Attributes
    ----------
    test : pandas.DataFrame
        The ``test()`` result the limits come from, unchanged (columns
        :data:`~pprof_py.inference.PROVIDER_TEST_COLUMNS`); displays can use it instead of testing again.
    providers : pandas.DataFrame
        Indexed like ``test``: ``observed`` and ``expected`` (NaN for Wald tests), ``estimate`` (O/E for count and
        score tests, the effect on the test's working scale for Wald tests), ``null_value``, ``precision``, ``lower``
        and ``upper`` (control limits at the test's level; ``-inf``/``inf`` where a side cannot be reached, NaN for
        untested providers), ``flag`` and ``null_group``.
    curves : pandas.DataFrame
        Limit curves in long form: ``null_group``, ``level``, ``critical`` (NaN when the level's p-value rule
        applies), ``test_level`` (True for the decision that produced the flags), ``precision``, ``lower``, ``upper``.
        Empty when the calibration varies within a null group.
    attrs : mapping
        ``test()``'s ``attrs`` plus ``model``, ``estimate_kind``, ``precision_kind``, ``limit_rule``, ``curve_kind``
        and ``levels``.
    """

    test: pd.DataFrame
    providers: pd.DataFrame
    curves: pd.DataFrame
    attrs: Mapping[str, Any]


def funnel_limits(model: Any, *args: Any, **kwargs: Any) -> FunnelLimits:
    """Funnel coordinates and control limits that agree with the model's ``test()`` by construction.

    Calls ``model.funnel_limits(*args, **kwargs)``. The model methods take the keywords of the model's ``test()``
    (except Monte Carlo options) plus ``levels``, the levels of the limit curves. They exist for
    :class:`~pprof_py.LogisticFixedEffectModel` (score test by default), :class:`~pprof_py.LogisticRandomEffectModel`
    (count tests only), :class:`~pprof_py.LogisticFERandomClusterModel`, :class:`~pprof_py.LogisticThreeStageModel`,
    :class:`~pprof_py.LinearFixedEffectModel` and :class:`~pprof_py.CoxPH`.

    Returns
    -------
    FunnelLimits

    Raises
    ------
    TypeError
        For models without a justified funnel, such as :class:`~pprof_py.LinearRandomEffectModel`, whose test of
        shrunken estimates no funnel can agree with (ADR-004).
    """
    method = getattr(model, "funnel_limits", None)
    if not callable(method):
        raise TypeError(_unsupported(model))
    return method(*args, **kwargs)


def _unsupported(model: Any) -> str:
    name = type(model).__name__
    if name == "LinearRandomEffectModel":
        return ("LinearRandomEffectModel flags providers with a Wald test of their shrunken estimates (BLUPs); funnel "
                "limits could not agree with those flags (ADR-004). Use an interval plot, or fit a "
                "LinearFixedEffectModel for a funnel.")
    return (f"{name} has no funnel limits. Supported: LogisticFixedEffectModel, LogisticRandomEffectModel (count "
            "tests), LogisticFERandomClusterModel, LogisticThreeStageModel, LinearFixedEffectModel and CoxPH.")


# --------------------------------------------------------------------------------------------------- decisions
class _Calibration(NullModel):
    """Fixed null means and SDs (those ``test()`` fitted), to decide hypothetical statistics with ``flags``."""

    def __init__(self, loc: Any, scale: Any) -> None:
        self.loc = np.asarray(loc, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)

    def parameters(self, z: Any) -> Tuple[np.ndarray, np.ndarray, None]:
        n = len(_as_z(z)[0])
        return np.broadcast_to(self.loc, (n,)).copy(), np.broadcast_to(self.scale, (n,)).copy(), None

    def describe(self) -> dict:
        return {"kind": "calibration"}


def _decide(z: Any, loc: Any, scale: Any, alternative: str, level: float, critical: Optional[float]) -> np.ndarray:
    """The flags ``test()`` would give these z-statistics (+1, -1, 0; NaN where z is missing)."""
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    out = flags(z, _Calibration(loc, scale), alternative=alternative, level=level, critical=critical)
    return out.to_numpy(dtype=np.float64, na_value=np.nan)


def _critical_value(alternative: str, level: float, critical: Optional[float]) -> float:
    if critical is not None:
        return float(critical)
    alpha = 1.0 - float(level)
    return float(norm.ppf(1.0 - (alpha / 2.0 if alternative == "two_sided" else alpha)))


def _z_from(tails: Sequence[Any], alternative: str) -> np.ndarray:
    """``count_test``'s z: mid-p tails for two-sided tests, full tails for one-sided tests."""
    two = alternative == "two_sided"
    return z_from_tails(tails[0] if two else tails[2], tails[1] if two else tails[3], alternative, EXACT_P_FLOOR)


# ----------------------------------------------------------------------------------------- count boundaries
def _first_last(f: np.ndarray, n: int) -> Tuple[int, int]:
    """(largest count flagged -1, smallest flagged +1) from flags at counts 0..n, checking monotonicity."""
    hi, lo = np.flatnonzero(f == 1), np.flatnonzero(f == -1)
    o_hi = int(hi[0]) if hi.size else n + 1
    o_lo = int(lo[-1]) if lo.size else -1
    if (hi.size and hi.size != n + 1 - o_hi) or (lo.size and lo.size != o_lo + 1):
        raise RuntimeError("flags are not monotone in the count; funnel limits are undefined")
    return o_lo, o_hi


def _bisect(zfun: Callable[[np.ndarray, np.ndarray], np.ndarray], nmax: np.ndarray,
            decide: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised over items: the largest count flagged -1 (-1 if none) and smallest flagged +1 (nmax+1 if none).

    ``zfun(o, items)`` gives z at counts ``o`` for those items and ``decide(z, items)`` their flags.
    """
    nmax = np.asarray(nmax, dtype=np.int64)
    found = []
    for target in (1, -1):
        lo, hi = np.full(nmax.size, -1, dtype=np.int64), nmax + 1
        active = np.flatnonzero(hi - lo > 1)
        while active.size:
            mid = (lo[active] + hi[active]) // 2
            hit = decide(zfun(mid, active), active) == target
            if target == 1:
                hi[active[hit]], lo[active[~hit]] = mid[hit], mid[~hit]
            else:
                lo[active[hit]], hi[active[~hit]] = mid[hit], mid[~hit]
            active = active[hi[active] - lo[active] > 1]
        found.append(hi if target == 1 else lo)
    return found[1], found[0]


def _bisect_one(zfun: Callable[[int], float], n: int, decide: Callable[[float], float]) -> Tuple[int, int]:
    o = _bisect(lambda mid, items: np.array([zfun(int(mid[0]))]), np.array([n]),
                lambda z, items: np.array([decide(float(z[0]))]))
    return int(o[0][0]), int(o[1][0])


def _poisson_nmax(expected: np.ndarray, observed: Optional[np.ndarray] = None) -> np.ndarray:
    """Upper end of the count search: beyond z's cap (EXACT_P_FLOOR) for any Poisson mean."""
    top = expected if observed is None else np.maximum(expected, observed)
    return np.ceil(top + 40.0 * np.sqrt(expected) + 50.0).astype(np.int64)


def _poisson_z(o: np.ndarray, expected: np.ndarray, alternative: str) -> np.ndarray:
    """Count-test z of ``o`` under Poisson(expected), with ``count_test``'s tail conventions."""
    o, e = np.asarray(o, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    pmf, cdf, sf = poisson.pmf(o, e), poisson.cdf(o, e), poisson.sf(o, e)
    p_ge = np.where(o > 0, poisson.sf(o - 1.0, e), 1.0)
    return _z_from((sf + 0.5 * pmf, cdf - 0.5 * pmf, p_ge, cdf), alternative)


def _ratio_limits(o_lo: np.ndarray, o_hi: np.ndarray, nmax: np.ndarray, expected: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.where(o_lo >= 0, (o_lo + 0.5) / expected, -np.inf)
    upper = np.where(o_hi <= nmax, (o_hi - 0.5) / expected, np.inf)
    return lower, upper


# ----------------------------------------------------------------------------------------------- utilities
def _positions(order: Any, index: pd.Index) -> np.ndarray:
    pos = pd.Index(np.asarray(order).ravel()).get_indexer(index)
    if (pos < 0).any():
        raise RuntimeError("test() returned providers that the model does not list")
    return pos


def _levels(levels: Optional[Iterable[float]], level: float) -> List[float]:
    values = [float(level)] if levels is None else [float(v) for v in np.atleast_1d(levels)]
    for v in values:
        if not 0.0 < v < 1.0:
            raise ValueError(f"levels must lie strictly between 0 and 1, got {v!r}")
    return sorted(set(values) | {float(level)})


def _calibration_groups(res: pd.DataFrame) -> Optional[List[Tuple[Any, np.ndarray, float, float]]]:
    """(null_group, providers, null_mean, null_sd) for each group; None when the calibration varies within one."""
    tested = res["flag"].notna().to_numpy()
    loc, scl = res["null_mean"].to_numpy(dtype=np.float64), res["null_sd"].to_numpy(dtype=np.float64)
    grp = res["null_group"]
    grouped = not grp.isna().all()
    keys = list(pd.unique(grp[tested & grp.notna().to_numpy()])) if grouped else [pd.NA]
    if grouped:
        try:
            keys = sorted(keys)
        except TypeError:
            pass
    out = []
    for key in keys:
        mask = tested & ((grp == key).fillna(False).to_numpy(dtype=bool) if grouped else True)
        if not mask.any():
            continue
        lm, sm = loc[mask], scl[mask]
        if not (np.all(lm == lm[0]) and np.all(sm == sm[0])):
            return None
        out.append((key, mask, float(lm[0]), float(sm[0])))
    return out


def _grid(precision: np.ndarray) -> np.ndarray:
    p = precision[np.isfinite(precision) & (precision > 0)]
    if p.size == 0:
        return np.empty(0)
    lo, hi = float(p.min()), float(p.max())
    return np.array([lo]) if lo == hi else np.geomspace(lo, hi, _GRID_POINTS)


def _curve_frame(rows: List[Dict[str, Any]], group_dtype: Any) -> pd.DataFrame:
    """Long frame of curve rows; ``null_group`` has the dtype of the test's ``null_group`` column."""
    cols = ["null_group", "level", "critical", "test_level", "precision", "lower", "upper"]
    if rows:
        frame = pd.concat([pd.DataFrame({k: v for k, v in r.items() if k != "null_group"}) for r in rows],
                          ignore_index=True)
        groups = [g for r in rows for g in r["null_group"]]
    else:
        frame = pd.DataFrame({c: pd.Series(dtype=bool if c == "test_level" else float) for c in cols[1:]})
        groups = []
    frame["null_group"] = pd.array(groups, dtype=group_dtype)
    return frame[cols]


def _check_agreement(est: np.ndarray, lower: np.ndarray, upper: np.ndarray, flag: np.ndarray,
                     z_adj: Optional[np.ndarray] = None, c: Optional[float] = None) -> None:
    """Raise unless tested providers lie outside their limits exactly when flagged (S4).

    For continuous statistics a provider whose calibrated z equals the critical value to within rounding is
    moved to the side its flag says (the limits are adjusted in place by one floating-point step).
    """
    tested = ~np.isnan(flag)
    flagged = tested & (flag != 0)
    outside = (est > upper) | (est < lower)
    bad = tested & (outside != flagged)
    if not bad.any():
        return
    tie = np.zeros_like(bad) if z_adj is None else bad & (np.abs(np.abs(z_adj) - c) < _TIE)
    if (bad & ~tie).any():
        raise RuntimeError(f"funnel limits disagree with test() flags for {int((bad & ~tie).sum())} provider(s)")
    for i in np.flatnonzero(tie):
        if flagged[i] and flag[i] > 0:
            upper[i] = np.nextafter(est[i], -np.inf)
        elif flagged[i]:
            lower[i] = np.nextafter(est[i], np.inf)
        else:
            upper[i], lower[i] = max(upper[i], est[i]), min(lower[i], est[i])


def _assemble(res: pd.DataFrame, model: Any, cols: Dict[str, np.ndarray], curves: pd.DataFrame,
              info: Dict[str, Any], levels: List[float]) -> FunnelLimits:
    providers = pd.DataFrame(cols, index=res.index)
    providers["flag"] = res["flag"].array
    providers["null_group"] = res["null_group"].array
    attrs = dict(res.attrs)
    attrs.update({"model": type(model).__name__, "levels": tuple(levels), **info})
    return FunnelLimits(test=res, providers=providers, curves=curves, attrs=attrs)


def _settings(res: pd.DataFrame) -> Tuple[str, float, Optional[float]]:
    a = res.attrs
    return a["alternative"], float(a["level"]), a.get("critical")


def _decision_for(level: float, test_level: float, critical: Optional[float]) -> Tuple[float, Optional[float]]:
    return (level, critical) if level == test_level else (level, None)


# ------------------------------------------------------------------------------------------------ builders
def build_funnel_limits(model: Any, run_test: Callable[[], pd.DataFrame], *, order: Any,
                        levels: Optional[Iterable[float]] = None) -> FunnelLimits:
    """Run ``run_test()`` while recording it, then build count, score or Wald limits from what it used."""
    with recording() as rec:
        res = run_test()
    alt, level, critical = _settings(res)
    levels_ = _levels(levels, level)
    kinds = {r["kind"] for r in rec}
    if "count" in kinds:
        counts = [r for r in rec if r["kind"] == "count"]
        if len(counts) != 1:
            raise RuntimeError("expected one count test inside test()")
        return _count_limits(model, res, counts[0], order, alt, level, critical, levels_)
    if "score" in kinds:
        return _score_limits(model, res, [r for r in rec if r["kind"] == "score"][0], order, alt, level, critical,
                             levels_)
    if res.attrs.get("test_method") == "wald":
        walds = [r for r in rec if r["kind"] == "wald"]
        return _wald_limits(model, res, walds[0]["df"] if walds else None, alt, level, critical, levels_)
    raise TypeError(f"no funnel construction for test_method={res.attrs.get('test_method')!r}")


def _count_limits(model: Any, res: pd.DataFrame, rec: Dict[str, Any], order: Any, alt: str, level: float,
                  critical: Optional[float], levels: List[float]) -> FunnelLimits:
    nulls, g0 = rec["nulls"], rec["g0"]
    if any(isinstance(null, MonteCarlo) for null in nulls):
        raise ValueError("Monte Carlo tests have no funnel limits; use an exact count test.")
    pos = _positions(order, res.index)
    loc, scl = res["null_mean"].to_numpy(dtype=np.float64), res["null_sd"].to_numpy(dtype=np.float64)
    n = len(res)
    observed, expected = rec["obs"][pos].astype(np.float64), np.empty(n)
    o_lo, o_hi, nmax = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    for i, j in enumerate(pos):
        null = nulls[j]
        if isinstance(null, (PlugIn, RowMixture)):
            probs = null.prob(g0) if isinstance(null, PlugIn) else _integrated_probs(null.eta(g0), null.var, null.n_nodes)
            tails = poibin_tails_all(probs, null.trials if isinstance(null, PlugIn) else None)
            expected[i], nmax[i] = tails[4], tails[0].size - 1
            f = _decide(_z_from(tails, alt), loc[i], scl[i], alt, level, critical)
            o_lo[i], o_hi[i] = _first_last(f, int(nmax[i]))
        elif isinstance(null, ClusterMixture):
            pmf = _clustered_pmf(null.eta(g0), null.cluster, null.mean, null.var, null.n_nodes)
            expected[i], nmax[i] = float(np.arange(pmf.size) @ pmf), pmf.size - 1
            o_lo[i], o_hi[i] = _bisect_one(
                lambda o, pmf=pmf: float(_z_from(_pmf_tails(pmf, o), alt)),
                int(nmax[i]), lambda z, i=i: float(_decide(z, loc[i], scl[i], alt, level, critical)[0]))
        else:
            raise TypeError(f"no funnel limits for count nulls of type {type(null).__name__}")
    lower, upper = _ratio_limits(o_lo, o_hi, nmax, expected)
    est = observed / expected
    flag = res["flag"].to_numpy(dtype=np.float64, na_value=np.nan)
    untested = np.isnan(flag)
    lower, upper = np.where(untested, np.nan, lower), np.where(untested, np.nan, upper)
    _check_agreement(est, lower, upper, flag)
    curves = _poisson_curves(res, expected, alt, level, critical, levels, _poisson_z_factory(alt))
    cols = {"observed": observed, "expected": expected, "estimate": est, "null_value": np.ones(n),
            "precision": expected, "lower": lower, "upper": upper}
    info = {"estimate_kind": "ratio", "precision_kind": "expected",
            "limit_rule": "count boundaries at half-integers, per provider",
            "curve_kind": "poisson_reference" if len(curves) else "unavailable"}
    return _assemble(res, model, cols, curves, info, levels)


def _poisson_z_factory(alt: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    return lambda o, e: _poisson_z(o, e, alt)


def _poisson_curves(res: pd.DataFrame, expected: np.ndarray, alt: str, level: float, critical: Optional[float],
                    levels: List[float], zfun: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> pd.DataFrame:
    groups = _calibration_groups(res)
    if groups is None:
        return _curve_frame([], res["null_group"].dtype)
    rows = []
    for key, mask, lg, sg in groups:
        grid = _grid(expected[mask])
        if grid.size == 0:
            continue
        nmax = _poisson_nmax(grid)
        for lev in levels:
            lv, crit = _decision_for(lev, level, critical)
            o_lo, o_hi = _bisect(lambda o, items: zfun(o, grid[items]), nmax,
                                 lambda z, items: _decide(z, lg, sg, alt, lv, crit))
            lower, upper = _ratio_limits(o_lo, o_hi, nmax, grid)
            rows.append({"null_group": [key] * grid.size, "level": lev, "critical": np.nan if crit is None else crit,
                         "test_level": lev == level, "precision": grid, "lower": lower, "upper": upper})
    return _curve_frame(rows, res["null_group"].dtype)


def _score_limits(model: Any, res: pd.DataFrame, rec: Dict[str, Any], order: Any, alt: str, level: float,
                  critical: Optional[float], levels: List[float]) -> FunnelLimits:
    pos = _positions(order, res.index)
    observed, expected, var0 = (np.asarray(rec[k], dtype=np.float64)[pos] for k in ("observed", "expected", "var0"))
    loc, scl = res["null_mean"].to_numpy(dtype=np.float64), res["null_sd"].to_numpy(dtype=np.float64)
    c = _critical_value(alt, level, critical)
    ok = var0 >= 1e-14                                   # test() sets z = 0 below this: never flagged
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.where(ok, np.sqrt(var0) / expected, np.nan)
        precision = np.where(ok, expected ** 2 / var0, np.nan)
    upper = np.where(ok, 1.0 + (loc + c * scl) * se, np.inf)
    lower = np.where(ok, 1.0 + (loc - c * scl) * se, -np.inf)
    if alt == "greater":
        lower = np.full_like(lower, -np.inf)
    elif alt == "less":
        upper = np.full_like(upper, np.inf)
    est = observed / expected
    flag = res["flag"].to_numpy(dtype=np.float64, na_value=np.nan)
    untested = np.isnan(flag)
    lower, upper = np.where(untested, np.nan, lower), np.where(untested, np.nan, upper)
    _check_agreement(est, lower, upper, flag, res["z_adjusted"].to_numpy(dtype=np.float64), c)
    curves = _linear_curves(res, precision, alt, level, critical, levels, lambda q, p: 1.0 + q / np.sqrt(p))
    cols = {"observed": observed, "expected": expected, "estimate": est, "null_value": np.ones(len(res)),
            "precision": precision, "lower": lower, "upper": upper}
    info = {"estimate_kind": "ratio", "precision_kind": "inverse_null_variance",
            "limit_rule": "score test inversion", "curve_kind": "exact" if len(curves) else "unavailable"}
    return _assemble(res, model, cols, curves, info, levels)


def _linear_curves(res: pd.DataFrame, precision: np.ndarray, alt: str, level: float, critical: Optional[float],
                   levels: List[float], limit: Callable[[float, np.ndarray], np.ndarray],
                   q: Callable[[float], float] = lambda z: z) -> pd.DataFrame:
    groups = _calibration_groups(res)
    if groups is None:
        return _curve_frame([], res["null_group"].dtype)
    rows = []
    for key, mask, lg, sg in groups:
        grid = _grid(precision[mask])
        if grid.size == 0:
            continue
        for lev in levels:
            lv, crit = _decision_for(lev, level, critical)
            c = _critical_value(alt, lv, crit)
            upper = limit(q(lg + c * sg), grid) if alt != "less" else np.full(grid.size, np.inf)
            lower = limit(q(lg - c * sg), grid) if alt != "greater" else np.full(grid.size, -np.inf)
            rows.append({"null_group": [key] * grid.size, "level": lev, "critical": np.nan if crit is None else crit,
                         "test_level": lev == level, "precision": grid, "lower": lower, "upper": upper})
    return _curve_frame(rows, res["null_group"].dtype)


def _wald_limits(model: Any, res: pd.DataFrame, df: Optional[float], alt: str, level: float,
                 critical: Optional[float], levels: List[float]) -> FunnelLimits:
    est = res["transformed"].to_numpy(dtype=np.float64)
    s = res["se_transformed"].to_numpy(dtype=np.float64)
    nv = res["null_transformed"].to_numpy(dtype=np.float64)
    loc, scl = res["null_mean"].to_numpy(dtype=np.float64), res["null_sd"].to_numpy(dtype=np.float64)
    c = _critical_value(alt, level, critical)

    def q(z: Any) -> Any:
        return z if df is None else z_to_t(z, df)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        upper = nv + q(loc + c * scl) * s
        lower = nv + q(loc - c * scl) * s
        precision = 1.0 / s ** 2
    if alt == "greater":
        lower = np.full_like(lower, -np.inf)
    elif alt == "less":
        upper = np.full_like(upper, np.inf)
    flag = res["flag"].to_numpy(dtype=np.float64, na_value=np.nan)
    untested = np.isnan(flag)
    lower, upper = np.where(untested, np.nan, lower), np.where(untested, np.nan, upper)
    _check_agreement(est, lower, upper, flag, res["z_adjusted"].to_numpy(dtype=np.float64), c)
    null_value = float(np.nanmedian(nv)) if np.isfinite(nv).any() else np.nan
    curves = _linear_curves(res, precision, alt, level, critical, levels,
                            lambda qv, p: null_value + qv / np.sqrt(p), lambda z: float(q(z)))
    nan = np.full(len(res), np.nan)
    cols = {"observed": nan, "expected": nan, "estimate": est, "null_value": nv, "precision": precision,
            "lower": lower, "upper": upper}
    info = {"estimate_kind": "effect", "precision_kind": "inverse_variance",
            "limit_rule": "Wald test inversion" + ("" if df is None else f" (Student-t, df={df:g})"),
            "curve_kind": "exact" if len(curves) else "unavailable"}
    return _assemble(res, model, cols, curves, info, levels)


def poisson_funnel_limits(model: Any, res: pd.DataFrame, zfun: Callable[[np.ndarray, np.ndarray], np.ndarray],
                          levels: Optional[Iterable[float]] = None) -> FunnelLimits:
    """Limits for tests of a Poisson count against its expected value (CoxPH), with the test's own ``zfun(O, E)``."""
    alt, level, critical = _settings(res)
    levels_ = _levels(levels, level)
    observed = res["observed"].to_numpy(dtype=np.float64)
    expected = res["expected"].to_numpy(dtype=np.float64)
    loc, scl = res["null_mean"].to_numpy(dtype=np.float64), res["null_sd"].to_numpy(dtype=np.float64)
    nmax = _poisson_nmax(expected, observed)
    o_lo, o_hi = _bisect(lambda o, items: zfun(o, expected[items]), nmax,
                         lambda z, items: _decide(z, loc[items], scl[items], alt, level, critical))
    lower, upper = _ratio_limits(o_lo, o_hi, nmax, expected)
    est = observed / expected
    flag = res["flag"].to_numpy(dtype=np.float64, na_value=np.nan)
    untested = np.isnan(flag)
    lower, upper = np.where(untested, np.nan, lower), np.where(untested, np.nan, upper)
    _check_agreement(est, lower, upper, flag)
    curves = _poisson_curves(res, expected, alt, level, critical, levels_, zfun)
    cols = {"observed": observed, "expected": expected, "estimate": est, "null_value": np.ones(len(res)),
            "precision": expected, "lower": lower, "upper": upper}
    info = {"estimate_kind": "ratio", "precision_kind": "expected",
            "limit_rule": "count boundaries at half-integers (Poisson null)",
            "curve_kind": "exact" if len(curves) else "unavailable"}
    return _assemble(res, model, cols, curves, info, levels_)
