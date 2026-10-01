"""Presentation data: one provider test with its denominators, statuses, provenance and capabilities (spec §6).

A :class:`ProviderProfile` holds the result of exactly one provider test. Its estimates, intervals, p-values and flags
are the test's own values; denominators and counts come from test-consistent sources (the funnel limits of the same
test, the CoxPH test's own columns, or the model's own data); nothing is re-estimated. Displays read profiles, check
what they need with :meth:`ProviderProfile.require`, and show the provenance.
"""
from __future__ import annotations

import inspect
import warnings
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from ...inference import PROVIDER_TEST_COLUMNS, degenerate_providers, funnel_limits
from ...inference.funnel import _unsupported

__all__ = ["CapabilityError", "ProviderProfile", "PROFILE_COLUMNS", "STATUSES"]

STATUSES = ("above", "below", "not_different", "not_tested", "suppressed")
"""Primary statuses: one per provider, from ``flag``; ``suppressed`` only under :meth:`ProviderProfile.with_min_volume`."""

TEST_COLUMNS = ("estimate", "se", "ci_lower", "ci_upper", "null_value", "flag", "p_value", "z_raw", "z_adjusted",
                "null_group")
_COUNT_COLUMNS = ("observed", "expected", "denominator", "person_time")
_FUNNEL_COLUMNS = ("funnel_estimate", "funnel_precision", "funnel_lower", "funnel_upper")
_ATTRIBUTE_COLUMNS = ("zero_events", "all_events", "finite_estimate")
PROFILE_COLUMNS = TEST_COLUMNS + _COUNT_COLUMNS + _FUNNEL_COLUMNS + ("status", "has_interval") + _ATTRIBUTE_COLUMNS
"""Columns of :attr:`ProviderProfile.data`, always present (NaN or NA where a source does not provide them)."""

_HINTS = {
    "intervals": "use a test that reports intervals, such as test_method='poibin_exact' or 'wald' (the score test "
                 "has none)",
    "funnel_limits": "build the profile with ProviderProfile.from_model(model, limits=True), which takes the limits "
                     "from the same test as the flags",
    "denominator": "build the profile from a fitted model, or pass denominators= to ProviderProfile.from_test()",
    "observed": "build the profile from a fitted model with a binary outcome, from a CoxPH test, or with limits=True",
    "expected": "use limits=True with a count or score test, or a CoxPH test",
    "degeneracy": "build the profile from a fitted model, or give the zero_events and finite_estimate roles to "
                  "ProviderProfile.from_frame()",
}

_ESTIMATORS = {
    "LogisticFixedEffectModel": "fixed effect (unshrunken)",
    "LinearFixedEffectModel": "fixed effect (unshrunken)",
    "LogisticFERandomClusterModel": "fixed effect (unshrunken; Stage 3)",
    "LogisticThreeStageModel": "fixed effect (unshrunken; Stage 3)",
    "LogisticRandomEffectModel": "random effect (shrunken BLUP)",
    "LinearRandomEffectModel": "random effect (shrunken BLUP)",
    "CoxPH": "observed/expected ratio",
}

_ESSENTIALS = ("model", "estimator", "measure", "scale", "test_method", "reference", "reference_value", "null_model",
               "alternative", "level", "critical")


class CapabilityError(ValueError):
    """A display needs a quantity its input does not provide; the message names it and how to obtain it."""


def _scale(measure: Optional[str], model_name: Optional[str]) -> Optional[str]:
    if measure == "gamma":
        if model_name and model_name.startswith("Linear"):
            return "difference"
        return "log_odds" if model_name and "Logistic" in model_name else "effect"
    if measure in ("indirect_ratio", "direct_ratio"):
        return "ratio"
    if measure in ("indirect_rate", "direct_rate"):
        return "rate"
    return None


def _aligned(values: Any, index: pd.Index) -> pd.Series:
    """A Series on ``index``: Series align by provider, arrays by position."""
    if isinstance(values, pd.Series):
        return values if values.index.equals(index) else values.reindex(index)
    return pd.Series(np.asarray(values, dtype=object), index=index)


def _float(values: Any, index: pd.Index) -> np.ndarray:
    if values is None:
        return np.full(len(index), np.nan)
    return pd.to_numeric(_aligned(values, index), errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)


def _boolean(values: Any, index: pd.Index) -> pd.arrays.BooleanArray:
    if values is None:
        return pd.array([pd.NA] * len(index), dtype="boolean")
    if isinstance(values, (bool, np.bool_)):
        return pd.array([bool(values)] * len(index), dtype="boolean")
    return pd.array(_aligned(values, index).astype("boolean"), dtype="boolean")


def _flag_array(values: Any, index: pd.Index) -> pd.arrays.IntegerArray:
    flag = pd.Series(values, index=index)
    num = pd.to_numeric(flag, errors="coerce")
    bad = flag.notna() & ~num.isin([-1, 0, 1])
    if bad.any():
        raise ValueError(f"flag must be -1, 0, 1 or missing; got {flag[bad].unique()[:5].tolist()}")
    return pd.array(num.to_numpy(dtype=object), dtype="Int64")


def _status(flag: pd.arrays.IntegerArray, suppressed: Optional[np.ndarray] = None) -> pd.Categorical:
    f = np.asarray(flag.to_numpy(dtype=np.float64, na_value=np.nan))
    s = np.where(np.isnan(f), "not_tested", np.where(f > 0, "above", np.where(f < 0, "below", "not_different")))
    if suppressed is not None:
        s = np.where(suppressed, "suppressed", s)
    return pd.Categorical(s, categories=STATUSES)


def _s3_violations(frame: pd.DataFrame, alternative: Optional[str]) -> list:
    """Providers whose interval and flag disagree: the interval must exclude the null value exactly when flagged."""
    flag = frame["flag"].to_numpy(dtype=np.float64, na_value=np.nan)
    lo, hi, nv = (frame[c].to_numpy(dtype=np.float64) for c in ("ci_lower", "ci_upper", "null_value"))
    ok = ~np.isnan(flag) & ~np.isnan(lo) & ~np.isnan(hi) & ~np.isnan(nv)
    above, below = lo > nv, hi < nv
    if alternative == "greater":
        implied = np.where(above, 1, 0)
    elif alternative == "less":
        implied = np.where(below, -1, 0)
    else:
        implied = np.where(above, 1, np.where(below, -1, 0))
    return list(frame.index[ok & (implied != flag)])


def _s4_violations(frame: pd.DataFrame) -> list:
    """Providers whose funnel position and flag disagree: outside the limits exactly when flagged."""
    flag = frame["flag"].to_numpy(dtype=np.float64, na_value=np.nan)
    est, lo, hi = (frame[c].to_numpy(dtype=np.float64) for c in ("funnel_estimate", "funnel_lower", "funnel_upper"))
    ok = ~np.isnan(flag) & ~np.isnan(est) & ~(np.isnan(lo) & np.isnan(hi))
    outside = (est > hi) | (est < lo)
    return list(frame.index[ok & (outside != (flag != 0))])


def _warn_violations(kind: str, ids: list) -> None:
    if ids:
        shown = ", ".join(map(str, ids[:10])) + (" ..." if len(ids) > 10 else "")
        rule = ("the interval excludes the null value exactly when the provider is flagged" if kind == "S3"
                else "a provider lies outside its funnel limits exactly when it is flagged")
        warnings.warn(f"{len(ids)} provider(s) break the rule that {rule} ({kind}): {shown}. Displays mark them.",
                      UserWarning, stacklevel=3)


def _default(model: Any, method: str, key: str, given: Mapping[str, Any]) -> Any:
    """The value of a keyword as the model's method would use it: given, or the method's default."""
    if key in given:
        return given[key]
    for target in (model, getattr(model, "stage3_", None)):
        fn = getattr(target, method, None) if target is not None else None
        if fn is None:
            continue
        p = inspect.signature(fn).parameters.get(key)
        if p is not None and p.default is not inspect.Parameter.empty:
            return p.default
    return None


def _model_counts(model: Any, index: pd.Index) -> Dict[str, Any]:
    """Observed counts, denominators and degeneracy from the model's own data, aligned with ``index``."""
    if callable(getattr(model, "_provider_event_counts", None)):
        d = degenerate_providers(model).reindex(index)
        trials = getattr(model, "N_", None)
        kind = "trials" if trials is not None and np.any(np.asarray(trials, dtype=np.float64) != 1) else "records"
        return {"observed": d["events"], "denominator": d["trials"], "denominator_kind": kind,
                "zero_events": d["zero_events"], "all_events": d["all_events"], "finite_estimate": d["finite_estimate"]}
    sizes, ids = getattr(model, "provider_sizes_", None), getattr(model, "provider_ids_", None)
    if sizes is not None and ids is not None:          # continuous outcomes: provider effects are always finite
        s = pd.Series(np.asarray(sizes, dtype=np.float64), index=pd.Index(np.asarray(ids).ravel()))
        return {"denominator": s.reindex(index), "denominator_kind": "records", "zero_events": False,
                "all_events": False, "finite_estimate": True}
    return {}


class ProviderProfile:
    """One provider test, ready for display: values, statuses, provenance and capabilities.

    Build profiles with :meth:`from_model`, :meth:`from_test` or :meth:`from_frame`; they are immutable.

    Attributes
    ----------
    data : pandas.DataFrame
        A copy of the provider rows, indexed by ``provider_id`` in the source's order, with the columns
        :data:`PROFILE_COLUMNS`. ``status`` is the provider's one primary status (:data:`STATUSES`), from ``flag``;
        ``zero_events``, ``all_events`` and ``finite_estimate`` are attributes that never change it (ADR-005).
    provenance : mapping
        The test's settings and the profile's sources (S2): ``model``, ``estimator``, ``measure``, ``scale``,
        ``test_method``, ``reference`` and ``reference_value``, ``null_model``, ``alternative``, ``level``,
        ``critical`` and more; ``None`` where the source does not say.
    capabilities : frozenset of str
        What the profile can supply: ``"flags"``, ``"intervals"``, ``"denominator"``, ``"observed"``,
        ``"expected"``, ``"funnel_limits"``, ``"degeneracy"``.
    excluded : pandas.DataFrame or None
        Providers removed by data preparation before the test (``None`` when unknown).
    funnel_curves : pandas.DataFrame or None
        The limit curves of :func:`~pprof_py.inference.funnel_limits` when the profile has funnel limits.
    """

    __slots__ = ("_frame", "_provenance", "_capabilities", "_excluded", "_curves")

    def __init__(self, frame: pd.DataFrame, provenance: Mapping[str, Any], *, excluded: Optional[pd.DataFrame] = None,
                 curves: Optional[pd.DataFrame] = None) -> None:
        if list(frame.columns) != list(PROFILE_COLUMNS):
            raise ValueError("a ProviderProfile frame must have exactly the columns PROFILE_COLUMNS; use the "
                             "from_model, from_test or from_frame constructors")
        if len(frame) == 0:
            raise ValueError("a ProviderProfile needs at least one provider")
        if not frame.index.is_unique:
            dup = frame.index[frame.index.duplicated()].unique()[:5].tolist()
            raise ValueError(f"provider_id values must be unique; duplicated: {dup}")
        frame = frame.copy()
        frame.index.name = "provider_id"
        caps = {"flags"}
        if frame["has_interval"].any():
            caps.add("intervals")
        for col in ("denominator", "observed", "expected"):
            if np.isfinite(frame[col].to_numpy(dtype=np.float64)).any():
                caps.add(col)
        if frame[["funnel_lower", "funnel_upper"]].notna().any().any():
            caps.add("funnel_limits")
        if frame["finite_estimate"].notna().all():
            caps.add("degeneracy")
        object.__setattr__(self, "_frame", frame)
        object.__setattr__(self, "_provenance", MappingProxyType(dict(provenance)))
        object.__setattr__(self, "_capabilities", frozenset(caps))
        object.__setattr__(self, "_excluded", None if excluded is None else excluded.copy())
        object.__setattr__(self, "_curves", None if curves is None else curves.copy())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ProviderProfile is immutable; build a new profile instead")

    # ------------------------------------------------------------------------------------------- accessors
    @property
    def data(self) -> pd.DataFrame:
        return self._frame.copy()

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    @property
    def capabilities(self) -> frozenset:
        return self._capabilities

    @property
    def excluded(self) -> Optional[pd.DataFrame]:
        return None if self._excluded is None else self._excluded.copy()

    @property
    def funnel_curves(self) -> Optional[pd.DataFrame]:
        return None if self._curves is None else self._curves.copy()

    def __len__(self) -> int:
        return len(self._frame)

    def status_counts(self) -> Dict[str, Optional[int]]:
        """Providers per primary status, plus the attributes and exclusions displays footnote (S6)."""
        f = self._frame
        out = {s: int((f["status"] == s).sum()) for s in STATUSES}
        out["zero_events"] = int(f["zero_events"].fillna(False).sum())
        out["no_finite_estimate"] = int((~f["finite_estimate"].fillna(True)).sum())
        out["no_interval"] = int((~f["has_interval"]).sum())
        out["excluded"] = None if self._excluded is None else len(self._excluded)
        return out

    def __repr__(self) -> str:
        p, c = self._provenance, self.status_counts()
        parts = ", ".join(f"{k.replace('_', ' ')} {c[k]}" for k in STATUSES if c[k])
        return (f"ProviderProfile({len(self)} providers; {p.get('model') or 'frame'}, "
                f"test_method={p.get('test_method')!r}, level={p.get('level')}; {parts})")

    def require(self, display: str, *needed: str) -> None:
        """Raise :class:`CapabilityError` unless the profile supplies everything ``display`` needs."""
        missing = [n for n in needed if n not in self._capabilities]
        if missing:
            hints = "; ".join(f"{n}: {_HINTS.get(n, 'not available from this source')}" for n in missing)
            raise CapabilityError(f"{display} needs {', '.join(missing)}, which this profile does not have ({hints}).")

    def with_min_volume(self, min_volume: float) -> "ProviderProfile":
        """A copy in which providers whose denominator is below ``min_volume`` have status ``suppressed``.

        Flags, values and every other column are unchanged; the rule and its counts go into the provenance, so
        displays footnote it (S10, S11).
        """
        self.require("with_min_volume", "denominator")
        f = self._frame.copy()
        small = f["denominator"].to_numpy(dtype=np.float64) < float(min_volume)
        f["status"] = _status(pd.array(f["flag"]), small)
        prov = dict(self._provenance)
        prov.update({"min_volume": float(min_volume), "min_volume_kind": prov.get("denominator_kind"),
                     "suppressed": int(small.sum())})
        return ProviderProfile(f, prov, excluded=self._excluded, curves=self._curves)

    # ------------------------------------------------------------------------------------------ adapters
    @classmethod
    def from_model(cls, model: Any, *args: Any, limits: bool = False, levels: Optional[Iterable[float]] = None,
                   **test_kwargs: Any) -> "ProviderProfile":
        """Run the model's test once and build its profile.

        Parameters
        ----------
        model
            A fitted provider model.
        *args, **test_kwargs
            Passed to ``model.test()`` (or ``model.funnel_limits()``), for example ``test_method``, ``reference``,
            ``null_model``, ``level``, ``providers``, or a CoxPH model's data.
        limits : bool, default False
            Also take funnel limits from :func:`~pprof_py.inference.funnel_limits` (the same single test). The
            funnel's defaults apply, such as the score test for logistic fixed-effect models.
        levels : sequence of float, optional
            Levels of the funnel curves (with ``limits=True``).

        Raises
        ------
        CapabilityError
            When ``limits=True`` and the model has no justified funnel (ADR-004).
        """
        method = "funnel_limits" if limits else "test"
        if limits:
            if not callable(getattr(model, "funnel_limits", None)):
                raise CapabilityError(_unsupported(model))
            fl = funnel_limits(model, *args, levels=levels, **test_kwargs)
            res, funnel = fl.test, fl
        else:
            res, funnel = model.test(*args, **test_kwargs), None
        given = {"reference": _default(model, method, "reference", test_kwargs),
                 "seed": test_kwargs.get("seed"), "providers": test_kwargs.get("providers")}
        return cls._build(res, model=model, funnel=funnel, settings=given, source="model")

    @classmethod
    def from_test(cls, result: pd.DataFrame, *, model: Any = None, denominators: Optional[pd.Series] = None,
                  denominator_kind: Optional[str] = None, excluded: Optional[pd.DataFrame] = None
                  ) -> "ProviderProfile":
        """Build a profile from a ``test()`` result.

        Parameters
        ----------
        result : pandas.DataFrame
            A frame with the columns :data:`~pprof_py.inference.PROVIDER_TEST_COLUMNS`; its ``attrs`` become the
            provenance.
        model : optional
            The fitted model that produced ``result``; supplies counts, denominators, degeneracy and exclusions.
        denominators : pandas.Series, optional
            Denominators indexed by provider (overrides the model's), with ``denominator_kind`` naming them.
        excluded : pandas.DataFrame, optional
            Providers removed before the test (default: the model's ``excluded_providers_``).
        """
        missing = [c for c in PROVIDER_TEST_COLUMNS if c not in result.columns]
        if missing:
            raise ValueError(f"result lacks the provider-test columns {missing}; pass a test() result, or use "
                             "ProviderProfile.from_frame() with a role map")
        return cls._build(result, model=model, denominators=denominators, denominator_kind=denominator_kind,
                          excluded=excluded, source="test")

    @classmethod
    def from_frame(cls, df: pd.DataFrame, *, roles: Mapping[str, str], provenance: Optional[Mapping[str, Any]] = None,
                   excluded: Optional[pd.DataFrame] = None) -> "ProviderProfile":
        """Build a profile from any frame, mapping profile roles to its columns.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per provider.
        roles : mapping
            Profile column -> ``df`` column. ``estimate`` and ``flag`` are required; ``provider_id`` names the
            identifier column (default: the index). Any of the other :data:`PROFILE_COLUMNS` except ``status`` and
            ``has_interval`` may be mapped.
        provenance : mapping, optional
            Settings to show with the display (``level``, ``reference``, ``null_model``, ``test_method``,
            ``estimator``, ...); missing essentials are recorded as ``None`` and shown as not stated.

        Warns
        -----
        UserWarning
            When intervals and flags disagree (S3) or funnel limits and flags disagree (S4); the providers are
            listed in the provenance (``s3_violations``, ``s4_violations``).
        """
        allowed = set(PROFILE_COLUMNS) - {"status", "has_interval"} | {"provider_id"}
        unknown = sorted(set(roles) - allowed)
        if unknown:
            raise ValueError(f"unknown roles {unknown}; allowed: {sorted(allowed)}")
        for role in ("estimate", "flag"):
            if role not in roles:
                raise ValueError(f"roles must map {role!r}")
        absent = sorted({c for c in roles.values()} - set(df.columns))
        if absent:
            raise ValueError(f"columns {absent} are not in the frame")
        index = pd.Index(df[roles["provider_id"]].to_numpy() if "provider_id" in roles else df.index, name="provider_id")
        cols: Dict[str, Any] = {}
        for c in PROFILE_COLUMNS:
            src = None if c not in roles else df[roles[c]].to_numpy()
            if c == "flag":
                cols[c] = _flag_array(src, index)
            elif c == "null_group":
                cols[c] = pd.array([pd.NA] * len(index), dtype="Int64") if src is None else pd.Series(src).to_numpy()
            elif c in _ATTRIBUTE_COLUMNS:
                cols[c] = _boolean(None if src is None else pd.Series(src, index=index), index)
            elif c not in ("status", "has_interval"):
                try:
                    cols[c] = _float(None if src is None else pd.Series(src, index=index), index)
                except (TypeError, ValueError):
                    raise ValueError(f"role {c!r} (column {roles[c]!r}) must be numeric") from None
        prov = {k: None for k in _ESSENTIALS}
        prov.update(dict(provenance or {}))
        prov.update({"source": "frame", "n_providers": len(index)})
        prov.setdefault("scale", None)
        frame = cls._assemble(index, cols)
        s3 = _s3_violations(frame, prov.get("alternative"))
        s4 = _s4_violations(frame) if "funnel_estimate" in roles else []
        _warn_violations("S3", s3)
        _warn_violations("S4", s4)
        prov["s3_violations"], prov["s4_violations"] = tuple(s3), tuple(s4)
        return cls(frame, prov, excluded=excluded)

    # ----------------------------------------------------------------------------------------- internals
    @staticmethod
    def _assemble(index: pd.Index, cols: Dict[str, Any]) -> pd.DataFrame:
        frame = pd.DataFrame({c: cols[c] for c in PROFILE_COLUMNS if c not in ("status", "has_interval")}, index=index)
        frame["status"] = _status(pd.array(frame["flag"]))
        lo, hi = frame["ci_lower"].to_numpy(dtype=np.float64), frame["ci_upper"].to_numpy(dtype=np.float64)
        frame["has_interval"] = ~(np.isnan(lo) & np.isnan(hi))
        return frame[list(PROFILE_COLUMNS)]

    @classmethod
    def _build(cls, res: pd.DataFrame, *, model: Any = None, funnel: Any = None, settings: Optional[dict] = None,
               denominators: Optional[pd.Series] = None, denominator_kind: Optional[str] = None,
               excluded: Optional[pd.DataFrame] = None, source: str) -> "ProviderProfile":
        index = pd.Index(res.index, name="provider_id")
        cols: Dict[str, Any] = {c: res[c].to_numpy(dtype=np.float64) for c in TEST_COLUMNS
                                if c not in ("flag", "null_group")}
        cols["flag"] = pd.array(res["flag"].array, dtype="Int64")
        cols["null_group"] = res["null_group"].array
        counts = _model_counts(model, index) if model is not None else {}
        kind = counts.get("denominator_kind")
        for c in ("observed", "expected", "person_time"):
            cols[c] = _float(res[c] if c in res.columns else counts.get(c), index)
        if "expected" in res.columns:                   # CoxPH: E governs the precision of O/E
            cols["denominator"], kind = _float(res["expected"], index), "expected"
        else:
            cols["denominator"] = _float(counts.get("denominator"), index)
        if denominators is not None:
            cols["denominator"], kind = _float(denominators, index), denominator_kind
        if funnel is not None:
            p = funnel.providers
            for c, src in zip(_FUNNEL_COLUMNS, ("estimate", "precision", "lower", "upper")):
                cols[c] = p[src].to_numpy(dtype=np.float64)
            for c in ("observed", "expected"):
                if np.isfinite(p[c].to_numpy(dtype=np.float64)).any():
                    cols[c] = p[c].to_numpy(dtype=np.float64)
        else:
            for c in _FUNNEL_COLUMNS:
                cols[c] = np.full(len(index), np.nan)
        if "observed" in res.columns:                   # CoxPH: O/E is finite even with no events
            zero = res["observed"].to_numpy(dtype=np.float64) == 0
            counts.update({"zero_events": zero, "all_events": False, "finite_estimate": True})
        for c in _ATTRIBUTE_COLUMNS:
            v = counts.get(c)
            cols[c] = _boolean(v, index)
        a = res.attrs
        name = None if model is None else type(model).__name__
        prov: Dict[str, Any] = {
            "source": source, "model": name, "package_version": _version(),
            "estimator": _ESTIMATORS.get(name) if name else None,
            "measure": a.get("measure"), "transform": a.get("transform"), "scale": _scale(a.get("measure"), name),
            "covariates": None if model is None else _covariates(model),
            "test_method": a.get("test_method"),
            "reference": a.get("reference") if (settings or {}).get("reference") is None else settings["reference"],
            "reference_value": a.get("reference"), "null_model": a.get("null_model"),
            "alternative": a.get("alternative"), "level": a.get("level"), "critical": a.get("critical"),
            "interval": a.get("interval"), "limits": a.get("limits"), "denominator_kind": kind,
            "providers": (settings or {}).get("providers"), "seed": (settings or {}).get("seed"),
            "n_providers": len(index),
        }
        null_values = res["null_value"].to_numpy(dtype=np.float64)
        prov["null_value"] = float(null_values[0]) if len(null_values) and np.all(null_values == null_values[0]) else None
        if funnel is not None:
            fa = funnel.attrs
            prov["funnel"] = {k: fa.get(k) for k in ("estimate_kind", "precision_kind", "limit_rule", "curve_kind",
                                                    "levels")}
        if excluded is None and model is not None:
            excluded = getattr(model, "excluded_providers_", None)
        prov["excluded"] = None if excluded is None else len(excluded)
        frame = cls._assemble(index, cols)
        s3 = _s3_violations(frame, prov["alternative"])
        _warn_violations("S3", s3)
        prov["s3_violations"], prov["s4_violations"] = tuple(s3), ()
        return cls(frame, prov, excluded=excluded, curves=None if funnel is None else funnel.curves)


def _covariates(model: Any) -> Optional[tuple]:
    names = getattr(model, "covariate_names_", None)
    return None if names is None else tuple(str(n) for n in names)


def _version() -> Optional[str]:
    from ... import __version__
    return __version__
