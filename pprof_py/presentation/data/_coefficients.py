"""Covariate effects ready for display: per-term estimates and intervals from ``summary()`` (spec §2.4)."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

from ._profile import CapabilityError

__all__ = ["COEFFICIENT_COLUMNS", "CoefficientProfile", "coefficient_profile"]

COEFFICIENT_COLUMNS = ("estimate", "se", "ci_lower", "ci_upper", "p_value")
_MAPS = {
    "standard": {"estimate": "estimate", "se": "std_error", "ci_lower": "ci_lower", "ci_upper": "ci_upper",
                 "p_value": "p_value"},
    "lme4": {"estimate": "Estimate", "se": "Std.Error", "ci_lower": "ci_lower", "ci_upper": "ci_upper",
             "p_value": "Pr(>|z|)"},
    "cox": {"estimate": "coef", "se": "se(coef)", "ci_lower": "lower_95%", "ci_upper": "upper_95%", "p_value": "p"},
}
_RATIO_SCALES = {"log_odds": "odds_ratio", "log_hazard": "hazard_ratio"}
_INTERCEPT = "(Intercept)"


def _scale(name: str) -> str:
    if name == "CoxPH":
        return "log_hazard"
    if "Logistic" in name:
        return "log_odds"
    if name.startswith("Linear"):
        return "difference"
    raise CapabilityError(f"no coefficient display for {name}")


class CoefficientProfile:
    """Per-term covariate effects with intervals and provenance; immutable.

    Build with :meth:`from_model` (from the model's ``summary()``) or :meth:`from_frame`. :meth:`exponentiate`
    gives odds or hazard ratios: the exponentiated estimates and bounds, a presentation-side derivation recorded in
    the provenance (S1).

    Attributes
    ----------
    data : pandas.DataFrame
        A copy of the rows, indexed by ``term`` in model order, with :data:`COEFFICIENT_COLUMNS`.
    provenance : mapping
        ``model``, ``scale`` (``"log_odds"``, ``"difference"``, ``"log_hazard"``, or after exponentiation
        ``"odds_ratio"``, ``"hazard_ratio"``), ``level``, ``null_value``, ``exponentiated``, ``source``.
    """

    __slots__ = ("_frame", "_provenance")

    def __init__(self, frame: pd.DataFrame, provenance: Mapping[str, Any]) -> None:
        if list(frame.columns) != list(COEFFICIENT_COLUMNS):
            raise ValueError("a CoefficientProfile frame must have exactly the columns COEFFICIENT_COLUMNS")
        if len(frame) == 0:
            raise ValueError("a CoefficientProfile needs at least one term")
        if not frame.index.is_unique:
            raise ValueError("terms must be unique")
        frame = frame.astype(float)
        frame.index = pd.Index([str(t) for t in frame.index], name="term")
        object.__setattr__(self, "_frame", frame)
        object.__setattr__(self, "_provenance", MappingProxyType(dict(provenance)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CoefficientProfile is immutable; build a new profile instead")

    @property
    def data(self) -> pd.DataFrame:
        return self._frame.copy()

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        p = self._provenance
        return f"CoefficientProfile({len(self)} terms; {p.get('model') or 'frame'}, scale={p.get('scale')!r})"

    @classmethod
    def from_model(cls, model: Any, *, level: float = 0.95, include_intercept: bool = False) -> "CoefficientProfile":
        """Covariate effects from ``model.summary(level=level)``; the intercept is left out unless requested.

        Raises
        ------
        CapabilityError
            For a CoxPH model at a level other than 0.95 (its ``summary()`` reports 95% intervals only), or a model
            without a coefficient summary.
        """
        name = type(model).__name__
        scale = _scale(name)
        if name == "CoxPH":
            if float(level) != 0.95:
                raise CapabilityError("CoxPH.summary() reports 95% intervals only; use level=0.95")
            summary, kind = model.summary(), "cox"
        else:
            summary = model.summary(level=level)
            kind = "lme4" if "Estimate" in summary.columns else "standard"
        cols = _MAPS[kind]
        frame = pd.DataFrame({role: pd.to_numeric(summary[col]).to_numpy(dtype=float) for role, col in cols.items()},
                             index=summary.index)[list(COEFFICIENT_COLUMNS)]
        if not include_intercept:
            frame = frame.loc[[t for t in frame.index if t != _INTERCEPT]]
        from ... import __version__
        prov = {"source": "model", "model": name, "scale": scale, "level": float(level), "null_value": 0.0,
                "exponentiated": False, "summary_columns": dict(cols), "package_version": __version__}
        return cls(frame, prov)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, *, roles: Mapping[str, str],
                   provenance: Optional[Mapping[str, Any]] = None) -> "CoefficientProfile":
        """Covariate effects from any frame: ``roles`` maps ``term`` (default: the index) and the columns."""
        missing = [r for r in ("estimate", "ci_lower", "ci_upper") if r not in roles]
        if missing:
            raise ValueError(f"roles must map {missing}")
        index = df[roles["term"]].to_numpy() if "term" in roles else df.index
        frame = pd.DataFrame({c: (pd.to_numeric(df[roles[c]]).to_numpy(dtype=float) if c in roles
                                  else np.full(len(df), np.nan)) for c in COEFFICIENT_COLUMNS}, index=index)
        prov = {"source": "frame", "model": None, "scale": None, "level": None, "null_value": 0.0,
                "exponentiated": False}
        prov.update(dict(provenance or {}))
        return cls(frame, prov)

    def exponentiate(self) -> "CoefficientProfile":
        """Odds or hazard ratios: ``exp`` of the estimates and bounds (``se`` and p-values kept on the model scale)."""
        scale = self._provenance.get("scale")
        if scale not in _RATIO_SCALES or self._provenance.get("exponentiated"):
            raise CapabilityError(f"exponentiate() applies to log-odds and log-hazard coefficients, not {scale!r}")
        f = self._frame.copy()
        for c in ("estimate", "ci_lower", "ci_upper"):
            f[c] = np.exp(f[c].to_numpy())
        prov = dict(self._provenance)
        prov.update({"scale": _RATIO_SCALES[scale], "exponentiated": True, "null_value": 1.0})
        return CoefficientProfile(f, prov)


COEFFICIENT_LABELS = {"odds_ratio": "Odds ratio", "hazard_ratio": "Hazard ratio", "log_odds": "Coefficient (log-odds)",
           "difference": "Coefficient (outcome units)", "log_hazard": "Coefficient (log-hazard)"}
COEFFICIENT_SHORT = {"odds_ratio": "OR", "hazard_ratio": "HR"}


def coefficient_profile(source: Any, *, exponentiate: Union[str, bool] = "auto", level: float = 0.95,
                        include_intercept: bool = False, terms: Optional[Iterable[str]] = None) -> CoefficientProfile:
    """The profile a forest or coefficient table shows: from a model or a profile, subset, exponentiated if asked."""
    prof = source if isinstance(source, CoefficientProfile) else CoefficientProfile.from_model(
        source, level=level, include_intercept=include_intercept)
    if terms is not None:
        wanted = [str(t) for t in terms]
        unknown = [t for t in wanted if t not in prof.data.index]
        if unknown:
            raise ValueError(f"unknown terms {unknown}; available: {list(prof.data.index)}")
        prov = dict(prof.provenance)
        prof = CoefficientProfile(prof.data.loc[wanted], prov)
    ratio = prof.provenance.get("scale") in ("log_odds", "log_hazard")
    if exponentiate == "auto":
        exponentiate = ratio
    if exponentiate and not prof.provenance.get("exponentiated"):
        prof = prof.exponentiate()
    return prof
