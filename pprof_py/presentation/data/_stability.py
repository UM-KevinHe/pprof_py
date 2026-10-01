"""Flag stability: the flags of one model under several test settings (S9: every flag comes from the test layer;
this module only runs the tests and lines up their flags)."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .._provenance import null_text, reference_text, test_text
from ._profile import CapabilityError, ProviderProfile

__all__ = ["FlagScenarios", "changing", "flag_scenarios", "stability_summary"]


@dataclass(frozen=True)
class FlagScenarios:
    """``flags``: nullable integer flags, providers by scenario (the first column is the base); ``estimate``: the base
    test's estimates; ``descriptions``: what each scenario tested."""

    flags: pd.DataFrame
    estimate: pd.Series
    descriptions: Dict[str, str]
    model: str


def _default_reference(model: Any) -> Any:
    """The reference a test uses when none is passed (``test()`` records only its value, not "median" or "mean")."""
    try:
        param = inspect.signature(model.test).parameters.get("reference")
    except (TypeError, ValueError):
        return None
    return None if param is None or param.default is inspect.Parameter.empty else param.default


def _describe(res: pd.DataFrame, spec: Any) -> str:
    prov = dict(ProviderProfile.from_test(res).provenance)
    if spec is not None:
        prov["reference"] = spec
    parts = [test_text(prov), null_text(prov.get("null_model"))]
    ref = reference_text(prov, False)
    if ref and ref != "not stated":
        parts.append(f"reference: {ref}")
    return "; ".join(p for p in parts if p)


def _accepts(model: Any, name: str) -> bool:
    try:
        params = inspect.signature(model.test).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _defaults(model: Any, base: pd.DataFrame, spec: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    attrs = base.attrs
    if _accepts(model, "reference") and attrs.get("reference") is not None:
        other = "median" if spec == "mean" else "mean"
        out[f"Reference: {other}"] = {"reference": other}
    if _accepts(model, "null_model"):
        from ...inference import EmpiricalNull

        kind = str((attrs.get("null_model") or {}).get("kind", "theoretical")) if isinstance(
            attrs.get("null_model"), Mapping) else "theoretical"
        if kind == "theoretical":
            out["Empirical null"] = {"null_model": EmpiricalNull.fitter()}
        else:
            out["Theoretical null"] = {"null_model": None}
    return out


def flag_scenarios(model: Any, args: tuple = (), base: Optional[Mapping[str, Any]] = None,
                   scenarios: Optional[Mapping[str, Mapping[str, Any]]] = None) -> FlagScenarios:
    """Run ``model.test(*args, **base)`` and one test per scenario (``base`` updated by the scenario's settings).

    Without ``scenarios``: an alternative reference (where the test takes one), the other null (theoretical or
    empirical), and, for models with ``sigma_sensitivity()``, the flags at the lower and upper bound of sigma's
    interval.
    """
    if not hasattr(model, "test"):
        raise CapabilityError(f"flag stability needs a fitted model with test(); got {type(model).__name__}")
    base = dict(base or {})
    first = model.test(*args, **base)
    runs = {"Base": first}
    default_ref = _default_reference(model)
    specs = {"Base": base.get("reference", default_ref)}
    chosen = (_defaults(model, first, specs["Base"]) if scenarios is None
              else {str(k): dict(v) for k, v in scenarios.items()})
    for label, settings in chosen.items():
        merged = {**base, **settings}
        runs[label] = model.test(*args, **merged)
        specs[label] = merged.get("reference", default_ref)
    flags = pd.DataFrame({label: res["flag"].reindex(first.index) for label, res in runs.items()})
    descriptions = {label: _describe(res, specs[label]) for label, res in runs.items()}
    if scenarios is None and hasattr(model, "sigma_sensitivity"):
        sens = model.sigma_sensitivity(**{k: v for k, v in base.items() if k != "providers"})
        sigma = sens["sigma"]
        for col, label in (("lower", "\u03c3 at lower bound"), ("upper", "\u03c3 at upper bound")):
            flags[label] = sens["flags"][col].reindex(first.index).astype("Int64")
            descriptions[label] = (f"as Base with \u03c3 fixed at the {col} bound of its interval "
                                   f"({float(sigma[col]):.3g}; estimate {float(sigma['estimate']):.3g}), from "
                                   "sigma_sensitivity()")
    flags = flags.astype("Int64")
    return FlagScenarios(flags=flags, estimate=first["estimate"].astype(float), descriptions=descriptions,
                         model=type(model).__name__)


def stability_summary(sc: FlagScenarios) -> pd.DataFrame:
    """Per scenario: providers above, below, not different, not tested, and flags changed relative to the base."""
    f = sc.flags
    base = f.iloc[:, 0]
    rows = {}
    for label in f.columns:
        col = f[label]
        changed = ~((col == base).fillna(False) | (col.isna() & base.isna()))
        rows[label] = {"above": int((col == 1).sum()), "below": int((col == -1).sum()),
                       "not_different": int((col == 0).sum()), "not_tested": int(col.isna().sum()),
                       "changed": int(changed.sum())}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "scenario"
    return out


def changing(sc: FlagScenarios) -> np.ndarray:
    """Providers whose status differs between at least two scenarios (not tested counts as a status)."""
    f = sc.flags.astype("float").fillna(np.inf).to_numpy()
    return ~(f == f[:, :1]).all(axis=1)
