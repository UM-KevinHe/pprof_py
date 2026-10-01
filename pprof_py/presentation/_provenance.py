"""Provenance in words: the test, null model, reference, counts and interval method shown with every output (S2)."""
from __future__ import annotations

from typing import Any, Mapping

from .data import ProviderProfile
from .formatting import fmt_count, fmt_number

__all__ = ["counts_text", "interval_method", "null_text", "pct", "reference_text", "test_text"]

_METHODS = {"poibin_exact": "exact Poisson-binomial", "exact": "exact", "score": "score", "wald": "Wald",
            "midp": "mid-p Poisson", "bootstrap_exact": "bootstrap", "resampling": "resampling"}
_ALTERNATIVES = {"two_sided": "two-sided", "greater": "one-sided (above)", "less": "one-sided (below)"}


def pct(level: float) -> str:
    return f"{float(level) * 100:g}%"


def null_text(nm: Any) -> str:
    if not isinstance(nm, Mapping):
        return "null model not stated"
    kind = nm.get("kind")
    if kind == "theoretical":
        return "theoretical null N(0, 1)"
    if kind == "fixed":
        return f"fixed null, mean {fmt_number(nm.get('null_mean'), 2)} and SD {fmt_number(nm.get('null_sd'), 2)}"
    if kind == "empirical":
        groups = list(nm.get("groups") or [])
        if len(groups) == 1:
            g = groups[0]
            return f"empirical null, mean {fmt_number(g['null_mean'], 2)} and SD {fmt_number(g['null_sd'], 2)}"
        if groups:
            m = [g["null_mean"] for g in groups]
            s = [g["null_sd"] for g in groups]
            return (f"empirical null in {len(groups)} groups, means {fmt_number(min(m), 2)} to {fmt_number(max(m), 2)} "
                    f"and SDs {fmt_number(min(s), 2)} to {fmt_number(max(s), 2)}")
        return "empirical null"
    return f"{kind} null"


def test_text(p: Mapping[str, Any]) -> str:
    method = p.get("test_method")
    parts = [f"{_METHODS.get(method, method)} test" if method else "test method not stated"]
    if p.get("alternative"):
        parts.append(_ALTERNATIVES.get(p["alternative"], p["alternative"]))
    if p.get("critical") is not None:
        parts.append(f"critical value {fmt_number(p['critical'], 2)}")
    elif p.get("level") is not None:
        parts.append(f"{pct(p['level'])} level per provider")
    return ", ".join(parts)


def reference_text(p: Mapping[str, Any], ratio: bool) -> str:
    """The reference, in words: a provider effect (and, for measures, the value it implies), or CoxPH's O/E."""
    spec, value = p.get("reference"), p.get("reference_value")
    if p.get("estimator") == "observed/expected ratio":          # CoxPH: no provider effects
        return f"O/E {fmt_number(1.0 if value is None else value, 2)}"
    if value is None:
        return "not stated"
    what = {"median": "median provider effect", "mean": "size-weighted mean provider effect"}.get(spec, "reference effect")
    text = f"{what} {fmt_number(value, 2)}"
    if p.get("measure") in (None, "gamma"):
        return text + (", where O/E = 1" if ratio else "")
    nv = p.get("null_value")
    return text + ("" if nv is None else f", where the measure equals {fmt_number(nv, 2)}")


def counts_text(profile: ProviderProfile) -> str:
    c, p = profile.status_counts(), profile.provenance
    bits = [f"{fmt_count(len(profile))} providers"]
    if c["excluded"]:
        bits.append(f"{fmt_count(c['excluded'])} excluded by data preparation")
    if c["not_tested"]:
        bits.append(f"{fmt_count(c['not_tested'])} not tested")
    if c["suppressed"]:
        bits.append(f"{fmt_count(c['suppressed'])} suppressed (fewer than {fmt_number(p.get('min_volume'), 0)} "
                    f"{p.get('min_volume_kind') or 'units'})")
    return "; ".join(bits)


def interval_method(prov: Any) -> str:
    if prov.get("limits") == "test inversion" or prov.get("interval") == "inversion":
        return "test-inversion "
    if prov.get("interval") == "scale_only":
        return "null-scaled "
    return ""
