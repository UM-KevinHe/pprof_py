"""Provider summary table: the profile's numbers with denominators, intervals, flags and generated footnotes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .._provenance import counts_text, interval_method, null_text, pct, reference_text, test_text
from ..data import CapabilityError
from ..data._resolve import resolve_profile
from ..formatting import FLAG_SYMBOLS, MISSING, fmt_count, fmt_flag, fmt_interval, fmt_number, fmt_p, resolve_digits
from ._spec import Column, TableSpec, excel_bytes, render_html, render_latex, render_markdown, render_text

__all__ = ["TableResult", "provider_table"]

_KEYS = ("provider", "n", "observed", "expected", "estimate_ci", "flag", "p_value")
_NEEDS = {"n": "denominator", "observed": "observed", "expected": "expected"}
_DENOMINATORS = {"records": "N", "trials": "Trials", "expected": "Expected events", "patients": "Patients",
                 "person_time": "Person-time"}
_PHRASES = {"records": "records", "trials": "trials", "expected": "expected events", "patients": "patients",
            "person_time": "person-time"}
_MEASURES = {"indirect_ratio": "Standardized ratio (O/E)", "indirect_rate": "Standardized rate",
             "direct_ratio": "Directly standardized ratio", "direct_rate": "Directly standardized rate"}
_EFFECTS = {"log_odds": "Provider effect (log-odds)", "difference": "Provider effect (difference)"}


class TableResult:
    """A rendered-on-demand table: HTML, Markdown, LaTeX, plain text, Excel and a tidy DataFrame.

    Every output is generated from one :class:`~pprof_py.presentation.tables.TableSpec`, so all formats show the
    same numbers, symbols and footnotes. Outputs are deterministic.
    """

    __slots__ = ("_spec", "_theme", "_intervals")

    def __init__(self, spec: TableSpec, *, theme: Any = None, intervals: Optional[pd.DataFrame] = None) -> None:
        from ..theme import get_theme
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_theme", get_theme(theme))
        object.__setattr__(self, "_intervals", intervals)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("TableResult is immutable")

    spec = property(lambda self: self._spec)
    theme = property(lambda self: self._theme, doc="The theme that styles the HTML output.")
    provenance = property(lambda self: self._spec.provenance)

    def to_html(self, *, standalone: bool = True) -> str:
        """HTML styled by the table's theme (fonts are named, never embedded), with the inline interval column when
        the table was made with ``intervals=True``; a classic theme gives the 0.6.0 HTML."""
        return render_html(self._spec, standalone=standalone, theme=self._theme, intervals=self._intervals)

    def to_markdown(self) -> str:
        return render_markdown(self._spec)

    def to_latex(self) -> str:
        return render_latex(self._spec)

    def to_text(self) -> str:
        return render_text(self._spec)

    def to_excel(self, path: Union[str, Path, None] = None) -> Union[bytes, Path]:
        """The table as an .xlsx workbook (needs ``pprof_py[excel]``): bytes, or the written path."""
        data = excel_bytes(self._spec)
        if path is None:
            return data
        p = Path(path)
        p.write_bytes(data)
        return p

    def to_frame(self) -> pd.DataFrame:
        """The underlying numbers and statuses (suppressed values removed), with the provenance in ``attrs``."""
        out = self._spec.values.copy()
        out.attrs = dict(self._spec.provenance)
        return out

    def _save(self, path: Union[str, Path], text: str) -> Path:
        p = Path(path)
        p.write_text(text, encoding="utf-8")
        return p

    def save_html(self, path: Union[str, Path], *, standalone: bool = True) -> Path:
        return self._save(path, self.to_html(standalone=standalone))

    def save_markdown(self, path: Union[str, Path]) -> Path:
        return self._save(path, self.to_markdown())

    def save_latex(self, path: Union[str, Path]) -> Path:
        return self._save(path, self.to_latex())

    def save_text(self, path: Union[str, Path]) -> Path:
        return self._save(path, self.to_text())

    def _repr_html_(self) -> str:
        return self.to_html(standalone=False)

    def __repr__(self) -> str:
        return self.to_text()


_GROUP_ORDER = ("above", "not_different", "below", "not_tested", "suppressed")
_GROUP_LABELS = {"above": "Above reference", "not_different": "Not different", "below": "Below reference",
                 "not_tested": "Not tested", "suppressed": "Suppressed"}


def _estimate_label(prov: Any) -> str:
    if prov.get("estimator") == "observed/expected ratio":
        return "O/E ratio"
    return _MEASURES.get(prov.get("measure")) or _EFFECTS.get(prov.get("scale"), "Estimate")


def _strings(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=object).ravel()


def provider_table(source: Any, *args: Any, columns: Optional[Iterable[str]] = None, digits: int = 2,
                   p_values: bool = False, min_volume: Optional[float] = None, caption: Optional[str] = None,
                   group_by: Optional[str] = None, intervals: bool = False, theme: Any = "publication",
                   **test_kwargs: Any) -> TableResult:
    """Provider results with denominators, intervals and flags, and footnotes generated from the test's settings.

    Parameters
    ----------
    source
        A fitted model (tested once here with ``test()``'s defaults), a ``test()`` result, or a
        :class:`~pprof_py.presentation.ProviderProfile`.
    *args, **test_kwargs
        Passed to the model's ``test()``. Not allowed with a profile or a ``test()`` result.
    columns : sequence of str, optional
        From ``"provider"``, ``"n"`` (the denominator), ``"observed"``, ``"expected"``, ``"estimate_ci"``,
        ``"flag"``, ``"p_value"``. Default: every available column except ``"p_value"``.
    digits : int, default 2
        Decimals of estimates and intervals; a flagged row gets more where rounding would make its interval touch
        the reference (S12), and a footnote says so.
    p_values : bool, default False
        Add the p-value column to the default columns.
    min_volume : float, optional
        Suppress providers whose denominator is below this (shown as ``S``; their numbers are also removed from
        :meth:`TableResult.to_frame` and Excel output).
    caption : str, optional
        Table caption (default: generated from the measure).
    group_by : {None, "status"}
        ``"status"`` groups the rows by the test's result (above, not different, below, not tested, suppressed),
        each group under a header row with its count; rows keep the provider order within a group. Every format
        shows the groups; Excel and :meth:`TableResult.to_frame` keep tidy rows, in the grouped order.
    intervals : bool, default False
        Add an inline interval column to the HTML output: each provider's interval on one shared scale, with the
        reference and the estimate's mark. Other formats are unchanged.
    theme : str or Theme
        Styles the HTML output (``"classic"`` gives the 0.6.0 HTML); the other formats do not depend on it.

    Returns
    -------
    TableResult

    Raises
    ------
    CapabilityError
        When a requested column needs a quantity the source does not provide.
    """
    if group_by not in (None, "status"):
        raise ValueError(f"group_by must be None or 'status', got {group_by!r}")
    prof = resolve_profile(source, args, test_kwargs, display="provider_table", limits=False)
    if min_volume is not None:
        prof = prof.with_min_volume(min_volume)
    f, prov = prof.data, prof.provenance
    caps = prof.capabilities
    kind = prov.get("denominator_kind")
    ratio = prov.get("scale") == "ratio"
    level = float(prov.get("level") or 0.95)
    available = ["provider"] + [k for k in ("n", "observed", "expected") if _NEEDS[k] in caps
                                and not (k == "expected" and kind == "expected")] + ["estimate_ci", "flag", "p_value"]
    wanted = list(columns) if columns is not None else [k for k in available if k != "p_value"] + (
        ["p_value"] if p_values else [])
    unknown = [k for k in wanted if k not in _KEYS]
    if unknown:
        raise ValueError(f"unknown columns {unknown}; choose from {list(_KEYS)}")
    for k in wanted:
        if k not in available:
            prof.require(f"provider_table column {k!r}", _NEEDS.get(k, k))
            raise CapabilityError(f"provider_table column {k!r} repeats the denominator, which is the expected count")

    est, lo, hi = (f[c].to_numpy(dtype=float) for c in ("estimate", "ci_lower", "ci_upper"))
    nv = f["null_value"].to_numpy(dtype=float)
    flag = f["flag"]
    status = f["status"].astype(str).to_numpy()
    suppressed = status == "suppressed"
    nofinite = ~f["finite_estimate"].fillna(True).to_numpy(dtype=bool) & (not ratio)
    no_bounds = np.isnan(lo) | np.isnan(hi)
    flag_f = flag.to_numpy(dtype=float, na_value=np.nan)
    row_digits, unresolved = resolve_digits(lo, hi, nv, list(np.where(no_bounds, np.nan, flag_f)), digits)
    ci = _strings(fmt_interval(est, lo, hi, row_digits))
    ci = np.where(nofinite, MISSING["no_finite_estimate"], ci)
    den = f["denominator"].to_numpy(dtype=float)
    cells = {
        "provider": [str(i) for i in f.index],
        "n": _strings(fmt_number(den, 1) if kind == "expected" else fmt_count(den)),
        "observed": _strings(fmt_count(f["observed"].to_numpy(dtype=float))),
        "expected": _strings(fmt_number(f["expected"].to_numpy(dtype=float), 1)),
        "estimate_ci": np.where(suppressed, MISSING["suppressed"], ci),
        "flag": np.where(suppressed, MISSING["suppressed"], _strings(fmt_flag(flag))),
        "p_value": np.where(suppressed, MISSING["suppressed"], _strings(fmt_p(f["p_value"].to_numpy(dtype=float)))),
    }
    table = pd.DataFrame({k: cells[k] for k in wanted}, index=f.index)

    label = _estimate_label(prov)
    headers = {"provider": ("Provider", "id", ""), "n": (_DENOMINATORS.get(kind, "N"), "count", ""),
               "observed": ("Observed", "count", ""), "expected": ("Expected", "estimate", ""),
               "estimate_ci": (f"{label} ({pct(level)} CI)", "interval", "a"), "flag": ("Flag", "flag", "b"),
               "p_value": ("p-value", "p_value", "")}
    cols = tuple(Column(k, headers[k][0], headers[k][1], marker=headers[k][2]) for k in wanted)
    notes = _notes(prov, wanted, table, nofinite & ~suppressed, suppressed, row_digits > digits, unresolved, level,
                   ratio, kind, min_volume)
    values = _values(f, suppressed, kind)
    iv = (pd.DataFrame({"estimate": est, "ci_lower": lo, "ci_upper": hi, "null_value": nv, "status": status},
                       index=f.index) if intervals else None)
    groups: Tuple[Tuple[str, int], ...] = ()
    if group_by == "status":                     # by the test's result; provider order within a group (D40)
        rank = {k: i for i, k in enumerate(_GROUP_ORDER)}
        order = np.argsort(np.array([rank.get(v, len(rank)) for v in status]), kind="stable")
        table, values = table.iloc[order], values.iloc[order]
        iv = iv.iloc[order] if iv is not None else None
        present = [k for k in _GROUP_ORDER if (status == k).any()] + sorted(set(status) - set(_GROUP_ORDER))
        groups = tuple((f"{_GROUP_LABELS.get(k, k)} ({fmt_count(int((status == k).sum()))})", int((status == k).sum()))
                       for k in present)
        notes.append(("", "Rows are grouped by the test's result, in provider order within each group."))
    formats = {"denominator" if kind is None else kind: "0.0" if kind == "expected" else "#,##0", "observed": "#,##0",
               "expected": "0.0", "estimate": "0." + "0" * digits, "ci_lower": "0." + "0" * digits,
               "ci_upper": "0." + "0" * digits, "p_value": "0.000", "flag": "0"}
    source_note = (f"Source: {prov.get('model') or 'user data'}; pprof_py {prov.get('package_version') or ''}".rstrip()
                   + f". {counts_text(prof)}.")
    spec = TableSpec(columns=cols, cells=table, values=values,
                     caption=caption or f"Provider results: {label}, {fmt_count(len(prof))} providers",
                     notes=tuple(notes), source_note=source_note, number_formats=formats, provenance=dict(prov),
                     groups=groups)
    return TableResult(spec, theme=theme, intervals=iv)


def _notes(prov: Any, wanted: List[str], table: pd.DataFrame, ne: np.ndarray, suppressed: np.ndarray,
           extra_digits: np.ndarray, unresolved: np.ndarray, level: float, ratio: bool, kind: Optional[str],
           min_volume: Optional[float]) -> List[Tuple[str, str]]:
    notes: List[Tuple[str, str]] = []
    if "estimate_ci" in wanted:
        text = (f"Estimates: {prov.get('estimator') or 'estimator not stated'}; {pct(level)} {interval_method(prov)}"
                "intervals from the same test as the flags")
        if prov.get("alternative") == "two_sided" and not prov.get("s3_violations"):
            text += "; an interval excludes the reference exactly when the provider is flagged"
        notes.append(("a", text + f". Reference: {reference_text(prov, False)}."))
    if "flag" in wanted:
        notes.append(("b", f"{FLAG_SYMBOLS[1]} above or {FLAG_SYMBOLS[-1]} below the reference; {FLAG_SYMBOLS[0]} not "
                           f"different; {MISSING['not_tested']} not tested. Test: {test_text(prov)}; "
                           f"{null_text(prov.get('null_model'))}."))
    if "estimate_ci" in wanted and ne.any():
        notes.append(("", f"{MISSING['no_finite_estimate']}: no finite estimate (no events or only events); the test "
                          "and its flag still apply."))
    if suppressed.any():
        notes.append(("", f"{MISSING['suppressed']}: suppressed, fewer than {fmt_number(min_volume, 0)} "
                          f"{_PHRASES.get(kind, 'units')}; values and flags are not shown."))
    if "estimate_ci" in wanted and (extra_digits & ~suppressed).any():
        notes.append(("", "Some intervals show extra decimals so that rounding does not blur whether they exclude "
                          "the reference."))
    if "estimate_ci" in wanted and (unresolved & ~suppressed).any():
        notes.append(("", f"For {fmt_count(int((unresolved & ~suppressed).sum()))} provider(s) the interval's distance "
                          "from the reference is below display precision."))
    text = " ".join(map(str, table.to_numpy().ravel()))
    if f"({MISSING['no_interval']})" in text:
        notes.append(("", f"{MISSING['no_interval']}: the test reports no interval."))
    if MISSING["not_applicable"] in text:
        notes.append(("", f"{MISSING['not_applicable']}: not applicable."))
    return notes


def _values(f: pd.DataFrame, suppressed: np.ndarray, kind: Optional[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=f.index)
    out["denominator" if kind is None else kind] = f["denominator"].to_numpy(dtype=float)
    for c in ("observed", "expected", "estimate", "ci_lower", "ci_upper", "p_value"):
        v = f[c].to_numpy(dtype=float).copy()
        if c not in ("observed", "expected"):
            v[suppressed] = np.nan
        out[c] = v
    flag = f["flag"].copy()
    flag[suppressed] = pd.NA
    out["flag"] = flag.astype("Int64").array
    out["status"] = f["status"].astype(str).to_numpy()
    return out
