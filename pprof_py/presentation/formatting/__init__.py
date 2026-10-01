"""Formatting shared by tables, axis labels, tooltips and alt text.

Pure, vectorised functions with explicit precision. A scalar gives a ``str``; a :class:`pandas.Series` gives a
Series with the same index; other array-likes give an object :class:`numpy.ndarray`. Inputs are never modified.

Rules (brief §6.4 and S12): a true minus sign (U+2212) and no negative zero; thousands separators; intervals as
``1.12 (0.95–1.31)``, with "to" when a bound is negative; p-values never shown as 0; non-finite values never shown
as ordinary numbers; one set of missing-value symbols (:data:`MISSING`) shared with figures.
"""
from __future__ import annotations

import math
import textwrap
from types import MappingProxyType
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd

MINUS = "\u2212"
EN_DASH = "\u2013"
INFINITY = "\u221e"
MISSING = MappingProxyType({"not_applicable": "\u2014", "no_finite_estimate": "NE", "not_tested": "NT",
                            "suppressed": "S", "no_interval": "NI"})
"""Symbols for the S6 statuses that replace a number."""
FLAG_SYMBOLS = MappingProxyType({1: "\u25b2", -1: "\u25bc", 0: "\u25cf"})
"""Above, below and not different from the reference; a missing flag is shown as ``MISSING["not_tested"]``."""
_SUPERSCRIPT = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")

__all__ = ["EN_DASH", "FLAG_SYMBOLS", "INFINITY", "MINUS", "MISSING", "fmt_count", "fmt_flag", "fmt_interval",
           "fmt_number", "fmt_p", "fmt_percent", "fmt_ratio", "resolve_digits", "wrap_text"]


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (pd.Series, np.ndarray, list, tuple, pd.Index)) and np.ndim(value) == 0


def _vectorize(func: Callable[..., str], *args: Any) -> Any:
    """Apply ``func`` elementwise (positionally) over the arguments, broadcasting scalars."""
    if all(_is_scalar(a) for a in args):
        return func(*args)
    series = [a for a in args if isinstance(a, pd.Series)]
    for s in series[1:]:
        if not s.index.equals(series[0].index):
            raise ValueError("Series arguments must share the same index")
    columns, n = [], None
    for a in args:
        if _is_scalar(a):
            columns.append(None)
            continue
        values = list(a.array) if isinstance(a, pd.Series) else list(np.asarray(a, dtype=object).ravel())
        if n is not None and len(values) != n:
            raise ValueError("array arguments must have the same length")
        n = len(values)
        columns.append(values)
    out = [func(*[a if c is None else c[i] for a, c in zip(args, columns)]) for i in range(n or 0)]
    if series:
        return pd.Series(out, index=series[0].index, name=series[0].name, dtype=object)
    return np.array(out, dtype=object)


def _check_digits(digits: Any) -> int:
    d = int(digits)
    if d != digits or d < 0:
        raise ValueError(f"digits must be a non-negative integer, got {digits!r}")
    return d


def _clean_sign(text: str) -> str:
    """True minus sign; drop the sign of a value that rounds to zero."""
    if not text.startswith("-"):
        return text
    body = text[1:]
    mantissa = body.split("e")[0]
    if mantissa.replace(",", "").replace(".", "").strip("0") == "":
        return body
    return MINUS + body


def _fixed(v: float, digits: int, grouping: bool) -> str:
    return _clean_sign(f"{v:,.{digits}f}" if grouping else f"{v:.{digits}f}")


def _sig(v: float, digits: int, grouping: bool) -> str:
    text = np.format_float_positional(v, precision=max(digits, 1), unique=False, fractional=False, trim="k")
    text = text[:-1] if text.endswith(".") else text
    if grouping:
        sign = "-" if text.startswith("-") else ""
        whole, _, frac = text.lstrip("-").partition(".")
        text = sign + f"{int(whole):,}" + ("." + frac if frac else "")
    return _clean_sign(text)


def _sci(v: float, digits: int) -> str:
    mantissa, exponent = f"{v:.{digits}e}".split("e")
    return f"{_clean_sign(mantissa)} \u00d7 10{str(int(exponent)).translate(_SUPERSCRIPT)}"


def _number(v: Any, digits: int, kind: str, grouping: bool, missing: str) -> str:
    if _is_missing(v):
        return missing
    v = float(v)
    if math.isinf(v):
        return INFINITY if v > 0 else MINUS + INFINITY
    d = _check_digits(digits)
    if kind == "fixed":
        return _fixed(v, d, grouping)
    if kind == "sig":
        return _sig(v, d, grouping)
    if kind == "sci":
        return _sci(v, d)
    raise ValueError(f"kind must be 'fixed', 'sig' or 'sci', got {kind!r}")


def fmt_number(x: Any, digits: Any = 2, *, kind: str = "fixed", grouping: bool = True,
               missing: str = MISSING["not_applicable"]) -> Any:
    """Format numbers with ``digits`` decimals (``kind="fixed"``), significant digits (``"sig"``) or scientific
    notation (``"sci"``). ``digits`` may be an array (one value per element)."""
    return _vectorize(lambda v, d: _number(v, d, kind, grouping, missing), x, digits)


def fmt_ratio(x: Any, digits: Any = 2, **kwargs: Any) -> Any:
    """Format ratios (two decimals by default)."""
    return fmt_number(x, digits, **kwargs)


def fmt_percent(x: Any, digits: Any = 1, *, scale: float = 100.0, missing: str = MISSING["not_applicable"]) -> Any:
    """Format proportions as percentages (``0.123`` gives ``"12.3%"``)."""
    def one(v: Any, d: Any) -> str:
        return missing if _is_missing(v) else _number(float(v) * scale, d, "fixed", True, missing) + "%"
    return _vectorize(one, x, digits)


def fmt_count(x: Any, *, missing: str = MISSING["not_applicable"]) -> Any:
    """Format whole-number counts with thousands separators; non-integers raise instead of being rounded."""
    def one(v: Any) -> str:
        if _is_missing(v):
            return missing
        f = float(v)
        if not math.isfinite(f) or not f.is_integer():
            raise ValueError(f"fmt_count expects whole numbers, got {v!r}; use fmt_number for other values")
        return _clean_sign(f"{int(f):,}")
    return _vectorize(one, x)


def fmt_interval(estimate: Any, lower: Any, upper: Any, digits: Any = 2, *, kind: str = "fixed",
                 grouping: bool = True, missing: str = MISSING["not_applicable"],
                 no_interval: str = MISSING["no_interval"]) -> Any:
    """Format ``estimate (lower–upper)``.

    "to" replaces the en dash when either bound is negative (``−0.40 (−0.62 to −0.18)``). Both bounds missing gives
    ``estimate (NI)``; a missing estimate gives ``missing``. One missing bound, or ``lower > upper``, raises.
    """
    def one(e: Any, lo: Any, hi: Any, d: Any) -> str:
        if _is_missing(e):
            return missing
        text = _number(e, d, kind, grouping, missing)
        lo_missing, hi_missing = _is_missing(lo), _is_missing(hi)
        if lo_missing and hi_missing:
            return f"{text} ({no_interval})"
        if lo_missing or hi_missing:
            raise ValueError("an interval needs both bounds; use NaN for both when there is no interval")
        lo_f, hi_f = float(lo), float(hi)
        if lo_f > hi_f:
            raise ValueError(f"lower bound {lo_f} exceeds upper bound {hi_f}")
        sep = " to " if (lo_f < 0 or hi_f < 0) else EN_DASH
        return f"{text} ({_number(lo_f, d, kind, grouping, missing)}{sep}{_number(hi_f, d, kind, grouping, missing)})"
    return _vectorize(one, estimate, lower, upper, digits)


def fmt_p(p: Any, *, digits: int = 3, threshold: float = 0.001, sci: bool = False,
          missing: str = MISSING["not_applicable"]) -> Any:
    """Format p-values: ``digits`` decimals, ``"<0.001"`` below ``threshold`` (scientific notation instead when
    ``sci=True``), never ``0``. Values outside [0, 1] raise."""
    d = _check_digits(digits)
    shown = max(d, max(0, -math.floor(math.log10(threshold))))

    def one(v: Any) -> str:
        if _is_missing(v):
            return missing
        v = float(v)
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"p-values lie in [0, 1], got {v!r}")
        if v < threshold:
            return _sci(v, 1) if (sci and v > 0) else "<" + f"{threshold:.{shown}f}"
        text = f"{v:.{d}f}"
        return "<" + f"{10.0 ** -d:.{d}f}" if float(text) == 0.0 else text
    return _vectorize(one, p)


def fmt_flag(flag: Any, *, not_tested: str = MISSING["not_tested"]) -> Any:
    """Symbols for test flags: ``1`` above (▲), ``-1`` below (▼), ``0`` not different (●), missing not tested."""
    def one(v: Any) -> str:
        if _is_missing(v):
            return not_tested
        f = float(v)
        if f not in (1.0, -1.0, 0.0):
            raise ValueError(f"flags are 1, -1, 0 or missing, got {v!r}")
        return FLAG_SYMBOLS[int(f)]
    return _vectorize(one, flag)


def _shown(x: float, digits: int) -> float:
    return x if not math.isfinite(x) else float(f"{x:.{digits}f}")


def resolve_digits(lower: Any, upper: Any, null_value: Any, flag: Any, digits: int = 2, *,
                   max_digits: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Digits per row so that displayed intervals agree with their flags (rounding-collision rule, S12).

    A flagged row (flag ±1) whose rounded interval would touch the rounded reference gets extra digits until it
    excludes it, up to ``max_digits``. Returns ``(digits, unresolved)``; ``unresolved`` marks rows that still
    collide (or whose interval does not exclude the reference at all) and need a footnote.
    """
    d0, cap = _check_digits(digits), _check_digits(max_digits)
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    nv = np.broadcast_to(np.asarray(null_value, dtype=float), lo.shape)
    fl = pd.array(list(flag) if not _is_scalar(flag) else [flag] * lo.size, dtype="Float64").to_numpy(
        dtype=float, na_value=np.nan)
    out = np.full(lo.shape, d0, dtype=int)
    unresolved = np.zeros(lo.shape, dtype=bool)
    for i in np.flatnonzero(np.isin(fl, (1.0, -1.0))):
        k = d0
        while not (_shown(lo[i], k) > _shown(nv[i], k) or _shown(hi[i], k) < _shown(nv[i], k)):
            if k >= cap:
                unresolved[i] = True
                break
            k += 1
        out[i] = k
    return out, unresolved


def wrap_text(text: str, width_mm: float, font_pt: float, *, char_em: float = 0.55) -> str:
    """Wrap ``text`` to lines that fit ``width_mm`` at ``font_pt`` (average glyph width ``char_em`` em)."""
    width = max(10, int(width_mm / (char_em * font_pt * 25.4 / 72.0)))
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)
