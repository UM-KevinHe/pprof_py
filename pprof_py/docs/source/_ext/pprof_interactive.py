"""Sphinx extension: interactive versions of the documentation figures, built from the gallery's fits.

Registered after ``pprof_gallery`` in ``conf.py``. On ``builder-inited`` it takes the gallery's fitted models
(``pprof_gallery.fitted()``) and rendered figures (``pprof_gallery.RESULTS``, for the alt text and caption), so
nothing is fitted twice, and for every display the gallery renders (funnel, interval plot, observed versus expected,
coefficient forest, data quality, null calibration, reliability, between-provider variation, shrinkage, flag
stability, several measures and measure agreement) it extracts one JSON record with exactly the data the static
renderer draws: provider or term ids, estimates, intervals or limits, statuses, counts and volumes, the histogram
bins and curves, the reference values, the axis labels, the legend and the table columns. Each record is embedded in a
self-contained HTML file, ``_static/interactive/<name>.html``, with the CSS and JavaScript from
``_ext/pprof_interactive_assets/`` inlined (plain JavaScript and inline SVG; no dependencies, nothing from the
network). It then rewrites ``presentation/_figures/<name>.md``, the snippet each display page includes directly
under its heading (the gallery writes the static image line there first; this extension replaces it): the
interactive view first, a lazily loaded ``<iframe>`` of that file with a link to open it in a new tab, then the
gallery's static figure (the same image line, rebuilt from the gallery's alt text) in a collapsed ``<details>`` with
the summary "Static figure (SVG, as used in print)". The display pages are unchanged. The snippet starts with the
MyST label ``(interactive-<name>)=``, which the gallery links to (``{ref}`` resolves it to ``#interactive-<name>``;
the same id is set on the block's ``<div>``, because MyST attaches the label to the raw HTML block, which carries no
id in the output). The interactive view is the primary one on the page;
the static SVG stays available and remains the printed version, and the package's presentation layer is untouched
(S13: its own outputs stay static and script-free).

The iframe has a fixed height (``IFRAME_HEIGHT``) as the no-JavaScript fallback. With JavaScript, the interactive
page posts its document height to the embedding page after rendering, on resize and whenever its body changes
size (``postMessage`` to the same origin only), and the few lines of script in the snippet set the iframe's height
from messages whose source is that iframe and whose origin is the page's own, so the frame neither leaves empty
space nor scrolls inside.

The interactive page detects its theme from ``?theme=dark|light`` in its URL, else from the embedding page
(Shibuya's ``data-color-mode`` or a ``data-theme`` attribute, same origin only), else from ``prefers-color-scheme``.
Files are rewritten only when their bytes change, as in the gallery, and values are rounded so repeated builds give
identical bytes.
"""
from __future__ import annotations

import html
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pprof_gallery

ASSETS = Path(__file__).with_name("pprof_interactive_assets")
DISPLAYS = ("funnel", "caterpillar", "observed_expected", "forest", "data_quality", "null_calibration", "reliability",
            "provider_variation", "shrinkage", "flag_stability", "multi_measure", "measure_agreement")
IFRAME_HEIGHT = {"funnel": 1040, "caterpillar": 1300, "observed_expected": 1140, "forest": 520, "data_quality": 1240,
                 "null_calibration": 1080, "reliability": 1300, "provider_variation": 1440, "shrinkage": 1500,
                 "flag_stability": 1900, "multi_measure": 1450, "measure_agreement": 1480}   # px, no-JavaScript fallback
HOVER = {"forest": "Hover for each term's values, sort the table, or read it as a table.",
         "data_quality": "Hover a bar for its definition and share, or a point for the provider; sort the tables, or "
                         "read them as tables.",
         "null_calibration": "Hover a bar for the providers in it, sort the tables, or read them as tables.",
         "provider_variation": "Hover a bar for the providers in it, sort the tables, or read them as tables.",
         "flag_stability": "Hover a row for the provider's flags in every scenario, sort the tables, or read them as "
                           "tables.",
         "multi_measure": "Hover a row for the provider's values in every measure, sort the table, or read it as a "
                          "table."}
DEFAULT_HOVER = "Hover for each provider's values, sort the table, or read it as a table."
STATIC_PREFIX = "../../"              # the display pages live in presentation/displays/, two levels below the root
LEVELS = (0.95, 0.998)                # the gallery's funnel levels (funnel(model) defaults)

STATUS_ORDER = ("above", "below", "not_different", "not_tested")
MARKERS = {"above": "triangle_up", "below": "triangle_down", "not_different": "circle", "not_tested": "circle_open",
           "zero_events": "square_open", "no_finite_estimate": "triangle_left", "excluded": "circle_open"}
SYMBOLS = {"above": "▲", "below": "▼", "not_different": "●", "not_tested": "○",
           "zero_events": "□", "no_finite_estimate": "◁", "excluded": "×"}
LABELS = {"zero_events": "Zero events, O/E = 0", "no_finite_estimate": "No finite estimate, at the axis edge",
          "excluded": "Excluded by data preparation"}
FLAG_STATUS = {1: "above", -1: "below", 0: "not_different"}
PRECISION_LABELS = {"expected": "Expected events, E", "inverse_null_variance": "Precision under the null, E²/V₀",
                    "inverse_variance": "Precision, 1/SE²"}                 # as the funnel renderer labels them
EFFECT_LABELS = {"log_odds": "Provider effect (log-odds)", "difference": "Provider effect (difference)"}
DENOMINATORS = {"records": "Records (N)", "trials": "Trials (N)", "expected": "Expected events (E)",
                "patients": "Patients (N)", "person_time": "Person-time"}

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} (interactive)</title>
<style>
{css}
</style>
</head>
<body>
<main id="root" class="pprof-interactive">
<header>
<h1>{title}</h1>
<p class="question">{question}</p>
</header>
<noscript><p class="note">This interactive version needs JavaScript; the static figure on the documentation page
shows the same data.</p></noscript>
<div id="chart"></div>
<p class="caption">{caption}</p>
</main>
<script type="application/json" id="pprof-data">{data}</script>
<script>
{js}
</script>
</body>
</html>
"""

# The snippet is MyST Markdown: the raw HTML blocks pass through, the Markdown image inside <details> is rendered by
# Sphinx (so the SVG is copied with the page), and the inline <script> is passed through as an HTML block.
# The script sets the iframe's height from the handshake described above; the height attribute is the fallback.
SNIPPET = """({label})=

<div class="pprof-interactive" id="{label}" style="width: 100%; margin: 0 0 1rem;">
<iframe src="{src}" title="{title} (interactive)" loading="lazy" width="100%" height="{height}"
style="display: block; width: 100%; height: {height}px; border: 1px solid rgba(128, 128, 128, 0.35); border-radius: 4px; background: transparent;"></iframe>
<p class="pprof-interactive-note">{hover}
<a href="{src}" target="_blank" rel="noopener">Open the interactive {lower} in a new tab</a>.</p>
</div>
<script>
(function () {{
  if (window.pprofInteractiveHeights) return;             /* one listener per page */
  window.pprofInteractiveHeights = true;
  window.addEventListener('message', function (e) {{
    var d = e.data;
    if (e.origin !== window.location.origin || !d || d.pprof !== 'height') return;   /* same origin only */
    var frames = document.querySelectorAll('.pprof-interactive iframe');
    for (var i = 0; i < frames.length; i++) {{
      if (frames[i].contentWindow !== e.source) continue;                             /* only our own frame */
      var px = Math.ceil(Number(d.height));
      if (px >= 200 && px <= 20000) frames[i].style.height = px + 'px';
    }}
  }});
}})();
</script>

<details class="pprof-static-figure">
<summary>Static figure (SVG, as used in print)</summary>

{static}
</details>
"""


def _num(value: Any, digits: int = 6) -> Optional[float]:
    """A JSON number rounded to ``digits`` decimals, or ``None`` for missing and non-finite values."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    r = round(v, digits)
    return int(r) if r.is_integer() and abs(r) < 1e15 else r


def _flag(value: Any) -> bool:
    try:
        return bool(value) and bool(value == value)      # pandas NA is not equal to itself
    except (TypeError, ValueError):
        return False


def _status_legend(prof: Any, statuses: Dict[str, Dict[str, str]], extra: List[str]) -> List[Dict[str, Any]]:
    """Legend entries for the statuses present, in the renderers' order, with their counts."""
    counts = prof.status_counts()
    entries = [{"kind": "mark", "status": k, "label": f"{statuses[k]['label']} ({counts[k]:,})"}
               for k in STATUS_ORDER if counts.get(k)]
    for k in extra:
        if counts.get(k):
            entries.append({"kind": "mark", "status": k, "label": f"{statuses[k]['label']} ({counts[k]:,})"})
    return entries


def _statuses(theme: Any) -> Dict[str, Dict[str, str]]:
    out = {}
    for key, marker in MARKERS.items():
        label = theme.status[key].label if key in theme.status else LABELS[key]
        out[key] = {"label": label, "marker": marker, "symbol": SYMBOLS[key]}
    return out


def _null_value(prov: Any, f: Any) -> Optional[float]:
    if prov.get("null_value") is not None:
        return float(prov["null_value"])
    nv = f["null_value"].dropna().unique()
    return float(nv[0]) if len(nv) == 1 else None


def _head(name: str, fig: Any, theme: Any) -> Dict[str, Any]:
    title, question, _ = pprof_gallery.ITEMS[name]
    return {"display": name, "title": title, "question": question, "alt": fig.alt_text, "caption": fig.caption,
            "statuses": _statuses(theme), "id_label": "provider"}


def _base(name: str, fig: Any, prof: Any, theme: Any) -> Dict[str, Any]:
    rec = _head(name, fig, theme)
    prov = prof.provenance
    rec.update({"level": float(prov.get("level") or 0.95), "test": prov.get("test_method"), "n_providers": int(len(prof))})
    return rec


def _flag_status(value: Any) -> str:
    """The status key of a test flag (1, -1, 0; NA is not tested)."""
    try:
        if value is None or value != value:
            return "not_tested"
        return FLAG_STATUS[int(value)]
    except (TypeError, ValueError, KeyError):
        return "not_tested"


def _bins(values: Any, edges: Any) -> List[Optional[int]]:
    """The histogram bin of each value (``numpy.histogram`` convention: the last bin is closed), ``None`` outside."""
    import numpy as np

    out: List[Optional[int]] = []
    lo, hi, k = float(edges[0]), float(edges[-1]), len(edges) - 1
    for v in np.asarray(values, dtype=float):
        if not np.isfinite(v) or v < lo or v > hi:
            out.append(None)
        else:
            out.append(min(int(np.searchsorted(edges, v, side="right")) - 1, k - 1))
    return out


def _pad_limits(*arrays: Any, extra: float) -> List[float]:
    """Axis limits as the multi-measure renderers' ``_limits``: the finite values and ``extra``, padded by 6%."""
    import numpy as np

    vals = np.r_[tuple(np.asarray(a, dtype=float)[np.isfinite(np.asarray(a, dtype=float))] for a in arrays) + (extra,)]
    a, b = float(vals.min()), float(vals.max())
    pad = 0.06 * ((b - a) or 1.0)
    return [_num(a - pad), _num(b + pad)]


def funnel_record(fe: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """Provider ratios against precision, the limit curves at each level and the reference, as funnel(model)."""
    import numpy as np
    from pprof_py.presentation._provenance import pct, reference_text
    from pprof_py.presentation.data._resolve import resolve_profile

    prof = resolve_profile(fe, (), {}, display="funnel", limits=True, levels=LEVELS)
    f, prov = prof.data, prof.provenance
    fa = dict(prov.get("funnel") or {})
    ratio = (fa.get("estimate_kind") or ("ratio" if prov.get("scale") == "ratio" else "effect")) == "ratio"
    level = float(prov.get("level") or 0.95)
    providers = []
    for pid, r in f.iterrows():
        status = str(r["status"])
        x, y = _num(r["funnel_precision"]), _num(r["funnel_estimate"])
        if status == "suppressed" or x is None or x <= 0 or y is None:
            continue
        providers.append({"id": str(pid), "status": status, "precision": x, "ratio": y,
                          "lower": _num(r["funnel_lower"]), "upper": _num(r["funnel_upper"]),
                          "observed": _num(r["observed"]), "expected": _num(r["expected"]),
                          "volume": _num(r["denominator"]), "p_value": _num(r["p_value"]),
                          "zero_events": _flag(r["zero_events"]) and ratio})
    curves = []
    c = prof.funnel_curves
    if c is not None and len(c):
        group = c["null_group"].astype(object).where(c["null_group"].notna(), "")
        for lev in sorted(c["level"].unique()):
            for g in sorted(group[c["level"] == lev].unique()):
                cg = c[(c["level"] == lev) & (group == g)].sort_values("precision", kind="mergesort")
                curves.append({"level": float(lev), "own": bool(np.isclose(float(lev), level)), "group": str(g),
                               "precision": [_num(v) for v in cg["precision"]],
                               "lower": [_num(v) for v in cg["lower"]], "upper": [_num(v) for v in cg["upper"]]})
    ref = 1.0 if ratio else _null_value(prov, f)
    rec = _base("funnel", fig, prof, theme)
    others = [pct(v) for v in LEVELS if not np.isclose(v, level)]
    rec["legend"] = _status_legend(prof, rec["statuses"], ["zero_events"]) + [
        {"kind": "line", "style": "own", "label": f"{pct(level)} limits ({fa.get('limit_rule') or 'as supplied'})"},
        {"kind": "line", "style": "other", "label": f"{', '.join(others)} limits, for reference only"},
        {"kind": "band", "label": f"Not flagged at {pct(level)}"},
        {"kind": "line", "style": "reference", "label": "Reference, O/E = 1" if ratio else "Reference"}]
    rec.update({
        "axes": {"x": {"label": PRECISION_LABELS.get(fa.get("precision_kind"), "Precision") + " (log scale)",
                       "scale": "log"},
                 "y": {"label": "Observed / expected (O/E)" if ratio else EFFECT_LABELS.get(prov.get("scale"),
                                                                                            "Estimate")}},
        "reference": {"value": _num(ref), "label": reference_text(prov, ratio)},
        "curves": curves, "providers": providers,
        "columns": [{"key": "id", "label": "Provider"}, {"key": "status", "label": "Status"},
                    {"key": "ratio", "label": "O/E", "digits": 2}, {"key": "observed", "label": "Observed", "digits": 0},
                    {"key": "expected", "label": "Expected", "digits": 1},
                    {"key": "precision", "label": "Precision", "digits": 1},
                    {"key": "lower", "label": f"Lower {pct(level)}", "digits": 2},
                    {"key": "upper", "label": f"Upper {pct(level)}", "digits": 2},
                    {"key": "volume", "label": DENOMINATORS.get(prov.get("denominator_kind"), "Volume"), "digits": 0},
                    {"key": "p_value", "label": "p-value", "digits": 3, "p": True}]})
    return rec


def caterpillar_record(prof: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """Provider estimates with their intervals and volumes, ordered as caterpillar(profile) draws them."""
    import numpy as np
    from pprof_py.presentation._provenance import pct, reference_text

    f, prov = prof.data, prof.provenance
    ratio = prov.get("scale") == "ratio"
    level = float(prov.get("level") or 0.95)
    ref = _null_value(prov, f)
    est = f["estimate"].to_numpy(dtype=float)
    status = f["status"].astype(str).to_numpy()
    finite = f["finite_estimate"].fillna(True).to_numpy(dtype=bool) | ratio
    drawn = (status != "suppressed") & (np.isfinite(est) | ~finite)
    order = np.argsort(np.where(np.isfinite(est), est, np.inf), kind="stable")      # the renderer's order
    providers = []
    for i in order:
        if not drawn[i]:
            continue
        r = f.iloc[i]
        below = (est[i] if np.isfinite(est[i]) else 0.0) < (ref if ref is not None else 0.0)
        providers.append({"id": str(f.index[i]), "status": status[i],
                          "estimate": _num(est[i]) if finite[i] else None,     # never the solver's clamp (ADR-005)
                          "edge": None if finite[i] else ("lower" if below else "upper"),
                          "lower": _num(r["ci_lower"]), "upper": _num(r["ci_upper"]),
                          "has_interval": _flag(r["has_interval"]), "volume": _num(r["denominator"]),
                          "observed": _num(r["observed"]), "expected": _num(r["expected"]),
                          "p_value": _num(r["p_value"]), "zero_events": _flag(r["zero_events"]) and ratio})
    rec = _base("caterpillar", fig, prof, theme)
    kind = prov.get("denominator_kind")
    rec["legend"] = ([{"kind": "line", "style": "interval", "label": f"{pct(level)} interval"}]
                     + _status_legend(prof, rec["statuses"], ["no_finite_estimate", "zero_events"])
                     + [{"kind": "line", "style": "reference", "label": f"Reference: {reference_text(prov, False)}"},
                        {"kind": "bar", "label": DENOMINATORS.get(kind, "Denominator") + " per provider"}])
    rec.update({
        "axes": {"x": {"label": EFFECT_LABELS.get(prov.get("scale"), "Estimate")},
                 "y": {"label": "Providers, ordered by estimate (not a ranking)"},
                 "volume": {"label": DENOMINATORS.get(kind, "Denominator"),
                            "integer": kind in ("records", "trials", "patients")}},
        "reference": {"value": _num(ref), "label": reference_text(prov, False)},
        "providers": providers,
        "columns": [{"key": "id", "label": "Provider"}, {"key": "status", "label": "Status"},
                    {"key": "estimate", "label": "Estimate", "digits": 2},
                    {"key": "lower", "label": f"Lower {pct(level)}", "digits": 2},
                    {"key": "upper", "label": f"Upper {pct(level)}", "digits": 2},
                    {"key": "volume", "label": DENOMINATORS.get(kind, "Volume"), "digits": 0},
                    {"key": "observed", "label": "Observed", "digits": 0},
                    {"key": "expected", "label": "Expected", "digits": 1},
                    {"key": "p_value", "label": "p-value", "digits": 3, "p": True}]})
    return rec


def observed_record(prof: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """Observed against expected events with each provider's limits in events, as observed_expected(profile)."""
    from pprof_py.presentation._provenance import pct

    f, prov = prof.data, prof.provenance
    level = float(prov.get("level") or 0.95)
    providers = []
    for pid, r in f.iterrows():
        status = str(r["status"])
        obs, exp_ = _num(r["observed"]), _num(r["expected"])
        if status == "suppressed" or obs is None or exp_ is None or exp_ <= 0:
            continue
        lo, hi = _num(r["funnel_lower"]), _num(r["funnel_upper"])
        providers.append({"id": str(pid), "status": status, "observed": obs, "expected": exp_,
                          "ratio": _num(r["funnel_estimate"]),
                          "lower": None if lo is None else _num(lo * exp_), "upper": None if hi is None else _num(hi * exp_),
                          "volume": _num(r["denominator"]), "p_value": _num(r["p_value"]),
                          "zero_events": _flag(r["zero_events"])})
    rec = _base("observed_expected", fig, prof, theme)
    rec["legend"] = _status_legend(prof, rec["statuses"], []) + [
        {"kind": "line", "style": "limit", "label": f"{pct(level)} limits of the test, in events"},
        {"kind": "line", "style": "reference", "label": "O = E"},
        {"kind": "line", "style": "guide", "label": "O/E = 0.5 and O/E = 2"}]
    rec.update({
        "axes": {"x": {"label": "Expected events, E (square-root scale)", "scale": "sqrt"},
                 "y": {"label": "Observed events, O (square-root scale)", "scale": "sqrt"}},
        "reference": {"value": 1.0, "label": "O = E"},
        "providers": providers,
        "columns": [{"key": "id", "label": "Provider"}, {"key": "status", "label": "Status"},
                    {"key": "observed", "label": "Observed", "digits": 0},
                    {"key": "expected", "label": "Expected", "digits": 1},
                    {"key": "ratio", "label": "O/E", "digits": 2},
                    {"key": "lower", "label": f"Lower {pct(level)} (events)", "digits": 1},
                    {"key": "upper", "label": f"Upper {pct(level)} (events)", "digits": 1},
                    {"key": "volume", "label": DENOMINATORS.get(prov.get("denominator_kind"), "Volume"), "digits": 0},
                    {"key": "p_value", "label": "p-value", "digits": 3, "p": True}]})
    return rec


def forest_record(fe: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """Covariate effects with their intervals, in model order, as forest(model) draws them."""
    import numpy as np
    from pprof_py.presentation._provenance import pct
    from pprof_py.presentation.data._coefficients import COEFFICIENT_LABELS, COEFFICIENT_SHORT, coefficient_profile
    from pprof_py.presentation.formatting import fmt_interval, fmt_number

    prof = coefficient_profile(fe)
    f, prov = prof.data, prof.provenance
    exp_axis = bool(prov.get("exponentiated"))
    null = float(prov.get("null_value", 1.0 if exp_axis else 0.0))
    lev = float(prov.get("level") or 0.95)
    label = COEFFICIENT_LABELS.get(prov.get("scale"), "Estimate")
    short = COEFFICIENT_SHORT.get(prov.get("scale"), "Estimate")
    texts = list(np.asarray(fmt_interval(f["estimate"], f["ci_lower"], f["ci_upper"], 2), dtype=object))
    terms = []
    for (term, r), text in zip(f.iterrows(), texts):
        lo, hi = float(r["ci_lower"]), float(r["ci_upper"])
        terms.append({"id": str(term), "estimate": _num(r["estimate"]), "lower": _num(lo), "upper": _num(hi),
                      "se": _num(r["se"]), "p_value": _num(r["p_value"]), "text": str(text),
                      "excludes": "yes" if (lo > null or hi < null) else "no"})
    rec = _head("forest", fig, theme)
    rec["id_label"] = "term"
    rec["legend"] = [{"kind": "shape", "shape": "circle", "cls": "mark filled dot", "label": "Estimate"},
                     {"kind": "line", "cls": "interval", "label": f"{pct(lev)} interval"},
                     {"kind": "line", "cls": "reference", "label": f"No association, {short} = {fmt_number(null, 0)}"}]
    rec.update({
        "axes": {"x": {"label": label + (" (log scale)" if exp_axis else "") + f"; no association at {fmt_number(null, 0)}",
                       "scale": "log" if exp_axis else "linear"}},
        "reference": {"value": _num(null), "label": f"{short} = {fmt_number(null, 0)}"},
        "column_head": f"{short} ({pct(lev)} CI)", "terms": terms,
        "table_caption": "Terms shown in the chart: hover or focus a row to find it in the chart; sort by a column.",
        "columns": [{"key": "id", "label": "Covariate", "type": "id"}, {"key": "estimate", "label": short, "digits": 2},
                    {"key": "lower", "label": f"Lower {pct(lev)}", "digits": 2},
                    {"key": "upper", "label": f"Upper {pct(lev)}", "digits": 2},
                    {"key": "se", "label": "SE (model scale)", "digits": 3},
                    {"key": "p_value", "label": "p-value", "digits": 3, "p": True},
                    {"key": "excludes", "label": f"Excludes {fmt_number(null, 0)}", "type": "text"}]})
    return rec


def quality_record(prof: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """Provider accounting and volumes by group, as data_quality(profile) draws them."""
    import numpy as np
    from pprof_py.presentation.data._quality import quality_accounting, quality_groups
    from pprof_py.presentation.formatting import fmt_number
    from pprof_py.presentation.tables._quality import _DEFINITIONS

    f, prov = prof.data, prof.provenance
    kind = prov.get("denominator_kind")
    rows = quality_accounting(prof)
    groups = quality_groups(prof)
    excluded = prof.excluded
    plot_excluded = excluded is not None and len(excluded) > 0 and kind == "records"
    status = f["status"].astype(str).to_numpy()
    nofinite = ~f["finite_estimate"].fillna(True).to_numpy(dtype=bool)
    zero = f["zero_events"].fillna(False).to_numpy(dtype=bool)
    nointerval = ~f["has_interval"].to_numpy(dtype=bool)
    ids = [str(i) for i in f.index]
    ex_ids = [str(i) for i in excluded.index] if excluded is not None else []
    members = {"total": ids + ex_ids, "excluded": ex_ids, "analysed": ids,
               "no_finite_estimate": [i for i, m in zip(ids, nofinite) if m],
               "zero_events": [i for i, m in zip(ids, zero) if m], "no_interval": [i for i, m in zip(ids, nointerval) if m]}
    for key in ("above", "below", "not_different", "not_tested", "suppressed"):
        members[key] = [i for i, s in zip(ids, status) if s == key]
    counts = {key: count for _, count, key in rows}
    total, analysed = counts["total"], counts["analysed"]
    accounting = []
    for label, count, key in rows:
        base = total if key in ("total", "excluded", "analysed") else analysed
        share = None if count is None or not base else _num(count / base, 4)
        accounting.append({"key": key, "label": label, "count": count, "share": share, "definition": _DEFINITIONS[key],
                           "ids": members.get(key, [])})
    order = ["Analysed", "No finite estimate", "Not tested", "Suppressed"]
    names = [g for g in order if (groups == g).any()]
    providers = []
    for i, pid in enumerate(ids):
        issues = [name for name, m in (("suppressed", status[i] == "suppressed"), ("not tested", status[i] == "not_tested"),
                                       ("no finite estimate", nofinite[i]), ("zero events", zero[i]),
                                       ("no interval", nointerval[i])) if m]
        providers.append({"id": pid, "group": str(groups.iloc[i]), "status": status[i], "volume": _num(f["denominator"].iloc[i]),
                          "issues": "; ".join(issues)})
    if plot_excluded:
        names.append("Excluded")
        for pid, r in excluded.iterrows():
            providers.append({"id": str(pid), "group": "Excluded", "status": "excluded", "volume": _num(r["n_records"]),
                              "issues": f"excluded: {r['reason']}"})
    unit = {"records": "records", "trials": "trials", "expected": "expected events", "patients": "patients",
            "person_time": "person-time"}.get(kind, "denominator")
    rec = _head("data_quality", fig, theme)
    rec["legend"] = [{"kind": "mark", "status": k, "label": theme.status[k].label + " (bar)"}
                     for k in ("above", "below", "not_different", "not_tested") if counts.get(k)] + [
        {"kind": "band", "cls": "abar ink", "label": "Providers counted (bar)"},
        {"kind": "band", "cls": "abar attr", "label": "Attribute of analysed providers (bar)"},
        {"kind": "shape", "shape": "circle", "cls": "mark filled dot", "size": 3.5, "label": "Analysed provider (volume)"},
        {"kind": "shape", "shape": "circle", "cls": "mark filled muted-dot", "size": 3.5,
         "label": "Provider in another group (volume)"}]
    if prov.get("min_volume") is not None:
        rec["legend"].append({"kind": "line", "cls": "reference dashed",
                              "label": f"Minimum volume {fmt_number(prov['min_volume'], 0)}"})
    rec.update({
        "axes": {"x": {"label": "Providers"}, "volume": {"label": f"{unit.capitalize()} per provider (log scale)", "scale": "log"}},
        "accounting": accounting, "groups": names, "providers": providers,
        "min_volume": _num(prov.get("min_volume")), "has_volume": "denominator" in prof.capabilities,
        "series": {"caption": "Provider accounting: every provider counted once; shares of the providers in the data "
                              "for the first three rows and of the analysed providers below.",
                   "columns": [{"key": "label", "label": "Providers", "type": "text"},
                               {"key": "count", "label": "Count", "digits": 0, "missing": "not recorded"},
                               {"key": "share", "label": "Share", "percent": True},
                               {"key": "definition", "label": "Definition", "type": "text", "wrap": True}],
                   "rows": [{"label": a["label"], "count": a["count"], "share": a["share"], "definition": a["definition"]}
                            for a in accounting]},
        "table_caption": "Providers in the volume strips: hover or focus a row to find it in the chart; sort by a column.",
        "columns": [{"key": "id", "label": "Provider"}, {"key": "group", "label": "Group", "type": "text"},
                    {"key": "status", "label": "Status"},
                    {"key": "volume", "label": DENOMINATORS.get(kind, "Volume"), "digits": 0},
                    {"key": "issues", "label": "Issues", "type": "text", "wrap": True}]})
    return rec


def calibration_record(fe: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """Raw z-statistics per null group with both nulls and both sets of flags, as null_calibration(model) draws them."""
    import numpy as np
    from pprof_py.inference import EmpiricalNull
    from pprof_py.presentation.data._calibration import calibration_profiles, calibration_summary

    kwargs = {"test_method": "score", "null_model": EmpiricalNull.fitter(size=fe.provider_sizes_, n_groups=3)}
    calibrated, theoretical = calibration_profiles(fe, (), kwargs, display="null_calibration")   # the gallery's call
    summary = calibration_summary(calibrated, theoretical)
    f = calibrated.data
    tested = f["flag"].notna().to_numpy()
    z = f["z_raw"].to_numpy(dtype=float)
    keys = f["null_group"].astype(object).where(f["null_group"].notna(), "all").to_numpy()
    finite = z[tested & np.isfinite(z)]
    med = float(np.median(finite))
    bound = max(6.0, abs(med) + 6.0 * 1.4826 * float(np.median(np.abs(finite - med))))
    lo = max(float(np.floor(finite.min() * 2) / 2), -np.ceil(bound))
    hi = min(float(np.ceil(finite.max() * 2) / 2), np.ceil(bound))
    edges = np.arange(lo, hi + 0.5, 0.5) if (hi - lo) / 0.5 <= 60 else np.linspace(lo, hi, 61)
    fitted_differs = bool(((summary["null_mean"] != 0.0) | (summary["null_sd"] != 1.0)).any())
    compared = theoretical is not None and fitted_differs
    groups = []
    for key in summary.index:
        row = summary.loc[key]
        zg = z[tested & (keys == key) & np.isfinite(z)]
        counts, _ = np.histogram(zg, bins=edges)
        g = {"key": str(key), "label": "All providers" if key == "all" else f"Group {key}", "providers": int(row["providers"]),
             "null_mean": _num(row["null_mean"]), "null_sd": _num(row["null_sd"]),
             "above_fitted": int(row["above_fitted"]), "below_fitted": int(row["below_fitted"]),
             "counts": [int(c) for c in counts], "beyond_left": int((zg < lo).sum()), "beyond_right": int((zg > hi).sum())}
        if theoretical is not None:
            g.update({"above_theoretical": int(row["above_theoretical"]), "below_theoretical": int(row["below_theoretical"]),
                      "changed": int(row["changed"])})
        groups.append(g)
    flag_t = (theoretical.data["flag"].reindex(f.index) if theoretical is not None else None)
    bins = _bins(z, edges)
    providers = []
    for i, pid in enumerate(f.index):
        st_f = _flag_status(f["flag"].iloc[i])
        st_t = _flag_status(flag_t.iloc[i]) if flag_t is not None else None
        providers.append({"id": str(pid), "group": str(keys[i]), "z": _num(z[i]) if tested[i] else None,
                          "status_fitted": st_f, "status_theoretical": st_t,
                          "changed": "yes" if (st_t is not None and st_t != st_f) else "", "bin": bins[i] if tested[i] else None})
    rec = _base("null_calibration", fig, calibrated, theme)
    rec["legend"] = [{"kind": "band", "cls": "bin", "label": "Raw z-statistics"},
                     {"kind": "line", "cls": "reference dashed", "label": "Theoretical null N(0, 1)"}]
    if fitted_differs:
        rec["legend"].append({"kind": "line", "cls": "curve fitted", "label": "Fitted null"})
    up, down = SYMBOLS["above"], SYMBOLS["below"]
    series_cols = [{"key": "label", "label": "Null group", "type": "text"}, {"key": "providers", "label": "Providers", "digits": 0},
                   {"key": "null_mean", "label": "Null mean", "digits": 2}, {"key": "null_sd", "label": "Null SD", "digits": 2}]
    series_rows = []
    for g in groups:
        r = {"label": g["label"], "providers": g["providers"], "null_mean": g["null_mean"], "null_sd": g["null_sd"],
             "fitted": f"{up} {g['above_fitted']} / {down} {g['below_fitted']}"}
        if compared:
            r.update({"theoretical": f"{up} {g['above_theoretical']} / {down} {g['below_theoretical']}", "changed": g["changed"]})
        series_rows.append(r)
    if compared:
        series_cols += [{"key": "theoretical", "label": "Flagged, theoretical null", "type": "text"},
                        {"key": "fitted", "label": "Flagged, fitted null", "type": "text"},
                        {"key": "changed", "label": "Changed", "digits": 0}]
    else:
        series_cols.append({"key": "fitted", "label": "Flagged", "type": "text"})
    columns = [{"key": "id", "label": "Provider"}, {"key": "group", "label": "Null group", "type": "text"},
               {"key": "z", "label": "Raw z", "digits": 2}]
    if compared:
        columns.append({"key": "status_theoretical", "label": "Theoretical null", "type": "status"})
    columns += [{"key": "status_fitted", "label": "Fitted null" if fitted_differs else "Status", "type": "status"}]
    if compared:
        columns.append({"key": "changed", "label": "Changed", "type": "text"})
    rec.update({
        "axes": {"x": {"label": "Raw z-statistic"}, "y": {"label": "Providers"}},
        "edges": [_num(e, 4) for e in edges], "groups": groups, "providers": providers,
        "fitted_differs": fitted_differs, "compared": compared,
        "series": {"caption": f"{up} above / {down} below the reference, per null group; changed: providers whose flag "
                              "differs between the two nulls.", "columns": series_cols, "rows": series_rows},
        "table_caption": "Providers in the histograms: hover or focus a row to find its bar; hover a bar for its providers.",
        "columns": columns})
    return rec


def reliability_record(iur: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """Each provider's reliability at its size, the overall IUR and n', as reliability(iur) draws them."""
    import numpy as np
    from pprof_py.presentation.formatting import fmt_number

    sizes = np.asarray(iur.group_sizes_, dtype=float)
    rel = np.asarray(iur.iur_groups_, dtype=float)
    labels = [str(v) for v in np.asarray(iur.group_labels_)]
    order = np.argsort(sizes, kind="stable")
    overall, n_prime = float(iur.iur_), float(iur.n_prime_)
    providers = [{"id": labels[i], "size": _num(sizes[i]), "reliability": _num(rel[i])} for i in order]
    rec = _head("reliability", fig, theme)
    rec["legend"] = [{"kind": "shape", "shape": "circle", "cls": "mark filled dot", "size": 3.5, "label": "Provider"},
                     {"kind": "line", "cls": "curve fitted", "label": "Reliability by size"},
                     {"kind": "line", "cls": "reference dashed", "label": f"Overall IUR {fmt_number(overall, 2)}"},
                     {"kind": "line", "cls": "guide dotted", "label": f"Effective size n′ = {fmt_number(n_prime, 0)}"}]
    boot = getattr(iur, "n_boot", None)
    series_rows = [{"label": "Overall IUR", "value": _num(overall)}, {"label": "Effective size n′", "value": _num(n_prime)},
                   {"label": "Between-provider variance", "value": _num(float(iur.s2_between_))},
                   {"label": "Within-provider variance", "value": _num(float(iur.s2_within_))},
                   {"label": "Providers", "value": int(sizes.size)}]
    if boot:
        series_rows.append({"label": "Bootstrap resamples", "value": int(boot)})
    rec.update({
        "axes": {"x": {"label": "Provider size (log scale)", "scale": "log"}, "y": {"label": "Reliability"}},
        "overall": _num(overall), "n_prime": _num(n_prime), "providers": providers,
        "series": {"caption": f"Reliability from {type(iur).__name__}: the overall IUR is the reliability at the effective size.",
                   "columns": [{"key": "label", "label": "Quantity", "type": "text"},
                               {"key": "value", "label": "Value", "digits": 4}], "rows": series_rows},
        "table_caption": "Providers in the chart: hover or focus a row to find it; sort by size or reliability.",
        "columns": [{"key": "id", "label": "Provider"}, {"key": "size", "label": "Size", "digits": 0},
                    {"key": "reliability", "label": "Reliability at this size", "digits": 2}]})
    return rec


def variation_record(re: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """The BLUP histogram, the fitted distribution and the implied range, as provider_variation(model) draws them."""
    import numpy as np
    from pprof_py.presentation._provenance import pct
    from pprof_py.presentation.data._variation import variation_summary
    from pprof_py.presentation.formatting import fmt_number

    v = variation_summary(re)
    b = v.blups.to_numpy(dtype=float)
    n = b.size
    wide = max(v.sigma, v.upper or v.sigma)
    half = max(float(np.nanmax(np.abs(b))), 3.0 * wide) * 1.05
    edges = np.linspace(-half, half, 25)
    counts, _ = np.histogram(b, bins=edges)
    bins = _bins(b, edges)
    ratio = v.scale == "log_odds"
    providers = [{"id": str(pid), "blup": _num(val), "odds_ratio": _num(np.exp(val)) if ratio else None, "bin": bins[i]}
                 for i, (pid, val) in enumerate(zip(v.blups.index, b))]
    p = pct(v.level)
    rec = _head("provider_variation", fig, theme)
    rec["legend"] = [{"kind": "band", "cls": "bin", "label": "Shrunken provider effects (BLUPs)"},
                     {"kind": "line", "cls": "curve fitted", "label": "Fitted between-provider distribution N(0, σ²)"}]
    if v.lower is not None:
        rec["legend"].append({"kind": "line", "cls": "curve dashed", "label": f"At the SD's {p} interval bounds"})
    rec["legend"] += [{"kind": "line", "cls": "reference", "label": "0, the average provider"},
                      {"kind": "line", "cls": "bracket", "label": f"{p} of true effects if normal"}]
    rows = [{"label": "Random-effect SD (σ)", "value": _num(v.sigma)},
            {"label": f"σ, {p} interval: lower", "value": _num(v.lower)}, {"label": f"σ, {p} interval: upper", "value": _num(v.upper)},
            {"label": f"Range of {p} of true effects: lower", "value": _num(v.range_lower)},
            {"label": f"Range of {p} of true effects: upper", "value": _num(v.range_upper)}]
    if v.ratio_lower is not None:
        rows += [{"label": "As odds ratios: lower", "value": _num(v.ratio_lower)}, {"label": "As odds ratios: upper", "value": _num(v.ratio_upper)}]
    rows += [{"label": "Providers", "value": n}, {"label": "SD of the BLUPs (descriptive)", "value": _num(float(np.std(b, ddof=1)))}]
    range_text = f"{fmt_number(v.range_lower, 2)} to {fmt_number(v.range_upper, 2)}"
    if v.ratio_lower is not None:
        range_text += f" (odds ratios {fmt_number(v.ratio_lower, 2)} to {fmt_number(v.ratio_upper, 2)})"
    sigma_text = (f"σ = {fmt_number(v.sigma, 2)} (no interval available)" if v.lower is None else
                  f"σ = {fmt_number(v.sigma, 2)} ({p} {v.interval_method} interval {fmt_number(v.lower, 2)}–{fmt_number(v.upper, 2)})")
    rec.update({
        "axes": {"x": {"label": {"log_odds": "Provider effect (log-odds; 0 = average)",
                                 "difference": "Provider effect (outcome units; 0 = average)"}[v.scale]},
                 "y": {"label": "Providers"}},
        "edges": [_num(e) for e in edges], "counts": [int(c) for c in counts], "half": _num(half),
        "sigma": _num(v.sigma), "lower": _num(v.lower), "upper": _num(v.upper), "level": float(v.level),
        "range": [_num(v.range_lower), _num(v.range_upper)], "range_text": range_text, "sigma_text": sigma_text,
        "degenerate": not np.isfinite(v.sigma) or v.sigma < 1e-6, "providers": providers,
        "series": {"caption": f"Between-provider variation ({v.model}): σ estimates the true variation; the BLUPs "
                              "spread less because they are shrunk toward the average.",
                   "columns": [{"key": "label", "label": "Quantity", "type": "text"},
                               {"key": "value", "label": "Value", "digits": 2, "missing": "not available"}], "rows": rows},
        "table_caption": "Providers in the histogram: hover or focus a row to find its bar; hover a bar for its providers.",
        "columns": [{"key": "id", "label": "Provider"}, {"key": "blup", "label": "BLUP (shrunken effect)", "digits": 2}]
                   + ([{"key": "odds_ratio", "label": "As an odds ratio", "digits": 2}] if ratio else [])})
    return rec


def shrinkage_record(fe: Any, re: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """Fixed-effect estimates against BLUPs, sized by volume, as shrinkage(fixed, random) draws them."""
    import numpy as np
    from pprof_py.presentation.data._shrinkage import reference_text, shrinkage_pairs
    from pprof_py.presentation.formatting import fmt_count, fmt_number

    pairs = shrinkage_pairs(fe, re)
    f, prov = pairs.frame, pairs.provenance
    x, y = f["fixed"].to_numpy(dtype=float), f["random"].to_numpy(dtype=float)
    vol = f["volume"].to_numpy(dtype=float)
    fin = f["fixed_finite"].to_numpy(dtype=bool) & np.isfinite(x)
    vals = np.r_[x[fin], y[np.isfinite(y)], 0.0]
    a, b = float(vals.min()), float(vals.max())
    pad = 0.06 * ((b - a) or 1.0)
    top = float(np.nanmax(vol)) if np.isfinite(vol).any() else 1.0
    scale = {"logistic": "log-odds", "linear": "outcome units"}.get(prov.get("family"), "effect")
    unit = {"records": "records", "trials": "trials", "patients": "patients"}.get(prov.get("denominator_kind"), "volume")
    providers = []
    for i, pid in enumerate(f.index):
        providers.append({"id": str(pid), "fixed": _num(x[i]) if fin[i] else None, "random": _num(y[i]),
                          "volume": _num(vol[i]), "change": _num(y[i] - x[i]) if fin[i] else None,
                          "edge": None if fin[i] else ("lower" if (x[i] if np.isfinite(x[i]) else -1.0) < 0 else "upper")})
    rec = _head("shrinkage", fig, theme)
    legend = []
    if np.isfinite(vol).any():
        for q in (0.1, 0.5, 0.9):
            v = float(f"{float(np.nanquantile(vol, q)):.2g}")
            legend.append({"kind": "shape", "shape": "circle", "cls": "mark open sdot", "size": _num(4.0 + 40.0 * v / top),
                           "area": True, "label": f"{fmt_count(v)} {unit}"})
    if int((~fin).sum()):
        legend.append({"kind": "shape", "shape": "triangle_left", "cls": "mark filled muted-dot", "size": 4.5,
                       "label": "No finite fixed-effect estimate, at the axis edge"})
    legend += [{"kind": "line", "cls": "reference", "label": "No shrinkage (the diagonal)"},
               {"kind": "line", "cls": "guide dashed", "label": "Complete pooling (0)"}]
    rec["legend"] = legend
    rec.update({
        "axes": {"x": {"label": f"Fixed effect, unshrunken ({scale})"}, "y": {"label": f"Random effect, shrunken ({scale})"}},
        "limits": [_num(a - pad), _num(b + pad)], "top": _num(top), "sigma": _num(prov.get("sigma")),
        "reference_text": reference_text(pairs), "providers": providers,
        "table_caption": "Providers in the chart: hover or focus a row to find it; change is the BLUP minus the fixed effect.",
        "columns": [{"key": "id", "label": "Provider"}, {"key": "volume", "label": unit.capitalize(), "digits": 0},
                    {"key": "fixed", "label": "Fixed effect", "digits": 2, "missing": "NE"},
                    {"key": "random", "label": "Random effect (BLUP)", "digits": 2},
                    {"key": "change", "label": "Change", "digits": 2, "missing": "NE"}]})
    return rec


def stability_record(fe: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """The flags of every provider flagged in at least one scenario, as flag_stability(model) draws them."""
    import numpy as np
    from pprof_py.presentation.data._stability import changing, flag_scenarios, stability_summary

    sc = flag_scenarios(fe, (), {"test_method": "score"})                 # the gallery's call
    f = sc.flags
    labels = list(f.columns)
    flagged = ((f == 1) | (f == -1)).fillna(False).to_numpy(dtype=bool).any(axis=1)
    rows = f.index[flagged]
    order = np.argsort(-sc.estimate.reindex(rows).to_numpy(), kind="stable")
    rows = rows[order]
    change_all = changing(sc)
    changed = dict(zip([str(i) for i in f.index], change_all))
    summary = stability_summary(sc)
    scenarios = [{"label": lab, "description": sc.descriptions[lab], "above": int(summary.loc[lab, "above"]),
                  "below": int(summary.loc[lab, "below"]), "not_different": int(summary.loc[lab, "not_different"]),
                  "not_tested": int(summary.loc[lab, "not_tested"]),
                  "changed": None if j == 0 else int(summary.loc[lab, "changed"])} for j, lab in enumerate(labels)]
    providers = []
    for pid in rows:
        p = {"id": str(pid), "estimate": _num(sc.estimate.loc[pid]), "changed": "yes" if changed[str(pid)] else ""}
        for j, lab in enumerate(labels):
            p[f"s{j}"] = _flag_status(f.loc[pid, lab])
        providers.append(p)
    rec = _head("flag_stability", fig, theme)
    rec["legend"] = [{"kind": "mark", "status": k, "label": theme.status[k].label}
                     for k in ("above", "below", "not_different", "not_tested")] + [
        {"kind": "shape", "shape": "diamond", "cls": "mark filled dot", "size": 4, "label": "Status differs between scenarios"}]
    rec.update({
        "scenarios": scenarios, "providers": providers,
        "counts": {"providers": int(len(f)), "flagged_any": int(len(rows)), "changing": int(change_all.sum())},
        "axes": {"y": {"label": f"{len(rows)} providers flagged in a scenario, ordered by the base estimate (not a ranking)"}},
        "series": {"caption": "Per scenario: providers above, below, not different and not tested, and flags changed "
                              "from the base; each scenario is a separate test.",
                   "columns": [{"key": "label", "label": "Scenario", "type": "text"}, {"key": "above", "label": "Above", "digits": 0},
                               {"key": "below", "label": "Below", "digits": 0}, {"key": "not_different", "label": "Not different", "digits": 0},
                               {"key": "not_tested", "label": "Not tested", "digits": 0},
                               {"key": "changed", "label": "Changed from base", "digits": 0, "missing": "—"},
                               {"key": "description", "label": "Test", "type": "text", "wrap": True}],
                   "rows": scenarios},
        "table_caption": "Providers flagged in at least one scenario: hover or focus a row to find it in the matrix.",
        "columns": [{"key": "id", "label": "Provider"}, {"key": "estimate", "label": "Base estimate", "digits": 2}]
                   + [{"key": f"s{j}", "label": lab, "type": "status"} for j, lab in enumerate(labels)]
                   + [{"key": "changed", "label": "Changes", "type": "text"}]})
    return rec


def multi_record(col: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """One panel per measure with providers in a common row order, as multi_measure(measures) draws them."""
    import numpy as np
    from pprof_py.presentation._provenance import test_text

    labels = col.labels
    ids = col.providers()
    first = col.column("estimate")[labels[0]]
    ids = ids[np.argsort(-first.reindex(ids).to_numpy(dtype=float), kind="stable")]
    measures, per = [], []
    for j, label in enumerate(labels):
        f = col[label].data.reindex(ids)
        est, lo, hi = (f[c].to_numpy(dtype=float) for c in ("estimate", "ci_lower", "ci_upper"))
        null = float(np.nanmedian(f["null_value"].to_numpy(dtype=float)))
        present = f["status"].notna().to_numpy()
        finite = present & np.isfinite(est) & f["finite_estimate"].fillna(True).to_numpy(dtype=bool)
        c = col[label].status_counts()
        measures.append({"label": label, "null": _num(null), "limits": _pad_limits(lo[finite], hi[finite], est[finite], extra=null),
                         "test": test_text(col[label].provenance), "above": c["above"], "below": c["below"],
                         "missing": int((~present).sum())})
        per.append((f, est, lo, hi, present, finite))
    providers = []
    for i, pid in enumerate(ids):
        p = {"id": str(pid)}
        for j, (f, est, lo, hi, present, finite) in enumerate(per):
            p[f"e{j}"] = _num(est[i]) if finite[i] else None
            p[f"l{j}"], p[f"u{j}"] = _num(lo[i]), _num(hi[i])
            p[f"s{j}"] = str(f["status"].iloc[i]) if present[i] else None
            p[f"k{j}"] = "finite" if finite[i] else ("ne" if present[i] else "na")
        providers.append(p)
    rec = _head("multi_measure", fig, theme)
    rec["legend"] = [{"kind": "mark", "status": k, "label": theme.status[k].label}
                     for k in ("above", "below", "not_different", "not_tested")] + [
        {"kind": "line", "cls": "interval", "label": "Interval (coloured when flagged)"},
        {"kind": "line", "cls": "reference", "label": "Each measure's own reference"}]
    columns = [{"key": "id", "label": "Provider"}]
    for j, label in enumerate(labels):
        columns += [{"key": f"e{j}", "label": f"{label}: estimate", "digits": 2, "missing": "NE"},
                    {"key": f"l{j}", "label": "lower", "digits": 2}, {"key": f"u{j}", "label": "upper", "digits": 2},
                    {"key": f"s{j}", "label": f"{label}: status", "type": "status", "missing": "n/a"}]
    rec.update({"measures": measures, "providers": providers,
                "axes": {"x": {"label": "Estimate"}, "y": {"label": "Providers, ordered by the first measure's estimate (not a ranking)"}},
                "table_caption": "Providers in every panel: hover or focus a row to find it; n/a: not in that measure; "
                                 "NE: no finite estimate.",
                "columns": columns})
    return rec


def agreement_record(col: Any, fig: Any, theme: Any) -> Dict[str, Any]:
    """Two measures' estimates with interval crosses and joint flag status, as measure_agreement(measures) draws them."""
    import numpy as np
    from pprof_py.presentation.figures._multi import _JOINT, joint_status

    x, y = col.labels[0], col.labels[1]
    ids = col.common()
    fx, fy = col[x].data.reindex(ids), col[y].data.reindex(ids)
    ex, ey = fx["estimate"].to_numpy(dtype=float), fy["estimate"].to_numpy(dtype=float)
    finite = (fx["finite_estimate"].fillna(True).to_numpy(dtype=bool) & fy["finite_estimate"].fillna(True).to_numpy(dtype=bool))
    ok = np.isfinite(ex) & np.isfinite(ey) & finite
    joint = joint_status(fx["flag"], fy["flag"]).to_numpy()
    counts = {key: int(((joint == key) & ok).sum()) for key, *_ in _JOINT}
    nx, ny = float(np.nanmedian(fx["null_value"])), float(np.nanmedian(fy["null_value"]))
    lox, hix = fx["ci_lower"].to_numpy(dtype=float), fx["ci_upper"].to_numpy(dtype=float)
    loy, hiy = fy["ci_lower"].to_numpy(dtype=float), fy["ci_upper"].to_numpy(dtype=float)
    shapes = {"same": "diamond", "opposite": "square", "one": "triangle_up", "neither": "circle", "untested": "circle"}
    filled = {key: fill for key, _, _, fill in _JOINT}
    texts = {key: text for key, text, _, _ in _JOINT}
    providers = []
    for i, pid in enumerate(ids):
        if not ok[i]:
            continue
        providers.append({"id": str(pid), "ex": _num(ex[i]), "ey": _num(ey[i]), "lox": _num(lox[i]), "hix": _num(hix[i]),
                          "loy": _num(loy[i]), "hiy": _num(hiy[i]), "joint": str(joint[i]), "joint_text": texts[str(joint[i])],
                          "sx": _flag_status(fx["flag"].iloc[i]), "sy": _flag_status(fy["flag"].iloc[i])})
    rec = _head("measure_agreement", fig, theme)
    rec["legend"] = [{"kind": "shape", "shape": shapes[key], "cls": f"mark {'filled' if filled[key] else 'open'} jt-{key}",
                      "size": 3 if key == "neither" else 4.2, "label": f"{texts[key]} ({counts[key]:,})"}
                     for key, *_ in _JOINT if counts[key]] + [
        {"kind": "line", "cls": "limit", "label": "Each measure's interval (cross)"},
        {"kind": "line", "cls": "reference", "label": "Each measure's reference"}]
    rec.update({
        "x": {"label": f"{x} (estimate)", "null": _num(nx), "limits": _pad_limits(lox[ok], hix[ok], ex[ok], extra=nx)},
        "y": {"label": f"{y} (estimate)", "null": _num(ny), "limits": _pad_limits(loy[ok], hiy[ok], ey[ok], extra=ny)},
        "joint": {key: {"shape": shapes[key], "filled": filled[key], "label": texts[key]} for key, *_ in _JOINT},
        "counts": {**counts, "not_placed": int((~ok).sum())}, "providers": providers,
        "table_caption": "Providers in both measures with finite estimates: hover or focus a row to find it in the chart.",
        "columns": [{"key": "id", "label": "Provider"}, {"key": "ex", "label": f"{x}: estimate", "digits": 2},
                    {"key": "sx", "label": f"{x}: status", "type": "status"},
                    {"key": "ey", "label": f"{y}: estimate", "digits": 2}, {"key": "sy", "label": f"{y}: status", "type": "status"},
                    {"key": "joint_text", "label": "Joint status", "type": "text"}]})
    return rec


def records() -> Dict[str, Dict[str, Any]]:
    """One JSON-ready record per interactive display, from the gallery's fits and rendered figures."""
    from pprof_py.presentation.theme import get_theme

    m = pprof_gallery.fitted()
    if not pprof_gallery.RESULTS:                    # the gallery did not run first: render its figures now
        for _ in pprof_gallery.figures():
            pass
    theme = get_theme("publication")
    results = pprof_gallery.RESULTS
    return {"funnel": funnel_record(m["fe"], results["funnel"], theme),
            "caterpillar": caterpillar_record(m["exact"], results["caterpillar"], theme),
            "observed_expected": observed_record(m["exact"], results["observed_expected"], theme),
            "forest": forest_record(m["fe"], results["forest"], theme),
            "data_quality": quality_record(m["exact"], results["data_quality"], theme),
            "null_calibration": calibration_record(m["fe"], results["null_calibration"], theme),
            "reliability": reliability_record(m["iur"], results["reliability"], theme),
            "provider_variation": variation_record(m["re"], results["provider_variation"], theme),
            "shrinkage": shrinkage_record(m["fe"], m["re"], results["shrinkage"], theme),
            "flag_stability": stability_record(m["fe"], results["flag_stability"], theme),
            "multi_measure": multi_record(m["measures"], results["multi_measure"], theme),
            "measure_agreement": agreement_record(m["measures"], results["measure_agreement"], theme)}


def _page(rec: Dict[str, Any], css: str, js: str) -> bytes:
    data = json.dumps(rec, sort_keys=True, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")
    return TEMPLATE.format(title=html.escape(rec["title"]), question=html.escape(rec["question"]),
                           caption=html.escape(rec["caption"]), css=css.strip(), js=js.strip(),
                           data=data).encode("utf-8")


def _static_snippet(name: str) -> str:
    """The static image with its alt text: the same line the gallery writes to ``_figures/<name>.md``."""
    fig = pprof_gallery.RESULTS[name]
    return f"![{pprof_gallery._alt(fig.alt_text)}](/_static/gallery/{name}.svg)"


def _snippet(name: str, rec: Dict[str, Any]) -> bytes:
    src = f"{STATIC_PREFIX}_static/interactive/{name}.html"
    return SNIPPET.format(src=src, title=html.escape(rec["title"]), lower=html.escape(rec["title"].lower()),
                          label=f"interactive-{name.replace('_', '-')}", hover=html.escape(HOVER.get(name, DEFAULT_HOVER)),
                          height=IFRAME_HEIGHT[name], static=_static_snippet(name)).encode("utf-8")


def build(app) -> None:
    src = Path(app.srcdir)
    out = src / "_static" / "interactive"
    out.mkdir(parents=True, exist_ok=True)
    per_figure = src / "presentation" / "_figures"
    per_figure.mkdir(parents=True, exist_ok=True)
    css = (ASSETS / "interactive.css").read_text(encoding="utf-8")
    js = (ASSETS / "interactive.js").read_text(encoding="utf-8")
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            recs = records()
    finally:
        logging.disable(logging.NOTSET)
    for name in DISPLAYS:
        pprof_gallery._write(out / f"{name}.html", _page(recs[name], css, js))
        pprof_gallery._write(per_figure / f"{name}.md", _snippet(name, recs[name]))   # replaces the gallery's static line


def setup(app):
    app.connect("builder-inited", build)          # after pprof_gallery's listener: same event, registered later
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
