"""Sphinx extension: interactive versions of the documentation figures, built from the gallery's fits.

Registered after ``pprof_gallery`` in ``conf.py``. On ``builder-inited`` it takes the gallery's fitted models
(``pprof_gallery.fitted()``) and rendered figures (``pprof_gallery.RESULTS``, for the alt text and caption), so
nothing is fitted twice, and for the funnel plot, the interval plot ("caterpillar") and the observed-versus-expected
display it extracts one JSON record: provider id, estimate, precision or interval, status, counts and volume, the
limit curves at each level, the reference value, the axis labels and the legend. Each record is embedded in a
self-contained HTML file, ``_static/interactive/<name>.html``, with the CSS and JavaScript from
``_ext/pprof_interactive_assets/`` inlined (plain JavaScript and inline SVG; no dependencies, nothing from the
network). It also writes ``presentation/_figures/<name>_interactive.md``: a ``<details>`` disclosure with a lazily
loaded ``<iframe>`` of that file and a link to open it in a new tab, which the display pages include under the static
figure. The static SVG stays the default and the printed version, and the package's presentation layer is untouched
(S13: its own outputs stay static and script-free).

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
DISPLAYS = ("funnel", "caterpillar", "observed_expected")
IFRAME_HEIGHT = {"funnel": 1040, "caterpillar": 1300, "observed_expected": 1140}   # px, from the rendered pages
STATIC_PREFIX = "../../"              # the display pages live in presentation/displays/, two levels below the root
LEVELS = (0.95, 0.998)                # the gallery's funnel levels (funnel(model) defaults)

STATUS_ORDER = ("above", "below", "not_different", "not_tested")
MARKERS = {"above": "triangle_up", "below": "triangle_down", "not_different": "circle", "not_tested": "circle_open",
           "zero_events": "square_open", "no_finite_estimate": "triangle_left"}
SYMBOLS = {"above": "▲", "below": "▼", "not_different": "●", "not_tested": "○",
           "zero_events": "□", "no_finite_estimate": "◁"}
LABELS = {"zero_events": "Zero events, O/E = 0", "no_finite_estimate": "No finite estimate, at the axis edge"}
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

SNIPPET = """<details class="pprof-interactive-figure">
<summary>Interactive version: hover for each provider's values, sort the table, or read it as a table (the static
figure above is the printed version)</summary>
<iframe src="{src}" title="{title} (interactive)" loading="lazy" width="100%" height="{height}"
style="border: 1px solid rgba(128, 128, 128, 0.35); border-radius: 4px; background: transparent;"></iframe>
<p><a href="{src}" target="_blank" rel="noopener">Open the interactive {lower} in a new tab</a></p>
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


def _base(name: str, fig: Any, prof: Any, theme: Any) -> Dict[str, Any]:
    title, question, _ = pprof_gallery.ITEMS[name]
    prov = prof.provenance
    return {"display": name, "title": title, "question": question, "alt": fig.alt_text, "caption": fig.caption,
            "level": float(prov.get("level") or 0.95), "test": prov.get("test_method"),
            "n_providers": int(len(prof)), "statuses": _statuses(theme)}


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
            "observed_expected": observed_record(m["exact"], results["observed_expected"], theme)}


def _page(rec: Dict[str, Any], css: str, js: str) -> bytes:
    data = json.dumps(rec, sort_keys=True, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")
    return TEMPLATE.format(title=html.escape(rec["title"]), question=html.escape(rec["question"]),
                           caption=html.escape(rec["caption"]), css=css.strip(), js=js.strip(),
                           data=data).encode("utf-8")


def _snippet(name: str, rec: Dict[str, Any]) -> bytes:
    src = f"{STATIC_PREFIX}_static/interactive/{name}.html"
    return SNIPPET.format(src=src, title=html.escape(rec["title"]), lower=html.escape(rec["title"].lower()),
                          height=IFRAME_HEIGHT[name]).encode("utf-8")


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
        pprof_gallery._write(per_figure / f"{name}_interactive.md", _snippet(name, recs[name]))


def setup(app):
    app.connect("builder-inited", build)          # after pprof_gallery's listener: same event, registered later
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
