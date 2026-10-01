"""Table specification (the public document model) and its renderers (spec §3; brief §6)."""
from __future__ import annotations

import html
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

__all__ = ["Column", "TableSpec", "render_html", "render_latex", "render_markdown", "render_text", "excel_bytes"]

_ROLES = ("id", "count", "estimate", "interval", "p_value", "flag", "text", "percent")
_SUPERSCRIPTS = {"a": "\u1d43", "b": "\u1d47", "c": "\u1d9c", "d": "\u1d48", "e": "\u1d49", "f": "\u1da0"}
_LONGTABLE_ROWS = 40


@dataclass(frozen=True)
class Column:
    """One table column.

    Attributes
    ----------
    key : str
        Column of :attr:`TableSpec.cells` holding the formatted text.
    header : str
        Header text.
    role : str
        One of ``"id"``, ``"count"``, ``"estimate"``, ``"interval"``, ``"p_value"``, ``"flag"``, ``"text"``,
        ``"percent"``; sets alignment and styling.
    marker : str
        Footnote marker shown after the header (``""`` for none).
    spanner : str, optional
        Grouped header above this column.
    """

    key: str
    header: str
    role: str
    marker: str = ""
    spanner: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"role must be one of {_ROLES}, got {self.role!r}")

    @property
    def align(self) -> str:
        return {"id": "left", "text": "left", "flag": "center"}.get(self.role, "right")


@dataclass(frozen=True, eq=False)
class TableSpec:
    """A library-independent table: formatted cells plus everything needed to render and footnote them.

    Attributes
    ----------
    columns : tuple of Column
    cells : pandas.DataFrame
        Formatted text, one row per table row, one column per :attr:`Column.key`.
    values : pandas.DataFrame
        The underlying numbers and statuses (tidy; written to Excel as numeric cells).
    number_formats : mapping
        Excel number format per column of ``values``.
    caption : str
    notes : tuple of (marker, text)
        Footnotes; a marker links a note to headers carrying it.
    source_note : str
    provenance : mapping
    """

    columns: Tuple[Column, ...]
    cells: pd.DataFrame
    values: pd.DataFrame
    caption: str
    notes: Tuple[Tuple[str, str], ...] = ()
    source_note: str = ""
    number_formats: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [c.key for c in self.columns if c.key not in self.cells.columns]
        if missing:
            raise ValueError(f"cells lack the columns {missing}")
        markers = {c.marker for c in self.columns if c.marker}
        noted = {m for m, _ in self.notes if m}
        if not markers <= noted:
            raise ValueError(f"header markers without a note: {sorted(markers - noted)}")

    def rows(self) -> List[List[str]]:
        return self.cells[[c.key for c in self.columns]].astype(str).to_numpy().tolist()


def _spanners(spec: TableSpec) -> List[Tuple[Optional[str], int]]:
    out: List[Tuple[Optional[str], int]] = []
    for c in spec.columns:
        if out and out[-1][0] == c.spanner and c.spanner is not None:
            out[-1] = (c.spanner, out[-1][1] + 1)
        else:
            out.append((c.spanner, 1))
    return out


# ------------------------------------------------------------------------------------------------------- HTML
_CSS = """.pprof-table{border-collapse:collapse;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;\
font-size:14px;font-variant-numeric:tabular-nums;border-top:2px solid #1a1a1a;border-bottom:2px solid #1a1a1a;\
color:#1a1a1a}
.pprof-table caption{caption-side:top;text-align:left;font-weight:600;padding:0 0 6px 0}
.pprof-table th,.pprof-table td{padding:3px 10px;vertical-align:top}
.pprof-table thead th{border-bottom:1px solid #1a1a1a;font-weight:600}
.pprof-table thead tr.spanners th{border-bottom:1px solid #8c8c8c}
.pprof-table tbody th{font-weight:400;text-align:left}
.pprof-table .left{text-align:left}.pprof-table .right{text-align:right}.pprof-table .center{text-align:center}
.pprof-table tfoot td{font-size:12px;color:#4d4d4d;border-top:1px solid #1a1a1a;padding-top:6px}
.pprof-table tfoot p{margin:2px 0}
@media print{.pprof-table{font-size:9pt}.pprof-table thead{display:table-header-group}}"""


def render_html(spec: TableSpec, *, standalone: bool = True) -> str:
    """Self-contained HTML: ``<caption>``, ``<th scope>``, footnotes in ``<tfoot>``; no scripts, fonts or links."""
    e = html.escape
    out = ['<table class="pprof-table">', f"<caption>{e(spec.caption)}</caption>", "<thead>"]
    groups = _spanners(spec)
    if any(name for name, _ in groups):
        cells = "".join(f'<th scope="colgroup" colspan="{n}" class="center">{e(name)}</th>' if name
                        else (f'<td colspan="{n}"></td>' if n > 1 else "<td></td>") for name, n in groups)
        out.append(f'<tr class="spanners">{cells}</tr>')
    head = "".join(f'<th scope="col" class="{c.align}">{e(c.header)}'
                   f'{f"<sup>{e(c.marker)}</sup>" if c.marker else ""}</th>' for c in spec.columns)
    out += [f"<tr>{head}</tr>", "</thead>", "<tbody>"]
    for row in spec.rows():
        tds = []
        for c, value in zip(spec.columns, row):
            tag = 'th scope="row"' if c.role == "id" else "td"
            tds.append(f'<{tag} class="{c.align}">{e(value)}</{tag.split()[0]}>')
        out.append(f"<tr>{''.join(tds)}</tr>")
    out.append("</tbody>")
    notes = [f"<p>{f'<sup>{e(m)}</sup> ' if m else ''}{e(t)}</p>" for m, t in spec.notes]
    if spec.source_note:
        notes.append(f"<p>{e(spec.source_note)}</p>")
    if notes:
        out.append(f'<tfoot><tr><td colspan="{len(spec.columns)}">{"".join(notes)}</td></tr></tfoot>')
    out.append("</table>")
    table = "\n".join(out)
    if not standalone:
        return f"<style>{_CSS}</style>\n{table}\n"
    return ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f"<title>{e(spec.caption)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n{table}\n</body>\n</html>\n")


# --------------------------------------------------------------------------------------------------- Markdown
def _sup(marker: str) -> str:
    return "".join(_SUPERSCRIPTS.get(ch, ch) for ch in marker)


def _flat_header(c: Column) -> str:
    """A header for formats without column spans (Markdown, text): the spanner, if any, becomes a prefix."""
    return (f"{c.spanner}: {c.header}" if c.spanner else c.header) + _sup(c.marker)


def render_markdown(spec: TableSpec) -> str:
    """GitHub-flavoured Markdown with an alignment row; footnotes as lines below the table."""
    def cell(text: str) -> str:
        return text.replace("|", "\\|")
    align = {"left": ":--", "right": "--:", "center": ":-:"}
    lines = [f"**{cell(spec.caption)}**", ""]
    lines.append("| " + " | ".join(cell(_flat_header(c)) for c in spec.columns) + " |")
    lines.append("|" + "|".join(align[c.align] for c in spec.columns) + "|")
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in spec.rows()]
    for m, t in spec.notes:                         # one paragraph per note: no trailing-space line breaks
        lines += ["", f"{_sup(m)} {t}".strip()]
    if spec.source_note:
        lines += ["", spec.source_note]
    return "\n".join(line.rstrip() for line in lines) + "\n"


# ------------------------------------------------------------------------------------------------------ LaTeX
_LATEX_TEXT = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
               "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}",
               ">": r"\textgreater{}", "|": r"\textbar{}"}
_LATEX_SYMBOLS = {"\u2212": "$-$", "\u2013": "--", "\u25b2": r"$\blacktriangle$", "\u25bc": r"$\blacktriangledown$",
                  "\u25cf": r"$\bullet$", "\u221e": r"$\infty$", "\u2014": "---", "\u2265": r"$\geq$",
                  "\u2264": r"$\leq$", "\u00b2": r"\textsuperscript{2}"}


def _tex(text: str) -> str:
    return "".join(_LATEX_TEXT.get(ch, _LATEX_SYMBOLS.get(ch, ch)) for ch in str(text))


def render_latex(spec: TableSpec) -> str:
    """LaTeX with booktabs rules; ``longtable`` above 40 rows. Needs ``booktabs``, ``longtable`` and ``amssymb``."""
    colspec = "".join({"left": "l", "right": "r", "center": "c"}[c.align] for c in spec.columns)
    head = []
    groups = _spanners(spec)
    if any(name for name, _ in groups):
        cells, rules, start = [], [], 1
        for name, k in groups:
            cells.append(rf"\multicolumn{{{k}}}{{c}}{{{_tex(name)}}}" if name else " & ".join([""] * k))
            if name:
                rules.append(rf"\cmidrule(lr){{{start}-{start + k - 1}}}")
            start += k
        head += [" & ".join(cells) + r" \\", " ".join(rules)]
    head.append(" & ".join(_tex(c.header) + (rf"\textsuperscript{{{_tex(c.marker)}}}" if c.marker else "")
                           for c in spec.columns) + r" \\")
    body = [" & ".join(_tex(v) for v in row) + r" \\" for row in spec.rows()]
    notes = [(rf"\textsuperscript{{{_tex(m)}}}~" if m else "") + _tex(t) for m, t in spec.notes]
    if spec.source_note:
        notes.append(_tex(spec.source_note))
    out = ["% Requires \\usepackage{booktabs}, \\usepackage{longtable} and \\usepackage{amssymb}."]
    if len(spec.cells) > _LONGTABLE_ROWS:
        out += [rf"\begin{{longtable}}{{{colspec}}}", rf"\caption{{{_tex(spec.caption)}}}\\", r"\toprule", *head,
                r"\midrule", r"\endfirsthead", r"\toprule", *head, r"\midrule", r"\endhead", r"\bottomrule",
                r"\endlastfoot", *body]
        out += [r"\end{longtable}"]
        if notes:
            out += [r"\begin{flushleft}\footnotesize", r"\\".join(notes), r"\end{flushleft}"]
    else:
        out += [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{_tex(spec.caption)}}}",
                rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *head, r"\midrule", *body, r"\bottomrule",
                r"\end{tabular}"]
        if notes:
            joined = (r"\\" + " ").join(notes)
            out += [r"\par\smallskip", r"\parbox{\linewidth}{\footnotesize " + joined + "}"]
        out += [r"\end{table}"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------------- text
def render_text(spec: TableSpec) -> str:
    """Plain text with aligned columns and rules; footnotes below."""
    rows = spec.rows()
    headers = [_flat_header(c) for c in spec.columns]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def line(values: Sequence[str]) -> str:
        parts = []
        for c, v, w in zip(spec.columns, values, widths):
            parts.append(v.ljust(w) if c.align == "left" else (v.center(w) if c.align == "center" else v.rjust(w)))
        return "  ".join(parts).rstrip()
    rule = "  ".join("-" * w for w in widths)
    out = [spec.caption, "=" * len(rule), line(headers), rule, *[line(r) for r in rows], "=" * len(rule)]
    out += [f"{_sup(m)} {t}".strip() for m, t in spec.notes]
    if spec.source_note:
        out.append(spec.source_note)
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------------ Excel
_EPOCH = datetime(2000, 1, 1)


def excel_bytes(spec: TableSpec, *, sheet: str = "providers") -> bytes:
    """An .xlsx workbook: numeric cells with number formats, frozen header, notes and provenance on a second sheet.

    Needs the optional dependency XlsxWriter (``pip install "pprof_py[excel]"``). The workbook's dates are fixed,
    so the same table gives the same bytes.
    """
    try:
        import xlsxwriter
    except ImportError as err:                      # pragma: no cover - exercised only without the extra
        raise ImportError('Excel output needs the optional dependency XlsxWriter: pip install "pprof_py[excel]"') from err
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    wb.set_properties({"title": spec.caption, "created": _EPOCH, "author": "", "comments": "pprof_py"})
    bold = wb.add_format({"bold": True, "bottom": 1})
    formats: Dict[str, Any] = {}
    ws = wb.add_worksheet(sheet)
    values = spec.values.reset_index()
    for j, name in enumerate(values.columns):
        ws.write_string(0, j, str(name), bold)
        fmt = spec.number_formats.get(name)
        if fmt and fmt not in formats:
            formats[fmt] = wb.add_format({"num_format": fmt})
        col = values[name]
        for i, v in enumerate(col.tolist(), start=1):
            if v is None or v is pd.NA or (isinstance(v, float) and v != v):
                ws.write_blank(i, j, None)
            elif isinstance(v, (bool,)) or not isinstance(v, (int, float)):
                ws.write_string(i, j, str(v))
            elif v in (float("inf"), float("-inf")):
                ws.write_string(i, j, "\u221e" if v > 0 else "\u2212\u221e")
            else:
                ws.write_number(i, j, float(v), formats.get(fmt) if fmt else None)
        ws.set_column(j, j, max(10, min(28, len(str(name)) + 2)))
    ws.freeze_panes(1, 0)
    notes = wb.add_worksheet("notes")
    lines = [spec.caption, ""] + [f"{m} {t}".strip() for m, t in spec.notes] + ([spec.source_note] if spec.source_note else [])
    lines += ["", "Provenance"] + [f"{k}: {v}" for k, v in spec.provenance.items() if not isinstance(v, (dict, list, tuple))]
    for i, text in enumerate(lines):
        notes.write_string(i, 0, str(text))
    notes.set_column(0, 0, 100)
    wb.close()
    return buf.getvalue()
