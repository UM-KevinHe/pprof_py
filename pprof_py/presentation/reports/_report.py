"""A single-file HTML report: sections, text, figures, tables and a generated methods and provenance appendix."""
from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Union

from .._provenance import null_text, pct, reference_text, test_text
from ..figures._result import FigureResult
from ..tables._provider import TableResult
from ..tables._spec import TABLE_CSS, table_css, table_markup
from ..theme import get_theme

__all__ = ["Report"]

_CSS = """body{margin:0;background:#fff;color:#1a1a1a;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;\
font-size:15px;line-height:1.5}
main{max-width:980px;margin:0 auto;padding:24px 20px 48px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:32px 0 8px}
.subtitle{color:#4d4d4d;margin:0 0 16px}
.note{border-left:3px solid #8c8c8c;padding:4px 12px;color:#4d4d4d;font-size:14px}
figure{margin:20px 0}figure img{max-width:100%;height:auto;display:block}
figcaption{font-size:14px;color:#1a1a1a;margin-top:6px}figcaption .label{font-weight:600}
details{font-size:14px;color:#4d4d4d;margin-top:4px}summary{cursor:pointer}
summary:focus-visible{outline:2px solid #1a1a1a;outline-offset:2px}
.table-wrap{overflow-x:auto;margin:20px 0}
footer{color:#4d4d4d;font-size:13px;border-top:1px solid #d9d9d9;margin-top:40px;padding-top:8px}
@media print{main{max-width:none;padding:0}figure,.table-wrap{break-inside:avoid}details{display:none}
h2{break-after:avoid}}"""


def _page_css(t: Any) -> str:
    """The page style built from the theme's tokens (the identity); the classic style is ``_CSS``."""
    face = f'"{t.typography.family}",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif'
    band = t.corridor or t.grid_color
    return (f"body{{margin:0;background:{t.background};color:{t.ink};font-family:{face};font-size:15px;line-height:1.55}}\n"
            "main{max-width:980px;margin:0 auto;padding:32px 24px 56px}\n"
            "h1{font-size:28px;font-weight:600;line-height:1.15;margin:0 0 6px;letter-spacing:-0.2px}\n"
            f"h2{{font-size:18px;font-weight:600;margin:36px 0 10px;padding-top:14px;border-top:1px solid {t.grid_color}}}\n"
            f".subtitle{{color:{t.muted};font-size:16px;margin:0 0 20px;max-width:70ch}}\n"
            "p{max-width:75ch}\n"
            f".note{{background:{band};padding:8px 12px;color:{t.ink};font-size:14px}}\n"
            "figure{margin:22px 0}figure img{max-width:100%;height:auto;display:block}\n"
            f"figcaption{{font-size:14px;color:{t.muted};margin-top:8px;max-width:80ch}}"
            f"figcaption .label{{font-weight:600;color:{t.ink}}}\n"
            f"details{{font-size:14px;color:{t.muted};margin-top:4px}}summary{{cursor:pointer}}\n"
            f"summary:focus-visible{{outline:2px solid {t.ink};outline-offset:2px}}\n"
            ".table-wrap{overflow-x:auto;margin:22px 0}\n"
            f"footer{{color:{t.muted};font-size:13px;border-top:1px solid {t.grid_color};margin-top:44px;padding-top:10px}}\n"
            "@media print{main{max-width:none;padding:0}figure,.table-wrap{break-inside:avoid}details{display:none}\n"
            "h2{break-after:avoid}}")


def _font_faces(family: str) -> str:
    """``@font-face`` rules embedding the bundled face's Regular and SemiBold files (``embed_fonts=True``); empty,
    with the fonts module's one warning, when the files are unavailable, so the page falls back to named fonts."""
    from ..theme import _fonts

    rules = []
    for weight, key in ((400, "normal"), (600, "semibold")):
        path = _fonts.font_file(key, "normal") if _fonts.is_bundled(family) else None
        if path is None:
            return ""
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        rules.append(f'@font-face{{font-family:"{family}";src:url(data:font/ttf;base64,{data}) format("truetype");'
                     f"font-weight:{weight};font-style:normal}}")
    return ('/* IBM Plex Sans: Copyright 2017 IBM Corp. with Reserved Font Name "Plex"; SIL Open Font License 1.1 */\n'
            + "\n".join(rules) + "\n")


def _row(kind: str, number: int, label: str, prov: Mapping[str, Any]) -> List[str]:
    model = prov.get("model") or prov.get("fixed_model") or prov.get("source") or "\u2014"
    test = test_text(prov) if prov.get("test_method") is not None else "\u2014"
    null = null_text(prov.get("null_model")) if "null_model" in prov else "\u2014"
    ref = reference_text(prov, False) if prov.get("reference_value") is not None else "\u2014"
    level = pct(prov["level"]) if isinstance(prov.get("level"), (int, float)) else "\u2014"
    return [f"{kind} {number}", label, str(model), test, null, ref, level]


class Report:
    """A report built block by block and written as one self-contained HTML file.

    Parameters
    ----------
    title : str
    subtitle : str, optional
    date : str, optional
        Printed in the header when given; no timestamp is added otherwise, so the same report gives the same bytes.
    theme : str or Theme, default "report"
        Styles the page and its tables (figures keep the theme they were drawn with); ``"classic"`` gives the 0.6.0
        report.
    embed_fonts : bool, default False
        Embed the bundled IBM Plex Sans (Regular and SemiBold, about 0.5 MB) so the page text shows in it everywhere;
        by default the font is named first and the reader's system font is used where it is not installed.

    Notes
    -----
    Figures are embedded as SVG images with their alt text and long description; tables keep their semantic markup.
    The methods and provenance appendix lists, for every figure and table, the model, test, null model, reference
    and level recorded with it. When the report contains tables, a note says that they carry provider-level values
    (S13). The file loads nothing from the network and contains no scripts.

    Examples
    --------
    >>> report = Report("Facility profile").section("Results").figure(funnel(model)).table(provider_table(model))
    >>> report.save("profile.html")                                           # doctest: +SKIP
    """

    def __init__(self, title: str, *, subtitle: Optional[str] = None, date: Optional[str] = None,
                 theme: Any = "report", embed_fonts: bool = False) -> None:
        self._title, self._subtitle, self._date = str(title), subtitle, date
        self._theme, self._embed = get_theme(theme), bool(embed_fonts)
        self._blocks: List[Tuple[str, Any, Optional[str]]] = []

    def section(self, heading: str) -> "Report":
        """Start a section with this heading."""
        self._blocks.append(("section", str(heading), None))
        return self

    def text(self, body: str) -> "Report":
        """Paragraphs of plain text (separated by blank lines); the text is escaped, not interpreted as HTML."""
        self._blocks.append(("text", str(body), None))
        return self

    def figure(self, fig: FigureResult, *, caption: Optional[str] = None) -> "Report":
        if not isinstance(fig, FigureResult):
            raise TypeError("figure() takes a FigureResult, as returned by the presentation figures")
        self._blocks.append(("figure", fig, caption))
        return self

    def table(self, table: TableResult) -> "Report":
        if not isinstance(table, TableResult):
            raise TypeError("table() takes a TableResult, as returned by the presentation tables")
        self._blocks.append(("table", table, None))
        return self

    def outline(self) -> List[str]:
        """The report's contents in order: sections, numbered figures and tables, and the appendix."""
        out, nf, nt = [], 0, 0
        for kind, item, caption in self._blocks:
            if kind == "section":
                out.append(f"Section: {item}")
            elif kind == "figure":
                nf += 1
                out.append(f"Figure {nf}: {item.kind}" + (f" ({caption})" if caption else ""))
            elif kind == "table":
                nt += 1
                out.append(f"Table {nt}: {item.spec.caption}")
            else:
                out.append("Text")
        return out + ["Appendix: methods and provenance"]

    def to_html(self) -> str:
        """The report as a complete HTML document (deterministic)."""
        from ... import __version__

        e = html.escape
        ident = self._theme.table_style == "identity"
        body, rows = [], []
        nf = nt = ns = 0
        open_section = False
        has_tables = any(kind == "table" for kind, _, _ in self._blocks)
        for kind, item, caption in self._blocks:
            if kind == "section":
                if open_section:
                    body.append("</section>")
                ns += 1
                body.append(f'<section aria-labelledby="sec-{ns}"><h2 id="sec-{ns}">{e(item)}</h2>')
                open_section = True
            elif kind == "text":
                body += [f"<p>{e(par.strip())}</p>" for par in item.split("\n\n") if par.strip()]
            elif kind == "figure":
                nf += 1
                svg = base64.b64encode(item.to_bytes("svg")).decode("ascii")
                body.append(f'<figure id="fig-{nf}"><img src="data:image/svg+xml;base64,{svg}" alt="{e(item.alt_text)}" '
                            f'aria-describedby="fig-{nf}-desc"><figcaption><span class="label">Figure {nf}.</span> '
                            f"{e(caption or item.alt_text)}</figcaption><details><summary>Description</summary>"
                            f'<p id="fig-{nf}-desc">{e(item.long_description)}</p></details></figure>')
                rows.append(_row("Figure", nf, item.kind, item.provenance))
            else:
                nt += 1
                markup = (table_markup(item.spec, caption_prefix=f"Table {nt}. ", theme=self._theme,
                                       intervals=getattr(item, "_intervals", None)) if ident
                          else table_markup(item.spec, caption_prefix=f"Table {nt}. "))
                body.append(f'<div class="table-wrap" id="tab-{nt}">{markup}</div>')
                rows.append(_row("Table", nt, item.spec.caption, item.spec.provenance))
        if open_section:
            body.append("</section>")
        head = ["Item", "Display", "Model", "Test", "Null model", "Reference", "Level"]
        appendix = ['<section aria-labelledby="methods"><h2 id="methods">Methods and provenance</h2>',
                    '<div class="table-wrap"><table class="pprof-table"><caption>Settings recorded with each figure '
                    "and table</caption><thead><tr>" + "".join(f'<th scope="col" class="left">{h}</th>' for h in head)
                    + "</tr></thead><tbody>"]
        for r in rows:
            appendix.append("<tr>" + f'<th scope="row" class="left">{e(r[0])}</th>'
                            + "".join(f'<td class="left">{e(v)}</td>' for v in r[1:]) + "</tr>")
        appendix.append("</tbody></table></div></section>")
        notes = []
        if has_tables:
            notes.append('<p class="note">The tables in this report list provider-level values. Check that sharing them '
                         "is allowed before the file leaves a secure environment; the figures carry only what they "
                         "draw.</p>")
        header = [f"<h1>{e(self._title)}</h1>"]
        if self._subtitle:
            header.append(f'<p class="subtitle">{e(self._subtitle)}</p>')
        footer = f"Generated with pprof_py {e(__version__)}" + (f" on {e(self._date)}" if self._date else "") + "."
        return ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                f"<title>{e(self._title)}</title>\n<style>{self._css()}</style>\n</head>\n<body>\n<main>\n"
                + "\n".join(header + notes + body + appendix)
                + f"\n<footer>{footer}</footer>\n</main>\n</body>\n</html>\n")

    def _css(self) -> str:
        if self._theme.table_style != "identity":
            return f"{_CSS}\n{TABLE_CSS}"
        faces = _font_faces(self._theme.typography.family) if self._embed else ""
        return f"{faces}{_page_css(self._theme)}\n{table_css(self._theme)}"

    def save(self, path: Union[str, Path]) -> Path:
        """Write the report; returns the path."""
        path = Path(path)
        path.write_bytes(self.to_html().encode("utf-8"))
        return path

    def _repr_html_(self) -> str:
        return self.to_html()

    def __repr__(self) -> str:
        return f"Report({self._title!r}; {len(self._blocks)} blocks)"
