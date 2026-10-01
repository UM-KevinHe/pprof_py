"""Tables: a library-independent specification with deterministic HTML, Markdown, LaTeX, text and Excel output.

:func:`provider_table` builds the provider summary table from a model, a ``test()`` result or a profile; its
footnotes come from the test's settings and the providers' statuses (spec §3; brief §6).
"""
from ._coefficients import coefficient_table
from ._provider import TableResult, provider_table
from ._spec import Column, TableSpec, render_html, render_latex, render_markdown, render_text

__all__ = ["Column", "TableResult", "TableSpec", "coefficient_table", "provider_table", "render_html", "render_latex", "render_markdown",
           "render_text"]
