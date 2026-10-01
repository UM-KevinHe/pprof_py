"""Tables: a library-independent specification with deterministic HTML, Markdown, LaTeX, text and Excel output.

:func:`provider_table` builds the provider summary table from a model, a ``test()`` result or a profile; its
footnotes come from the test's settings and the providers' statuses (spec §3; brief §6).
"""
from ._calibration import null_calibration_table
from ._multi import multi_measure_table
from ._coefficients import coefficient_table
from ._quality import data_quality_table
from ._reliability import reliability_table
from ._shrinkage import shrinkage_table
from ._stability import flag_stability_table
from ._variation import provider_variation_table
from ._provider import TableResult, provider_table
from ._spec import Column, TableSpec, render_html, render_latex, render_markdown, render_text

__all__ = ["Column", "TableResult", "TableSpec", "coefficient_table", "data_quality_table", "flag_stability_table", "multi_measure_table", "null_calibration_table", "provider_table", "provider_variation_table", "reliability_table", "shrinkage_table", "render_html", "render_latex", "render_markdown",
           "render_text"]
