"""Figures: deterministic, accessible renderings of presentation data (spec §2, §7, §9).

Every renderer returns a :class:`FigureResult`; none uses pyplot or leaves global state behind.
"""
from ._calibration import null_calibration
from ._caterpillar import caterpillar
from ._forest import forest
from ._quality import data_quality
from ._funnel import funnel
from ._observed import observed_expected
from ._reliability import reliability
from ._result import FigureResult

__all__ = ["FigureResult", "caterpillar", "data_quality", "forest", "funnel", "null_calibration", "observed_expected", "reliability"]
