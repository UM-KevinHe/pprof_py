"""Figures: deterministic, accessible renderings of presentation data (spec §2, §7, §9).

Every renderer returns a :class:`FigureResult`; none uses pyplot or leaves global state behind.
"""
from ._calibration import null_calibration
from ._caterpillar import caterpillar
from ._forest import forest
from ._quality import data_quality
from ._funnel import funnel
from ._multi import measure_agreement, multi_measure
from ._observed import observed_expected
from ._reliability import reliability
from ._shrinkage import shrinkage
from ._stability import flag_stability
from ._variation import provider_variation
from ._result import FigureResult

__all__ = ["FigureResult", "caterpillar", "data_quality", "flag_stability", "forest", "funnel", "measure_agreement", "multi_measure", "null_calibration", "observed_expected", "provider_variation", "reliability", "shrinkage"]
