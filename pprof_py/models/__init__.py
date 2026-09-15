"""Public estimator classes: public configuration, input handling, fitting,
predictions, and fitted attributes.  See ``models.logistic``, ``models.linear``,
and ``models.survival`` for the model families; ``models.mixins`` for Protocol
contracts.

Tier 4 retired ``BaseModel``.  Tier 6 moved ``validate_and_convert_inputs``
and ``ValidatedInputs`` to ``data.validation`` (their canonical home,
parallel to ``data.survival_validation``) and deleted ``models.base``."""
from ..data.validation import validate_and_convert_inputs, ValidatedInputs
from .mixins import SummaryMixin, TestMixin, PlotMixin

__all__ = [
    "validate_and_convert_inputs",
    "ValidatedInputs",
    "SummaryMixin",
    "TestMixin",
    "PlotMixin",
]
