"""Public estimator classes: configuration, input handling, fitting,
predictions, and fitted attributes.  See ``models.logistic``,
``models.linear``, and ``models.survival`` for the model families;
``models.mixins`` for Protocol contracts.

``validate_and_convert_inputs`` and ``ValidatedInputs`` live in
``data.validation`` (their canonical home, parallel to
``data.survival_validation``) and are re-exported here for backward
compatibility."""
from ..data.validation import validate_and_convert_inputs, ValidatedInputs
from .mixins import SummaryMixin, TestMixin, PlotMixin

__all__ = [
    "validate_and_convert_inputs",
    "ValidatedInputs",
    "SummaryMixin",
    "TestMixin",
    "PlotMixin",
]
