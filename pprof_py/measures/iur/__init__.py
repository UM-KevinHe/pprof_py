"""Inter-Unit Reliability (IUR) estimation.

Three estimators are provided, each implementing the sklearn
``BaseEstimator`` interface (``__init__`` → ``.fit()`` → fitted
attributes with trailing underscore):

* :class:`BootstrapIUR` — stratified bootstrap variance-decomposition.
* :class:`SplitHalfIUR` — split-half correlation-based reliability.
* :class:`DirectIUR` — direct computation from pre-estimated SEs.

A built-in measure function :func:`ratio_measure` computes
``sum(obs) / sum(exp)`` per group; custom callables with the same
signature may be substituted via the ``measure_fn`` parameter.
"""
from .bootstrap import BootstrapIUR
from .direct import DirectIUR
from .measures import ratio_measure
from .split_half import SplitHalfIUR

__all__ = [
    "BootstrapIUR",
    "SplitHalfIUR",
    "DirectIUR",
    "ratio_measure",
]
