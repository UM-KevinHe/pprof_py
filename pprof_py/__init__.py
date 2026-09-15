"""pprof_py: a general statistical-computing package for provider profiling
and healthcare analytics.

Public API
----------
    from pprof_py import LogisticFixedEffectModel, LogisticMixedEffectModel, LogisticRandomEffectModel
    from pprof_py import LinearFixedEffectModel, LinearRandomEffectModel
    from pprof_py import CoxPH, PenalizedCoxPH, PenalizedCoxPHCV
    from pprof_py import CauseSpecificCoxPH, FineGrayPH
    from pprof_py import CoxPHSelector

Internal implementation (algorithms, data handling, inference, plotting) is
organized under `pprof_py.models`, `pprof_py.algorithms`, `pprof_py.data`,
`pprof_py.inference`, `pprof_py.measures`, and `pprof_py.plotting`; analysts
are expected to import model classes from the package root rather than from
those internal modules directly.

`LinearRandomEffectModel` is now re-exported from the package root.
"""
from .models.linear import LinearFixedEffectModel, LinearRandomEffectModel
from .models.logistic import (
    LogisticFixedEffectModel,
    LogisticMixedEffectModel,
    LogisticRandomEffectModel,
)
from .models.survival import (
    CoxPH,
    PenalizedCoxPH,
    PenalizedCoxPHCV,
    CauseSpecificCoxPH,
    FineGrayPH,
)
from .selection import CoxPHSelector
from .exceptions import NotFittedError
from .utils import setup_logger, proc_freq, sigmoid
from .plotting import plot_caterpillar
from .inference import huber_location_scale, estimate_empirical_null

__version__ = "0.2.0"

__all__ = [
    "LinearFixedEffectModel",
    "LinearRandomEffectModel",
    "LogisticFixedEffectModel",
    "LogisticMixedEffectModel",
    "LogisticRandomEffectModel",
    "CoxPH",
    "PenalizedCoxPH",
    "PenalizedCoxPHCV",
    "CauseSpecificCoxPH",
    "FineGrayPH",
    "CoxPHSelector",
    "NotFittedError",
    "setup_logger",
    "proc_freq",
    "sigmoid",
    "plot_caterpillar",
    "huber_location_scale",
    "estimate_empirical_null",
    "__version__",
]
