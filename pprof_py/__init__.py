"""pprof_py: a general statistical-computing package for provider profiling
and healthcare analytics.

Public API
----------
    from pprof_py import LogisticFixedEffectModel, LogisticFERandomClusterModel, LogisticRandomEffectModel
    from pprof_py import LinearFixedEffectModel, LinearRandomEffectModel
    from pprof_py import CoxPH, PenalizedCoxPH, PenalizedCoxPHCV
    from pprof_py import GroupLassoCoxPH, GroupLassoCoxPHCV, ProviderPenalizedCoxPH
    from pprof_py import DiscreteSurvival, DiscreteSurvivalCV
    from pprof_py import ProviderPenalizedDiscreteSurvival, ProviderPenalizedDiscreteSurvivalCV
    from pprof_py import FrailtyCoxPH, TimeVaryingCoxPH
    from pprof_py import CauseSpecificCoxPH, FineGrayPH
    from pprof_py import CoxPHSelector

Internal implementation (algorithms, data handling, inference, plotting) is
organized under `pprof_py.models`, `pprof_py.algorithms`, `pprof_py.data`,
`pprof_py.inference`, `pprof_py.measures`, and `pprof_py.plotting`; analysts
are expected to import model classes from the package root rather than from
those internal modules directly.

`LinearRandomEffectModel` is now re-exported from the package root.
"""
from .models.linear import (
    LinearFixedEffectModel,
    LinearRandomEffectModel,
    PenalizedLinear,
    PenalizedLinearCV,
    GroupLassoLinear,
)
from .models.logistic import (
    LogisticFixedEffectModel,
    LogisticFERandomClusterModel,
    LogisticRandomEffectModel,
    LogisticThreeStageModel,
    PenalizedLogistic,
    PenalizedLogisticCV,
    GroupLassoLogistic,
    GroupLassoLogisticCV,
    ProviderPenalizedLogistic,
    ProviderPenalizedLogisticCV,
)
from .models.survival import (
    CoxPH,
    PenalizedCoxPH,
    PenalizedCoxPHCV,
    GroupLassoCoxPH,
    GroupLassoCoxPHCV,
    ProviderPenalizedCoxPH,
    DiscreteSurvival,
    DiscreteSurvivalCV,
    ProviderPenalizedDiscreteSurvival,
    ProviderPenalizedDiscreteSurvivalCV,
    FrailtyCoxPH,
    TimeVaryingCoxPH,
    CauseSpecificCoxPH,
    FineGrayPH,
)
from .selection import CoxPHSelector
from .exceptions import NotFittedError
from .utils import setup_logger, proc_freq, sigmoid
from .plotting import plot_caterpillar

__version__ = "0.4.1"

__all__ = [
    # Linear
    "LinearFixedEffectModel",
    "LinearRandomEffectModel",
    "PenalizedLinear",
    "PenalizedLinearCV",
    "GroupLassoLinear",
    # Logistic
    "LogisticFixedEffectModel",
    "LogisticFERandomClusterModel",
    "LogisticThreeStageModel",
    "LogisticRandomEffectModel",
    "PenalizedLogistic",
    "PenalizedLogisticCV",
    "GroupLassoLogistic",
    "GroupLassoLogisticCV",
    "ProviderPenalizedLogistic",
    "ProviderPenalizedLogisticCV",
    # Survival
    "CoxPH",
    "PenalizedCoxPH",
    "PenalizedCoxPHCV",
    "GroupLassoCoxPH",
    "GroupLassoCoxPHCV",
    "ProviderPenalizedCoxPH",
    "DiscreteSurvival",
    "DiscreteSurvivalCV",
    "ProviderPenalizedDiscreteSurvival",
    "ProviderPenalizedDiscreteSurvivalCV",
    "FrailtyCoxPH",
    "TimeVaryingCoxPH",
    "CauseSpecificCoxPH",
    "FineGrayPH",
    # Selection
    "CoxPHSelector",
    # Utilities
    "NotFittedError",
    "setup_logger",
    "proc_freq",
    "sigmoid",
    "plot_caterpillar",
    "__version__",
]
