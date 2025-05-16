# pprof_py/__init__.py
# """
# pprof_py: A package that provides a variety of risk-adjusted models for provider profiling, efficiently handling large-scale provider data.
# """
__version__ = "0.1.0"
#
# # Import key functions to make them available directly
# # e.g., from pprof_py import linear_fixed_effect, linear_random_effect
from .linear_fixed_effect import LinearFixedEffectModel  
from .linear_random_effect import LinearRandomEffectModel
from .logistic_fixed_effect import LogisticFixedEffectModel
from .logistic_random_effect import LogisticRandomEffectModel
from .utils import setup_logger, proc_freq, sigmoid  # Importing all utility functions
from .plotting import plot_caterpillar
# # Optionally control what `from pprof_py import *` imports
__all__ = ['LinearFixedEffectModel', 'LinearRandomEffectModel', 
           'LogisticFixedEffectModel', 'LogisticRandomEffectModel',
            'setup_logger', 'proc_freq', 'sigmoid', 'plot_caterpillar',
            "__version__"
  ]
# # __all__ is a convention in Python that defines what symbols will be exported when
# # `from module import *` is used. It helps in controlling the public API of the module. 