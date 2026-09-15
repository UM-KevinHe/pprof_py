"""Variable-selection meta-estimators built on top of CoxPH."""
from .selector import CoxPHSelector
from .criteria import aic, bic

__all__ = ["CoxPHSelector", "aic", "bic"]
