"""Data preparation and input validation, kept separate from model
mathematics.

- ``validation`` -- structural checks and input conversion for
  grouped/provider models (parallel to ``survival_validation`` for CoxPH).
- ``preparation`` -- provider screening policy (``DataPrep``,
  ``DataPrepOptions``).
- ``glmm_prep`` -- data preparation for the three-stage model
  (``glmm_data_prep``, R's ``glmm.data.prep``).
- ``survival_validation`` / ``survival_data`` -- CoxPH's internal
  representation.
"""
from .validation import (  # noqa: F401
    check_missingness,
    check_variation,
    check_correlation,
    check_vif,
    run_structural_checks,
    ValidatedInputs,
    validate_and_convert_inputs,
)
from .preparation import DataPrep, DataPrepOptions  # noqa: F401
from .glmm_prep import GLMMPreparedData, glmm_data_prep  # noqa: F401
