"""Data preparation and input validation, kept separate from model
mathematics.

- ``validation`` -- structural checks and input conversion for
  grouped/provider models (parallel to ``survival_validation`` for CoxPH).
- ``preparation`` -- provider screening policy (``DataPrep``,
  ``DataPrepOptions``).
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
