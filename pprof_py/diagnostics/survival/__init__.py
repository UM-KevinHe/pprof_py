"""Tooling for validating CoxPH against data it wasn't tested against
here -- especially real production data.

    from pprof_py.diagnostics.survival import preflight_report
    from pprof_py.diagnostics.survival.validate_against_r import run_validation

See `validate_against_r.py`'s module docstring for the JSON spec format
and a full usage example.
"""
from .preflight import preflight_report, PreflightResult

__all__ = ["preflight_report", "PreflightResult"]
