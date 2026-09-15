"""Package-wide exception classes for pprof_py.

This module provides the single, canonical ``NotFittedError`` used by every
model family in the package (logistic, linear, random-effect, mixed-effect).

CoxPH retains its own textually-identical ``NotFittedError`` defined in
``models/survival/coxph.py`` — this is a deliberate, documented choice
(see REFACTORING_PLAN.md § 6) to leave CoxPH untouched.  A future,
unrelated CoxPH change may swap its local class for this one at zero
marginal risk; it is not worth a CoxPH-specific commit on its own.

Tier 3 of the refactoring plan (foundational contracts) introduced this
module to consolidate the two pre-existing ``NotFittedError`` classes
(``models/base.py`` and ``models/survival/coxph.py``) and the stray
``ValueError`` in ``LogisticMixedEffectModel._check_fitted`` into one
importable, catchable exception type.
"""


class NotFittedError(RuntimeError):
    """Raised when a fitted-only attribute or method is accessed before ``fit()``.

    This is the single package-wide exception for "model not fitted yet"
    conditions.  Every model family (except CoxPH, which keeps its own
    identical class — see module docstring) should raise this, so that
    downstream code can write one ``except NotFittedError`` clause that
    covers every estimator in the package.

    Inherits from ``RuntimeError`` (matching the pre-existing convention
    in both ``models/base.py`` and ``models/survival/coxph.py``), **not**
    from ``ValueError``, so that existing ``except ValueError`` clauses do
    not accidentally swallow it.

    Examples
    --------
    >>> from pprof_py.exceptions import NotFittedError
    >>> raise NotFittedError("Call fit() first.")
    Traceback (most recent call last):
        ...
    pprof_py.exceptions.NotFittedError: Call fit() first.
    """
