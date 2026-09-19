"""Package-wide exception classes for pprof_py.

This module provides the single, canonical ``NotFittedError`` used by every
model family in the package — logistic, linear, random-effect, mixed-effect,
and survival (including CoxPH).

All model families import ``NotFittedError`` from this module, so downstream
code can write one ``except pprof_py.NotFittedError`` clause that covers
every estimator in the package.
"""


class NotFittedError(RuntimeError):
    """Raised when a fitted-only attribute or method is accessed before ``fit()``.

    This is the single package-wide exception for "model not fitted yet"
    conditions.  Every model family — including CoxPH and all survival
    models — imports and raises this class, so downstream code can write
    one ``except NotFittedError`` clause that covers every estimator.

    Inherits from ``RuntimeError``, **not** from ``ValueError``, so that
    existing ``except ValueError`` clauses do not accidentally swallow it.

    Examples
    --------
    >>> from pprof_py.exceptions import NotFittedError
    >>> raise NotFittedError("Call fit() first.")
    Traceback (most recent call last):
        ...
    pprof_py.exceptions.NotFittedError: Call fit() first.
    """
