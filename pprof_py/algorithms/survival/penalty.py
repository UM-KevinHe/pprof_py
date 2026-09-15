"""Elastic-net penalty primitives and column standardization.

Pure NumPy, no dependence on pandas, the likelihood engine, or the
estimator layer -- this sits at the same architectural level as
`ties.py` (a numerical kernel), and is deliberately kept separate from
`coordinate_descent.py` so the penalty *math* (what glmnet calls
`pen_function`) and column scaling (`weighted_mean_sd`) can be tested
and reasoned about independently of the optimization loop that uses
them.

Every convention here is matched against glmnet 4.1-8's actual R
source (not the paper or the docs alone -- see
`docs/R_COMPATIBILITY.md`, Phase 3 section, for file/line references):

* Standardization divides each column by its *weighted population*
  standard deviation (denominator = sum(weight), not sum(weight) - 1)
  and does NOT center. `glmnet`'s `weighted_mean_sd()`
  (R/glmnetFlex.R) computes the mean only as an intermediate for the
  variance formula; the Cox family never subtracts it from `x`
  (`xm <- rep(0.0, nvars)` unconditionally in `coxpath.R`). This is
  specific to the Cox model's lack of an intercept -- see this
  package's own `CoxPH`, which centers `X` for Newton-Raphson
  conditioning but proves (utils/numerical.py) that centering never
  changes `coef_`. The same argument shows centering would be
  harmless here too, but we match glmnet's exact convention (scale
  only) so the fitted path lines up with `glmnet()` at intermediate
  lambda values, not just in the limit.
* Penalty factors are rescaled so they sum to `p` (the number of
  columns), matching glmnet's documented "penalty factors are
  internally rescaled to sum to nvars".
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def soft_threshold(z: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    """Elementwise soft-thresholding operator: sign(z) * max(|z| - thresh, 0).

    `thresh` broadcasts against `z`; both scalar and per-coordinate
    thresholds (the latter needed once `penalty_factor` varies by
    column) are supported.
    """
    return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)


def elastic_net_penalty_value(
    beta: np.ndarray, alpha: float, penalty_factor: np.ndarray
) -> float:
    """sum_j penalty_factor_j * [alpha*|beta_j| + 0.5*(1-alpha)*beta_j^2].

    This is glmnet's `pen_function` (R/glmnetFlex.R): no dependence on
    lambda -- the caller multiplies by lambda separately, since the
    same penalty *shape* is reused across the whole regularization
    path and inside the per-lambda coordinate-descent objective.
    """
    return float(np.sum(penalty_factor * (alpha * np.abs(beta) + 0.5 * (1.0 - alpha) * beta**2)))


def weighted_column_scale(
    X: np.ndarray, weight: np.ndarray, standardize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column scale factors for penalized fitting.

    Returns
    -------
    xs : ndarray, shape (p,)
        Divisor applied to each column of X before fitting (weighted
        population standard deviation if `standardize`, else 1.0 for
        every column). Coefficients are divided by `xs` at the end to
        report them in the original units of X (see coordinate_descent.py).
    degenerate : ndarray of bool, shape (p,)
        True for columns whose weighted variance is numerically zero
        (matches glmnet's `xv[xv < 10*eps] <- 0` safeguard). These
        columns cannot be standardized meaningfully; callers must
        exclude them from penalized fitting (forced coefficient 0)
        rather than divide by a near-zero scale.
    """
    w = np.asarray(weight, dtype=np.float64)
    w_sum = w.sum()
    w_norm = w / w_sum
    xm = w_norm @ X
    centered = X - xm
    xv = w_norm @ (centered**2)
    degenerate = xv < 10.0 * np.finfo(np.float64).eps
    xv = np.where(degenerate, 0.0, xv)
    if standardize:
        xs = np.sqrt(xv)
    else:
        xs = np.ones(X.shape[1])
    # Degenerate columns get xs=1 (never divide by ~0); the caller is
    # responsible for excluding them from the penalized fit entirely.
    xs = np.where(degenerate, 1.0, xs)
    return xs, degenerate


def rescale_penalty_factors(penalty_factor: Optional[np.ndarray], p: int) -> np.ndarray:
    """Validate and rescale user-supplied penalty factors to sum to `p`.

    `penalty_factor=None` gives the all-ones default (every variable
    penalized equally). A 0 marks an always-unpenalized variable and
    is preserved exactly (never rescaled away from 0) -- this is the
    mechanism for "clearly defined penalized vs. unpenalized
    variables" the estimator exposes. Matches glmnet's documented
    "penalty factors are internally rescaled to sum to nvars"; unlike
    glmnet we do not silently treat negative factors as 0 -- a
    negative penalty factor is almost always a user error (it would
    make the penalty *encourage* a nonzero coefficient), so it raises.
    """
    if penalty_factor is None:
        return np.ones(p)
    pf = np.asarray(penalty_factor, dtype=np.float64)
    if pf.shape != (p,):
        raise ValueError(f"penalty_factor must have shape ({p},), got {pf.shape}")
    if np.any(pf < 0):
        raise ValueError("penalty_factor must be non-negative")
    total = pf.sum()
    if total <= 0:
        raise ValueError("penalty_factor cannot be all-zero (nothing would be penalized; "
                          "use an unpenalized CoxPH fit instead)")
    return pf * (p / total)
