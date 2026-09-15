"""Newton-Raphson optimization for the Cox partial likelihood.

R's coxph uses (case-weighted) Newton-Raphson with step-halving,
converging on the relative change in log-likelihood between iterations
(`coxph.control(eps = 1e-9)` by default, `iter.max = 20`). Because the
Cox partial likelihood is concave in beta for the standard exponential
risk-score form used here, Newton-Raphson converges to the same unique
maximizer regardless of the exact step-halving schedule -- so this
implementation does not need to reproduce R's iteration-by-iteration
path, only its converged answer, which docs/R_COMPATIBILITY.md documents
matches to the tolerances found during validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizationResult:
    beta: np.ndarray
    log_likelihood: float
    score: np.ndarray
    information: np.ndarray
    n_iter: int
    converged: bool
    message: str


def newton_raphson(
    objective,
    beta0: np.ndarray,
    max_iter: int = 20,
    eps: float = 1e-9,
    max_step_halving: int = 20,
) -> OptimizationResult:
    """Maximize `objective(beta) -> (log_likelihood, score, information)`.

    `eps` is compared against the relative change in log-likelihood
    between successive accepted steps, matching R's
    `coxph.control(eps = 1e-9)` default. `max_iter` matches R's
    `iter.max = 20` default.

    Step-halving: if a full Newton step would decrease the log-likelihood
    (or produce a non-finite value), the step is halved repeatedly (up to
    `max_step_halving` times) until it doesn't -- the same safeguard
    R's coxph.fit uses against overshooting early in the iteration when
    beta0 (=0 here, matching R) is far from beta_hat.
    """
    beta = np.asarray(beta0, dtype=np.float64).copy()
    loglik, score, info = objective(beta)
    message = "converged"
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(info, score, rcond=None)[0]
            message = "information matrix was singular or near-singular at least once; used a pseudo-inverse step"

        new_beta = beta + step
        new_loglik, new_score, new_info = objective(new_beta)

        n_halving = 0
        while (not np.isfinite(new_loglik) or new_loglik < loglik) and n_halving < max_step_halving:
            step = step / 2.0
            new_beta = beta + step
            new_loglik, new_score, new_info = objective(new_beta)
            n_halving += 1

        denom = abs(loglik) if loglik != 0 else 1.0
        rel_change = abs(new_loglik - loglik) / denom

        beta, loglik, score, info = new_beta, new_loglik, new_score, new_info

        if rel_change < eps:
            converged = True
            break
    else:
        message = f"reached max_iter={max_iter} without meeting eps={eps} relative log-likelihood tolerance"

    return OptimizationResult(beta, loglik, score, info, iteration, converged, message)
