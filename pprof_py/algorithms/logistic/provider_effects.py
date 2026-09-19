"""Logistic-specific provider-effect Newton step with median-clamp bounding.

The two-layer architecture:
  - Provider effects gamma_k (unpenalized) are updated via Newton step
    with median-clamp bounding to prevent extreme provider estimates.
  - Covariate effects beta (penalized) are updated via standard
    proximal-Newton + CD.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from .likelihood import _safe_expit

logger = logging.getLogger(__name__)


def compute_provider_indices(
    provider_id: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Map provider IDs to contiguous 0..K-1 indices.

    Parameters
    ----------
    provider_id : ndarray, shape (n,)
        Provider identifiers (arbitrary hashable, typically int or str).

    Returns
    -------
    provider_idx : ndarray of int, shape (n,)
        Contiguous provider index per observation (0..K-1).
    provider_labels : ndarray, shape (K,)
        Unique provider labels (original IDs).
    n_providers : int
        Number of unique providers.
    """
    provider_labels, provider_idx = np.unique(
        provider_id, return_inverse=True,
    )
    return provider_idx, provider_labels, len(provider_labels)


def logistic_provider_score_info(
    y: np.ndarray,
    eta: np.ndarray,
    weight: np.ndarray,
    provider_idx: np.ndarray,
    n_providers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-provider score and information for the logistic model.

    For provider k:
        score_k = sum_{i: prov(i)=k} w_i * (y_i - p_i)
        info_k  = sum_{i: prov(i)=k} w_i * p_i * (1 - p_i)

    Parameters
    ----------
    y : ndarray, shape (n,)
        Binary outcome.
    eta : ndarray, shape (n,)
        Linear predictor (X @ beta + gamma[provider_idx] + offset).
    weight : ndarray, shape (n,)
        Per-observation weights.
    provider_idx : ndarray of int, shape (n,)
        Provider index per observation.
    n_providers : int

    Returns
    -------
    score : ndarray, shape (K,)
        Score per provider.
    info : ndarray, shape (K,)
        Information (diagonal) per provider.
    """
    p = _safe_expit(eta)
    resid = weight * (y - p)
    w_var = weight * p * (1.0 - p)

    score = np.zeros(n_providers, dtype=np.float64)
    info = np.zeros(n_providers, dtype=np.float64)
    np.add.at(score, provider_idx, resid)
    np.add.at(info, provider_idx, w_var)

    return score, info


def provider_bound_clamp(
    gamma: np.ndarray,
    bound: float,
) -> np.ndarray:
    """Median-clamp bounding for provider effects.

    Clamps each gamma_k to within ``bound`` of the median.  This
    prevents extreme provider effects that can arise with small
    providers (few observations).  Matches grplasso's
    ``provider_bound_clamp`` and the ``gamma_bound`` parameter
    in ``pp.lasso``.

    Parameters
    ----------
    gamma : ndarray, shape (K,)
        Provider effects.
    bound : float
        Maximum deviation from median.

    Returns
    -------
    ndarray, shape (K,)
        Bounded provider effects.
    """
    median_gamma = float(np.median(gamma))
    return np.clip(gamma, median_gamma - bound, median_gamma + bound)


def logistic_provider_newton_step(
    y: np.ndarray,
    eta: np.ndarray,
    weight: np.ndarray,
    provider_idx: np.ndarray,
    n_providers: int,
    gamma: np.ndarray,
    gamma_bound: float = 10.0,
    damping: float = 1.0,
) -> np.ndarray:
    """One Newton step for provider effects with median-clamp bounding.

    gamma_k^{new} = gamma_k^{old} + damping * score_k / info_k

    followed by median-clamp bounding.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Binary outcome.
    eta : ndarray, shape (n,)
        Current linear predictor.
    weight : ndarray, shape (n,)
    provider_idx : ndarray of int, shape (n,)
    n_providers : int
    gamma : ndarray, shape (K,)
        Current provider effects.
    gamma_bound : float
        Maximum deviation from median.
    damping : float
        Step-size damping (1.0 = full Newton step).

    Returns
    -------
    ndarray, shape (K,)
        Updated provider effects.
    """
    score, info = logistic_provider_score_info(
        y, eta, weight, provider_idx, n_providers,
    )

    # Newton step with safeguard for near-zero info.
    info_safe = np.maximum(info, 1e-12)
    gamma_new = gamma + damping * score / info_safe

    # Median-clamp bounding.
    gamma_new = provider_bound_clamp(gamma_new, gamma_bound)

    return gamma_new


def logistic_provider_deviance(
    y: np.ndarray,
    eta: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Weighted binomial deviance."""
    p = _safe_expit(eta)
    eps = np.finfo(np.float64).tiny
    ll = np.sum(
        weight * (
            y * np.log(np.maximum(p, eps))
            + (1.0 - y) * np.log(np.maximum(1.0 - p, eps))
        )
    )
    return float(-2.0 * ll)
