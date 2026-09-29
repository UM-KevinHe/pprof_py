"""Cluster-robust (sandwich) variance for Cox proportional hazards models.

The construction follows ``survival::coxph``:

    V_robust = V_naive @ U.T @ U @ V_naive,

where ``U`` contains score residuals collapsed within cluster.  The score
residual kernels follow the current ``survival`` C implementations
``coxscore2.c`` (right-censored data) and ``agscore3.c`` (start/stop data).

Unlike the R implementation, this module does not materialize the full
observation-by-feature score-residual matrix.  Score contributions are
aggregated into clusters during the time sweep, so memory is O(n_clusters*p).
This is important for large provider-level or subject-level datasets.

Numba is used when available for production-scale performance; a pure-Python
fallback is provided for environments where Numba is not installed.
"""
from __future__ import annotations

import warnings

import numpy as np

from ...algorithms.survival.risk_sets import njit, _HAS_NUMBA
from ...utils.numerical import safe_exp


_FEW_CLUSTERS = 30   # below this the cluster sandwich is known to understate the variance (REV-021)


@njit(cache=True)
def _score_cluster_right_numba(
    stop,
    event,
    weight,
    score,
    X,
    strata,
    cluster,
    n_clusters,
    order,
    efron,
):
    """Streaming score residuals for right-censored Cox data.

    Semantic port of survival/src/coxscore2.c, but cluster-aggregated so the
    observation-by-feature score-residual matrix is never materialized.
    """
    n, p = X.shape
    out = np.zeros((n_clusters, p))
    pos = 0
    while pos < n:
        current_stratum = strata[order[pos]]
        end = pos + 1
        while end < n and strata[order[end]] == current_stratum:
            end += 1

        ptr = end - 1
        denom = 0.0
        cumhaz = 0.0
        xhaz = np.zeros(p)
        running_x = np.zeros(p)
        while ptr >= pos:
            t = stop[order[ptr]]
            batch_end = ptr
            deaths = 0
            e_denom = 0.0
            meanwt = 0.0
            death_x = np.zeros(p)

            # coxscore2.c: initialize residuals before updating the current
            # event-time hazard, then add the rows to the risk set.
            while ptr >= pos:
                i = order[ptr]
                if stop[i] != t:
                    break
                c = cluster[i]
                w_score = weight[i] * score[i]
                denom += w_score
                for j in range(p):
                    x = X[i, j]
                    running_x[j] += w_score * x
                    out[c, j] += weight[i] * score[i] * (x * cumhaz - xhaz[j])
                if event[i] != 0:
                    deaths += 1
                    e_denom += w_score
                    meanwt += weight[i]
                    for j in range(p):
                        death_x[j] += w_score * X[i, j]
                ptr -= 1

            if deaths > 0:
                if deaths < 2 or not efron:
                    hazard = meanwt / denom
                    for j in range(p):
                        xbar = running_x[j] / denom
                        xhaz[j] += xbar * hazard
                        for q in range(ptr + 1, ptr + deaths + 1):
                            k = order[q]
                            c = cluster[k]
                            out[c, j] += weight[k] * (X[k, j] - xbar)
                    cumhaz += hazard
                else:
                    meanwt_avg = meanwt / deaths
                    for dd in range(deaths):
                        downwt = dd / deaths
                        temp = denom - downwt * e_denom
                        hazard = meanwt_avg / temp
                        cumhaz += hazard
                        for j in range(p):
                            xbar = (running_x[j] - downwt * death_x[j]) / temp
                            xhaz[j] += xbar * hazard
                            for q in range(ptr + 1, ptr + deaths + 1):
                                k = order[q]
                                c = cluster[k]
                                diff = X[k, j] - xbar
                                out[c, j] += weight[k] * (
                                    diff / deaths
                                    + diff * score[k] * hazard * downwt
                                )

        # coxscore2.c's end-of-stratum term. Apply it row-by-row so we do
        # not need an n_clusters-by-p scratch matrix for every stratum.
        for q in range(pos, end):
            i = order[q]
            c = cluster[i]
            w_score = weight[i] * score[i]
            for j in range(p):
                out[c, j] += w_score * (xhaz[j] - X[i, j] * cumhaz)

        pos = end

    return out


@njit(cache=True)
def _score_cluster_counting_numba(
    start,
    stop,
    event,
    weight,
    score,
    X,
    strata,
    cluster,
    n_clusters,
    order_stop,
    order_start,
    efron,
):
    """Streaming score residuals for (start, stop] Cox data.

    Semantic port of survival/src/agscore3.c. Each observation's score
    residual is completed when it leaves the risk set; no n-by-p residual
    matrix is materialized.
    """
    n, p = X.shape
    out = np.zeros((n_clusters, p))
    start_cursor = 0
    stop_cursor = 0
    while stop_cursor < n:
        current_stratum = strata[order_stop[stop_cursor]]
        end_stop = stop_cursor + 1
        while end_stop < n and strata[order_stop[end_stop]] == current_stratum:
            end_stop += 1

        # Locate the same stratum in start-sorted order. The cursor only moves
        # forward because both arrays are sorted by strata.
        while start_cursor < n and strata[order_start[start_cursor]] < current_stratum:
            start_cursor += 1
        start_pos = start_cursor
        end_start = start_pos
        while end_start < n and strata[order_start[end_start]] == current_stratum:
            end_start += 1

        ptr_stop = end_stop - 1
        ptr_start = end_start - 1
        denom = 0.0
        cumhaz = 0.0
        xhaz = np.zeros(p)
        running_x = np.zeros(p)

        while ptr_stop >= stop_cursor:
            dtime = stop[order_stop[ptr_stop]]

            # Finish observations whose start is at/after the current event
            # time, exactly as agscore3.c does.
            while ptr_start >= start_pos:
                k = order_start[ptr_start]
                if start[k] < dtime:
                    break
                risk = score[k] * weight[k]
                c = cluster[k]
                for j in range(p):
                    out[c, j] += weight[k] * (
                        -score[k] * (cumhaz * X[k, j] - xhaz[j])
                    )
                    running_x[j] -= risk * X[k, j]
                denom -= risk
                ptr_start -= 1

            batch_end = ptr_stop
            deaths = 0
            e_denom = 0.0
            meanwt = 0.0
            death_x = np.zeros(p)

            # Initialize residuals at this stop time and add observations to
            # the risk set.
            while ptr_stop >= stop_cursor:
                k = order_stop[ptr_stop]
                if strata[k] != current_stratum or stop[k] != dtime:
                    break
                c = cluster[k]
                risk = score[k] * weight[k]
                for j in range(p):
                    out[c, j] += weight[k] * score[k] * (
                        X[k, j] * cumhaz - xhaz[j]
                    )
                    running_x[j] += risk * X[k, j]
                denom += risk
                if event[k] != 0:
                    deaths += 1
                    e_denom += risk
                    meanwt += weight[k]
                    for j in range(p):
                        death_x[j] += risk * X[k, j]
                ptr_stop -= 1

            if deaths > 0:
                if deaths < 2:
                    hazard = meanwt / denom
                    cumhaz += hazard
                    for j in range(p):
                        xbar = running_x[j] / denom
                        xhaz[j] += xbar * hazard
                        for q in range(batch_end, ptr_stop, -1):
                            k = order_stop[q]
                            if event[k] != 0:
                                c = cluster[k]
                                out[c, j] += weight[k] * (X[k, j] - xbar)
                else:
                    meanwt_avg = meanwt / deaths
                    mh1 = np.zeros(p)
                    mh2 = np.zeros(p)
                    mh3 = np.zeros(p)
                    for dd in range(deaths):
                        downwt = dd / deaths
                        d2 = denom - downwt * e_denom
                        hazard = meanwt_avg / d2
                        cumhaz += hazard
                        for j in range(p):
                            xbar = (running_x[j] - downwt * death_x[j]) / d2
                            xhaz[j] += xbar * hazard
                            mh1[j] += hazard * downwt
                            mh2[j] += xbar * hazard * downwt
                            mh3[j] += xbar / deaths
                    for q in range(batch_end, ptr_stop, -1):
                        k = order_stop[q]
                        if event[k] != 0:
                            c = cluster[k]
                            for j in range(p):
                                out[c, j] += weight[k] * (
                                    (X[k, j] - mh3[j])
                                    + score[k] * (X[k, j] * mh1[j] - mh2[j])
                                )

        # Finish observations still active at the end of the stratum.
        while ptr_start >= start_pos:
            k = order_start[ptr_start]
            c = cluster[k]
            for j in range(p):
                out[c, j] += weight[k] * (
                    -score[k] * (cumhaz * X[k, j] - xhaz[j])
                )
            ptr_start -= 1

        start_cursor = end_start
        stop_cursor = end_stop

    return out


def _cluster_codes(cluster) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(cluster)
    if values.ndim != 1:
        raise ValueError("cluster must be one-dimensional")
    if values.size == 0:
        return np.array([], dtype=np.int64), values.copy()
    if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
        raise ValueError("cluster must not contain NaN or infinite values")
    labels, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int64, copy=False), labels


def cluster_score_residuals(
    X: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    weight: np.ndarray,
    eta: np.ndarray,
    strata: np.ndarray,
    cluster: np.ndarray,
    ties: str = "breslow",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cluster-collapsed Cox score residuals.

    The return value is ``(U, labels)`` where ``U[g]`` is the p-vector score
    residual summed over all observations belonging to cluster ``g``.

    Robust score residuals are available for Breslow and Efron ties. The
    implementation follows survival's ``Ccoxscore2`` / ``Cagscore3``
    semantics and multiplies the score residual by the case weight exactly
    once, as ``residuals.coxph(..., type='score', weighted=TRUE)`` does.
    """
    tie_name = ties if isinstance(ties, str) else getattr(ties, "name", None)
    tie_name = str(tie_name).lower()
    if tie_name not in ("breslow", "efron"):
        raise NotImplementedError(
            "Robust Cox variance is implemented for ties='breslow' and 'efron' only."
        )

    X = np.asarray(X, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    stop = np.asarray(stop, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    strata = np.asarray(strata, dtype=np.int64)
    cluster_codes, labels = _cluster_codes(cluster)

    n = X.shape[0]
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if cluster_codes.shape[0] != n:
        raise ValueError(f"cluster has length {cluster_codes.shape[0]}, expected {n}")
    for name, arr in (
        ("start", start), ("stop", stop), ("event", event),
        ("weight", weight), ("eta", eta), ("strata", strata),
    ):
        if arr.shape[0] != n:
            raise ValueError(f"{name} has length {arr.shape[0]}, expected {n}")
    if not np.all(np.isfinite(eta)):
        raise ValueError("linear predictors must be finite for robust variance")

    score = safe_exp(eta)

    # The right-censored kernel is valid for start == 0 throughout. For
    # genuine delayed-entry/counting-process data, use the agscore3 analogue.
    if not _HAS_NUMBA:
        warnings.warn(
            "Numba is not installed; robust score residuals will use a pure-Python "
            "fallback that is much slower for large datasets. Install numba for "
            "production-scale performance.",
            stacklevel=2,
        )
    if np.all(start == 0.0):
        U = _score_cluster_right_numba(
            stop, event, weight, score, X, strata,
            cluster_codes, len(labels),
            np.lexsort((-event, stop, strata)),
            tie_name == "efron",
        )
    else:
        U = _score_cluster_counting_numba(
            start, stop, event, weight, score, X, strata,
            cluster_codes, len(labels),
            np.lexsort((-event, stop, strata)),
            np.lexsort((start, strata)),
            tie_name == "efron",
        )

    if not np.all(np.isfinite(U)):
        raise FloatingPointError("Non-finite cluster score residual encountered")
    return U, labels


def robust_covariance(
    naive_covariance: np.ndarray,
    cluster_scores: np.ndarray,
) -> np.ndarray:
    """Compute the Cox sandwich covariance ``V_naive U'U V_naive``.

    ``naive_covariance`` should be the same model-based covariance already
    computed from the fitted information matrix. Reusing it avoids a second
    matrix inversion and guarantees that robust inference is built from the
    exact covariance stored as ``CoxPH.naive_covariance_``.
    """
    naive_covariance = np.asarray(naive_covariance, dtype=np.float64)
    cluster_scores = np.asarray(cluster_scores, dtype=np.float64)
    if naive_covariance.ndim != 2 or naive_covariance.shape[0] != naive_covariance.shape[1]:
        raise ValueError("naive_covariance must be a square matrix")
    p = naive_covariance.shape[0]
    if cluster_scores.ndim != 2 or cluster_scores.shape[1] != p:
        raise ValueError("cluster_scores must have shape (n_clusters, n_features)")
    if cluster_scores.shape[0] == 0:
        raise ValueError("cluster_scores must contain at least one cluster")
    if not np.all(np.isfinite(naive_covariance)):
        raise ValueError("naive_covariance must contain only finite values")
    if not np.all(np.isfinite(cluster_scores)):
        raise ValueError("cluster_scores must contain only finite values")

    if cluster_scores.shape[0] < _FEW_CLUSTERS:
        warnings.warn(f"The robust variance uses {cluster_scores.shape[0]} clusters; with fewer than {_FEW_CLUSTERS}, the "
                      "sandwich estimator can understate standard errors.", UserWarning, stacklevel=2)
    meat = cluster_scores.T @ cluster_scores
    robust = naive_covariance @ meat @ naive_covariance
    return 0.5 * (robust + robust.T)
