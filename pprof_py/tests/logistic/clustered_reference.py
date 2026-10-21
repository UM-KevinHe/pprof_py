"""Reference implementation of pprofR's patient-clustered provider tests, for the test suite only.

A plain NumPy transcription of the definitions in pprofR 0.2.0 (``R/test_fe.R``), written for clarity
rather than speed and independent of pprof_py's own code. ``test_pprofr_clustered_reference.py`` checks it
against R's values from ``r_reference/run_pprofr_clustered.R``; later rounds test the package against it.

Rows are used in the order given, and within a cluster that order is the order of the patient's rows. A
cluster is a (provider, patient) pair, so a patient treated at two providers forms two clusters. With row
residuals ``r = Y - N p`` and variances ``v = N p (1 - p)``:

* ``U_i`` is the sum of ``r`` over provider i's rows and ``I_i`` the sum of ``v``;
* robust: ``V_i`` is the sum over i's clusters k of ``(sum of r over k)^2``;
* stabilized: ``Vs_i`` is the sum over i's clusters of ``a_k' R a_k`` with ``a = sqrt(v)``, where
  ``R[s, t]`` averages ``e_s e_t`` (``e = r / sqrt(v)``) over the clusters with at least ``max(s, t)``
  rows, pooled over every cluster of every provider and not centred. Rows with ``v = 0`` get
  ``e = a = 0`` but still count as rows.

The Wald tests use the fitted p, ``SE = sqrt(V_i) / I_i`` or ``sqrt(Vs_i) / I_i`` (beta treated as known)
and ``z = (gamma_i - gamma0) / SE``. The score tests use ``p = expit(gamma0 + Z beta)`` for every
provider, residuals and ``R`` included: ``z = U_i / sqrt(V_i)`` or ``U_i / sqrt(Vs_i)``. The plain
(beta-fixed) score test is ``U_i / sqrt(I_i)``. ``gamma0`` defaults to the median of gamma-hat.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.special import expit
from scipy.stats import norm

TESTS = ("robust_wald", "stabilized_wald", "robust_score", "stabilized_score", "score")


@dataclass(frozen=True)
class Layout:
    """Where each row sits: its provider, its (provider, patient) cluster and its position in the cluster."""

    providers: np.ndarray         # provider labels, sorted
    provider: np.ndarray          # provider of each row, as an index into ``providers``
    cluster: np.ndarray           # cluster of each row, numbered by first appearance
    cluster_provider: np.ndarray  # provider of each cluster
    position: np.ndarray          # 0-based position of each row within its cluster, in row order
    members: Tuple[np.ndarray, ...]  # the rows of each cluster, in row order


def layout(provider, patient) -> Layout:
    provider = np.asarray(provider).astype(str)
    patient = np.asarray(patient).astype(str)
    providers, provider_code = np.unique(provider, return_inverse=True)
    provider_code = provider_code.reshape(-1)
    codes, members, cluster_provider = {}, [], []
    cluster = np.empty(len(provider), dtype=np.intp)
    position = np.empty(len(provider), dtype=np.intp)
    for j, key in enumerate(zip(provider, patient)):
        c = codes.setdefault(key, len(codes))
        if c == len(members):
            members.append([])
            cluster_provider.append(provider_code[j])
        position[j] = len(members[c])
        members[c].append(j)
        cluster[j] = c
    return Layout(providers, provider_code, cluster, np.asarray(cluster_provider, dtype=np.intp), position,
                  tuple(np.asarray(m, dtype=np.intp) for m in members))


def standardized(r, v):
    """``e = r / sqrt(v)`` and ``a = sqrt(v)``, both 0 where ``v = 0``."""
    positive = v > 0
    a = np.where(positive, np.sqrt(np.where(positive, v, 1.0)), 0.0)
    e = np.where(positive, r / np.where(positive, a, 1.0), 0.0)
    return e, a


def pooled_rhat(e, lay: Layout):
    """The pooled matrix ``R`` and ``m``, where ``m[q]`` counts the clusters with more than ``q`` rows."""
    dim = max(len(rows) for rows in lay.members)
    S = np.zeros((dim, dim))
    m = np.zeros(dim)
    for rows in lay.members:
        s = len(rows)
        S[:s, :s] += np.outer(e[rows], e[rows])
        m[:s] += 1
    q = np.maximum.outer(np.arange(dim), np.arange(dim))
    return S / m[q], m


def meats(r, v, lay: Layout, rhat: Optional[np.ndarray] = None) -> dict:
    """Per provider (in ``lay.providers`` order): ``U``, ``I``, ``V`` and ``Vs``, plus ``R`` and ``m``."""
    k = len(lay.providers)
    e, a = standardized(r, v)
    m = None
    if rhat is None:
        rhat, m = pooled_rhat(e, lay)
    cluster_sum = np.bincount(lay.cluster, weights=r, minlength=len(lay.members))
    W = np.array([a[rows] @ rhat[:len(rows), :len(rows)] @ a[rows] for rows in lay.members])
    return dict(U=np.bincount(lay.provider, weights=r, minlength=k),
                I=np.bincount(lay.provider, weights=v, minlength=k),
                V=np.bincount(lay.cluster_provider, weights=cluster_sum ** 2, minlength=k),
                Vs=np.bincount(lay.cluster_provider, weights=W, minlength=k), R=rhat, m=m)


def linear_predictor(Z, beta):
    """``Z beta``, summed column by column (the order of R's reference BLAS)."""
    Z = np.asarray(Z, dtype=float)
    out = np.zeros(Z.shape[0])
    for j, b in enumerate(np.asarray(beta, dtype=float)):
        out = out + Z[:, j] * b
    return out


def two_sided(z, level: float = 0.95):
    """Two-sided p-values and flags as pprofR's ``z_test_output()``: 1 higher, -1 lower, 0 as expected."""
    z = np.asarray(z, dtype=float)
    upper = norm.sf(z)
    alpha = 1 - level
    flag = np.where(upper < alpha / 2, 1.0, np.where(upper <= 1 - alpha / 2, 0.0, -1.0))
    return 2 * np.minimum(upper, 1 - upper), np.where(np.isnan(z), np.nan, flag)


def clustered_tests(y, n, Z, provider, patient, gamma, beta, gamma0: Optional[float] = None, *,
                    lay: Optional[Layout] = None, rhat: Optional[dict] = None, level: float = 0.95) -> dict:
    """The four clustered tests and the plain score test, as pprofR's ``test_fe()`` computes them.

    ``gamma`` holds the provider effects in sorted-label order, ``beta`` the effects of ``Z``'s columns.
    ``rhat`` optionally replaces the pooled matrices: ``{"null": matrix, "fit": matrix}``. Returns, per
    provider, ``<test>_stat``, ``<test>_p`` and ``<test>_flag``, ``<test>_se`` for the Wald tests, the robust
    Wald interval, and the ingredients at the null and at the fit.
    """
    lay = layout(provider, patient) if lay is None else lay
    y, n = np.asarray(y, dtype=float), np.asarray(n, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    g0 = float(np.median(gamma)) if gamma0 is None else float(gamma0)
    xbeta = linear_predictor(Z, beta)
    p_fit = expit(gamma[lay.provider] + xbeta)
    p_null = expit(g0 + xbeta)
    rhat = rhat or {}
    fit = meats(y - n * p_fit, n * p_fit * (1 - p_fit), lay, rhat.get("fit"))
    null = meats(y - n * p_null, n * p_null * (1 - p_null), lay, rhat.get("null"))
    out = dict(gamma0=g0, fit=fit, null=null,
               robust_wald_se=np.sqrt(fit["V"]) / fit["I"], stabilized_wald_se=np.sqrt(fit["Vs"]) / fit["I"])
    out["robust_wald_stat"] = (gamma - g0) / out["robust_wald_se"]
    out["stabilized_wald_stat"] = (gamma - g0) / out["stabilized_wald_se"]
    out["robust_score_stat"] = null["U"] / np.sqrt(null["V"])
    out["stabilized_score_stat"] = null["U"] / np.sqrt(null["Vs"])
    out["score_stat"] = null["U"] / np.sqrt(null["I"])
    for test in TESTS:
        out[f"{test}_p"], out[f"{test}_flag"] = two_sided(out[f"{test}_stat"], level)
    q = norm.ppf(1 - (1 - level) / 2)
    out["robust_wald_ci_lower"] = gamma - q * out["robust_wald_se"]
    out["robust_wald_ci_upper"] = gamma + q * out["robust_wald_se"]
    return out
