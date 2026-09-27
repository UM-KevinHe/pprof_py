"""Indirect and direct standardized ratios for the Cox model (SMR, SHR).

Definitions follow "Tutorial for Standardized Measures of Survival Outcomes: Illustrating the Standardized
Mortality Ratio for Dialysis Facilities" (Section 3).  For patient (or counting-process row) ``i`` with entry
time ``B_i`` (0 without left truncation), exit time ``X_i``, event indicator ``delta_i``, provider ``j`` and
fitted risk score ``exp(eta_i)``, ``eta_i = Z_i beta_hat + offset_i``:

* the national baseline ``Lambda_0`` is the Breslow estimator over all rows with ``eta`` as offset.  With a
  provider-stratified fit (``strata=provider``) this is He and Schaubel's (2014) two-stage estimate; with an
  unstratified fit it is the pooled model's baseline;
* provider ``j``'s baseline ``Lambda_0j`` is the Breslow estimator over provider ``j``'s rows at the same
  ``beta_hat`` -- the stratified model's baseline for that provider;
* indirect: ``O_j / E_j`` with ``O_j`` provider j's events and ``E_j = ````sum_````{i in j} exp(eta_i) [Lambda_0(X_i) -
  Lambda_0(B_i)]``, its expected events at the national baseline;
* direct: ``E^(j) / O`` with ``O`` the total events and ``E^(j) = ````sum_````{all i} exp(eta_i) [Lambda_0j(X_i) -
  Lambda_0j(B_i)]``, the events expected if every patient had provider j's baseline.

Because each Breslow increment at an event time ``t`` of provider ``j`` is ``1 / RS_j(t)`` per event, ``E^(j)``
equals the sum over provider j's events of ``RS(t) / RS_j(t)``, the ratio of the population's to the provider's
risk-set sums of ``exp(eta)`` at the event time, which needs one pass over the events.  And the national
expected events add up to the observed ones, ``sum_j E_j = O``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...data.survival_validation import validate_fit_inputs

_STDZ = ("indirect", "direct")


def _suffix_sums(keys: np.ndarray, weights: np.ndarray):
    """Sorted ``keys`` and ``S[k] = sum of weights with key >= keys_sorted[k]`` (``S[n] = 0``)."""
    order = np.argsort(keys, kind="mergesort")
    suffix = np.concatenate([np.cumsum(weights[order][::-1])[::-1], [0.0]])
    return keys[order], suffix


def _at_risk_sums(query, entry, exit_, weights):
    """``sum_i weights_i * 1(entry_i < q <= exit_i)`` for each ``q`` in ``query``."""
    ex, s_ex = _suffix_sums(exit_, weights)
    en, s_en = _suffix_sums(entry, weights)
    return s_ex[np.searchsorted(ex, query, side="left")] - s_en[np.searchsorted(en, query, side="left")]


def cox_standardized_expectations(eta, entry, exit_, event, provider_codes, n_providers):
    """Observed and expected events per provider for the indirect and direct standardized ratios.

    Parameters
    ----------
    eta : ndarray, shape (n,)
        Fitted linear predictor ``Z beta_hat + offset``.
    entry, ``exit_`` : ndarray, shape (n,)
        At-risk interval ``(entry, exit]`` of each row.
    event : ndarray, shape (n,)
        Event indicator (0/1) at ``exit_``.
    provider_codes : ndarray of int, shape (n,)
        Provider of each row, coded ``0 .. n_providers - 1``.
    n_providers : int

    Returns
    -------
    dict of ndarray, shape (n_providers,)
        ``observed`` (O_j), ``indirect_expected`` (E_j), ``direct_expected`` (E^(j)) and ``person_time``.
    """
    eta = np.asarray(eta, dtype=np.float64)
    entry = np.asarray(entry, dtype=np.float64)
    exit_ = np.asarray(exit_, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    codes = np.asarray(provider_codes, dtype=np.int64)
    # Every quantity below is invariant to a common factor in exp(eta).
    w = np.exp(eta - np.max(eta))
    is_event = event > 0

    # National Breslow baseline at the distinct event times.
    times, d = np.unique(exit_[is_event], return_counts=True)
    rs_pop_times = _at_risk_sums(times, entry, exit_, w)
    cum_hazard = np.cumsum(d / rs_pop_times)

    def national(t):
        k = np.searchsorted(times, t, side="right") - 1
        return np.where(k >= 0, cum_hazard[np.clip(k, 0, None)], 0.0)

    indirect_rows = w * (national(exit_) - national(entry))

    # Provider risk-set sums at each event row's own time: composite integer keys (provider, time rank).
    grid = np.unique(np.concatenate([entry, exit_]))
    span = grid.size + 1
    key = lambda c, t: c * span + np.searchsorted(grid, t)           # noqa: E731
    ev_codes, ev_times = codes[is_event], exit_[is_event]
    kx, sx = _suffix_sums(key(codes, exit_), w)
    kb, sb = _suffix_sums(key(codes, entry), w)
    lo, hi = key(ev_codes, ev_times), (ev_codes + 1) * span
    rs_provider = ((sx[np.searchsorted(kx, lo)] - sx[np.searchsorted(kx, hi)])
                   - (sb[np.searchsorted(kb, lo)] - sb[np.searchsorted(kb, hi)]))
    rs_pop_events = _at_risk_sums(ev_times, entry, exit_, w)

    return {
        "observed": np.bincount(codes, weights=event, minlength=n_providers),
        "indirect_expected": np.bincount(codes, weights=indirect_rows, minlength=n_providers),
        "direct_expected": np.bincount(ev_codes, weights=rs_pop_events / rs_provider, minlength=n_providers),
        "person_time": np.bincount(codes, weights=exit_ - entry, minlength=n_providers),
    }


class CoxPHMeasuresMixin:
    """Standardized measures for ``CoxPH``."""

    def calculate_standardized_measures(
        self,
        X,
        duration=None,
        event=None,
        start=None,
        stop=None,
        *,
        provider_id,
        offset=None,
        providers=None,
        stdz="indirect",
    ) -> dict:
        """Indirect and direct standardized ratios (SMR, SHR) per provider.

        Fit the model with ``strata=provider`` for He and Schaubel's two-stage approach (the provider's own
        baseline absorbs its effect, and the national baseline is the Breslow estimator with the fitted
        covariate effects as offset), or without strata for the pooled model; then pass the same data here.
        Baselines are Breslow estimators, as in the tutorial, whatever ``ties`` fit the coefficients (with no
        tied event times the Breslow and Efron baselines coincide).  Each row counts once: sample weights are
        not used.

        Parameters
        ----------
        X, duration, event, start, stop, offset
            The data the model was fitted to (as in ``fit``).
        provider_id : array-like, shape (n,)
            Provider of each row.
        providers : array-like, optional
            Providers to report (default: all).  The baselines always use every row.
        stdz : {"indirect", "direct"} or list of them, default "indirect"
            ``"indirect"``: ``O_j / E_j``, provider j's observed events over those expected for its patients at
            the national baseline.  ``"direct"``: ``E^(j) / O``, the events expected if every patient had
            provider j's baseline, over the total observed; for either, above 1 means more events than the
            national norm.

        Returns
        -------
        dict of DataFrame
            ``"indirect"``: ``provider_id``, ``indirect_ratio``, ``observed``, ``expected``, ``person_time``
            (time at risk, the grouping variable of the tutorial's empirical null).  ``"direct"``:
            ``provider_id``, ``direct_ratio``, ``observed`` (the total O), ``expected`` (E^(j)), ``n_pop``.
        """
        self._check_is_fitted()
        wanted = [stdz] if isinstance(stdz, str) else list(stdz)
        if not wanted or any(s not in _STDZ for s in wanted):
            raise ValueError(f"stdz must be 'indirect', 'direct' or a list of them, got {stdz!r}")
        clean = validate_fit_inputs(X, duration=duration, event=event, start=start, stop=stop, offset=offset)
        X_arr = clean["X"]
        if getattr(self, "fit_intercept", False):
            X_arr = np.column_stack([X_arr, np.ones(X_arr.shape[0])])
        prov = np.asarray(provider_id)
        if prov.shape != (X_arr.shape[0],):
            raise ValueError(f"provider_id must have one entry per row ({X_arr.shape[0]}), got shape {prov.shape}")
        labels, codes = np.unique(prov, return_inverse=True)
        eta = X_arr @ self.coef_ + clean["offset"]
        exp_ = cox_standardized_expectations(eta, clean["start"], clean["stop"], clean["event"], codes, labels.size)
        keep = np.ones(labels.size, dtype=bool) if providers is None else np.isin(labels, np.atleast_1d(providers))
        out = {}
        if "indirect" in wanted:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = exp_["observed"] / exp_["indirect_expected"]
            out["indirect"] = pd.DataFrame({
                "provider_id": labels, "indirect_ratio": ratio, "observed": exp_["observed"],
                "expected": exp_["indirect_expected"], "person_time": exp_["person_time"],
            })[keep].reset_index(drop=True)
        if "direct" in wanted:
            total = float(np.sum(clean["event"]))
            out["direct"] = pd.DataFrame({
                "provider_id": labels, "direct_ratio": exp_["direct_expected"] / total,
                "observed": np.full(labels.size, total), "expected": exp_["direct_expected"],
                "n_pop": np.full(labels.size, X_arr.shape[0]),
            })[keep].reset_index(drop=True)
        return out
