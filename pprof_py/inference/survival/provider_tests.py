"""Provider tests for the Cox model's indirect standardized ratios (SMR, SHR).

Implements the inference of "Tutorial for Standardized Measures of Survival Outcomes" (Section 4) for the ratios
``O_j / E_j`` of :meth:`CoxPH.calculate_standardized_measures`, with ``O_j`` treated as Poisson with mean ``E_j``:

* ``test_method="midp"``: the two-sided mid-p test (Eq. 10), turned into a z-statistic (Eq. 14) that the null
  model calibrates -- the theoretical N(0, 1) or an empirical null, for example grouped by quartiles of person-time
  (Section 4.2).  The limits invert the calibrated test (Section 4.2.2), so a provider is flagged exactly when
  its interval excludes 1;
* ``test_method="exact"``: the exact Poisson test (Eq. 9, doubled tail capped at 0.999) with the tutorial's
  limits (Eq. 11-12: Byar's approximation when ``E_j >= 100``, the exact chi-square limits otherwise); theoretical
  null only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from ..decision import provider_test, resolve_null_model
from ..zstat import ZFrame
from .empirical_null import poisson_midp_zscore
from .inference import poisson_exact_test

_METHODS = ("midp", "exact")


def _midp_limits(obs, expected, mu, sd, alpha):
    """Limits for the Poisson mean from inverting the calibrated two-sided mid-p test.

    For observed count ``O`` the mid-p z-statistic ``z(O, e)`` (Eq. 10 and 14) falls continuously from +4.75 to
    -4.75 as the Poisson mean ``e`` grows (the p-value floor of 1e-6 caps it), so the calibrated p-value
    ``2 * (1 - Phi(|z - mu| / sd))`` rises to 1 where ``z = mu`` and falls on either side.  The limits are where
    it equals ``alpha``; a side on which it stays above ``alpha`` has limit 0 (lower) or infinity (upper).  This
    is Section 4.2.2's inversion, without its case analysis, which assumes the root lies in given ranges.
    """
    lower = np.empty(obs.size)
    upper = np.empty(obs.size)
    for i, (o, e, m, s_) in enumerate(zip(obs, expected, mu, sd)):
        z = lambda t: float(poisson_midp_zscore([o], [t])[0])                   # noqa: E731
        excess = lambda t: 2.0 * norm.sf(abs(z(t) - m) / s_) - alpha           # noqa: E731
        lo_end, hi_end = 1e-10 * max(e, 1.0), 10.0 * (o + e + 10.0)
        while z(hi_end) > max(m, -4.75) and hi_end < 1e12:
            hi_end *= 10.0
        if z(lo_end) <= m:
            mode = lo_end
        elif z(hi_end) >= m:
            mode = hi_end
        else:
            mode = brentq(lambda t: z(t) - m, lo_end, hi_end, xtol=1e-12 * max(e, 1.0))
        lower[i] = 0.0 if excess(lo_end) >= 0.0 else brentq(excess, lo_end, mode, xtol=1e-10 * max(e, 1.0))
        upper[i] = np.inf if excess(hi_end) >= 0.0 else brentq(excess, mode, hi_end, xtol=1e-10 * max(e, 1.0))
    return lower, upper


class CoxPHInferenceMixin:
    """Provider tests for ``CoxPH``."""

    def test(
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
        test_method: str = "midp",
        null_model=None,
        level: float = 0.95,
    ) -> pd.DataFrame:
        """Test each provider's indirect standardized ratio against 1 (SMR tutorial, Section 4).

        Parameters
        ----------
        X, duration, event, start, stop, offset, provider_id
            As in :meth:`calculate_standardized_measures`: the data the model was fitted to and each row's
            provider.
        providers : array-like, optional
            Providers to report (default: all).  An empirical null is always fitted on every provider.
        test_method : {"midp", "exact"}, default "midp"
            ``"midp"``: the two-sided mid-p test, calibrated by ``null_model``.  ``"exact"``: the exact Poisson
            test with Byar or chi-square limits, under the theoretical null.
        null_model : NullModel or callable, optional
            ``None`` for the theoretical null.  The tutorial's empirical null groups providers by quartiles of
            person-time: ``EmpiricalNull.fitter(size=person_time, n_groups=4, estimator=HUBER_RLM)``, with
            ``person_time`` from ``calculate_standardized_measures`` (``"midp"`` only).
        level : float, default 0.95
            Two-sided confidence level: providers are flagged when the p-value is below ``1 - level``.

        Returns
        -------
        pandas.DataFrame
            Indexed by provider, with the columns of every provider test
            (:data:`~pprof_py.inference.PROVIDER_TEST_COLUMNS`) and ``observed``, ``expected`` and
            ``person_time``.  ``estimate`` is the ratio ``O_j / E_j`` and ``null_value`` 1; ``z_raw`` is the
            mid-p z-statistic (for ``"exact"``, the normal quantile of the exact p-value, signed by the
            direction); ``flag`` is +1 for more events than expected, -1 for fewer; ``ci_lower`` and
            ``ci_upper`` are limits for the ratio.
        """
        if test_method not in _METHODS:
            raise ValueError(f"test_method must be one of {_METHODS}, got {test_method!r}")
        if not 0.0 < float(level) < 1.0:
            raise ValueError("level must be strictly between 0 and 1.")
        if test_method == "exact" and null_model is not None:
            raise ValueError("test_method='exact' uses the theoretical null; the empirical null calibrates the "
                             "mid-p z-statistics (test_method='midp').")
        m = self.calculate_standardized_measures(X, duration, event, start, stop, provider_id=provider_id,
                                                 offset=offset)["indirect"]
        obs = m["observed"].to_numpy(dtype=np.float64)
        exp_ = m["expected"].to_numpy(dtype=np.float64)
        ids = pd.Index(m["provider_id"].to_numpy(), name="provider_id")
        alpha = 1.0 - float(level)
        if test_method == "midp":
            z = poisson_midp_zscore(obs, exp_)
        else:
            p_exact, lo_exact, hi_exact = poisson_exact_test(obs, exp_, alpha=alpha)
            z = np.sign(obs - exp_) * norm.isf(p_exact / 2.0)
        zf = ZFrame.from_arrays(z, ids)
        null = resolve_null_model(null_model, zf)
        out = provider_test(zf, null, level=level)
        out["estimate"] = obs / exp_
        out["null_value"] = 1.0
        if test_method == "midp":
            loc, scl, _ = null.parameters(zf)
            lo, hi = _midp_limits(obs, exp_, np.broadcast_to(loc, obs.shape), np.broadcast_to(scl, obs.shape), alpha)
            out["ci_lower"], out["ci_upper"] = lo / exp_, hi / exp_
        else:
            out["ci_lower"], out["ci_upper"] = lo_exact, hi_exact
        out["observed"], out["expected"] = obs, exp_
        out["person_time"] = m["person_time"].to_numpy(dtype=np.float64)
        out.attrs["test_method"] = test_method
        out.attrs["measure"] = "indirect_ratio"
        out.attrs["reference"] = 1.0
        if providers is not None:
            out = out[out.index.isin(np.atleast_1d(providers))]
        return out

    def funnel_limits(self, X, duration=None, event=None, start=None, stop=None, *, provider_id, offset=None,
                      providers=None, test_method: str = "midp", null_model=None, level: float = 0.95, levels=None):
        """Funnel coordinates and control limits that agree with :meth:`test` by construction.

        For each provider, the smallest event count the test flags high and the largest it flags low under its
        Poisson null; limits sit half-way between counts on the O/E scale, so no provider lies on a line. Curves
        over the expected count E are exact.

        Parameters
        ----------
        X, duration, event, start, stop, provider_id, offset, providers, test_method, null_model, level
            As in :meth:`test`.
        levels : sequence of float, optional
            Levels of the limit curves (default: ``level`` only). Flags exist only at ``level``.

        Returns
        -------
        FunnelLimits
            See :func:`pprof_py.inference.funnel_limits`.
        """
        from ..funnel import poisson_funnel_limits
        res = self.test(X, duration, event, start, stop, provider_id=provider_id, offset=offset, providers=providers,
                        test_method=test_method, null_model=null_model, level=level)
        alpha = 1.0 - float(level)
        if test_method == "midp":
            def zfun(o, e):
                return poisson_midp_zscore(o, e)
        else:
            def zfun(o, e):
                return np.sign(o - e) * norm.isf(poisson_exact_test(o, e, alpha=alpha)[0] / 2.0)
        return poisson_funnel_limits(self, res, zfun, levels)
