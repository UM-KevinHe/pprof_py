"""Null distributions for provider z-statistics.

A null model gives, for each provider, the mean and standard deviation of the
normal distribution its z-statistic is referred to:
``z_adjusted = (z_raw - null_mean) / null_sd``.

* :class:`TheoreticalNull`: N(0, 1).
* :class:`FixedNull`: N(mean, sd^2) with values supplied by the caller.
* :class:`EmpiricalNull`: mean and SD estimated robustly from the z-statistics,
  overall or within groups of providers.

Argument names and defaults follow EmpiNull's robust calibration
(``fit_mask``, ``common_mean``, ``min_group_size``, ``small_group``, quantile
grouping, bisquare estimation), so its workflows translate directly.
"""
from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..zstat import ZFrame
from .estimators import DEFAULT_ESTIMATOR, LocationScale
from .grouping import assign_groups

__all__ = ["NullModel", "TheoreticalNull", "FixedNull", "EmpiricalNull", "EmpiricalNullWarning"]


class EmpiricalNullWarning(UserWarning):
    """A group fell back to the theoretical null, a fit did not converge, or providers lack a group."""


def _as_z(z) -> Tuple[np.ndarray, Optional[pd.Index]]:
    if isinstance(z, ZFrame):
        return z.z, z.index
    if isinstance(z, pd.Series):
        return z.to_numpy(dtype=np.float64), z.index
    return np.asarray(z, dtype=np.float64).ravel(), None


class NullModel:
    """Base class. Subclasses provide per-provider null mean and SD."""

    def parameters(self, z) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return ``(null_mean, null_sd, group)`` arrays aligned with ``z``."""
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class TheoreticalNull(NullModel):
    """The standard normal null, N(0, 1)."""

    def parameters(self, z):
        n = len(_as_z(z)[0])
        return np.zeros(n), np.ones(n), None

    def describe(self) -> dict:
        return {"kind": "theoretical", "null_mean": 0.0, "null_sd": 1.0}


@dataclass(frozen=True)
class FixedNull(NullModel):
    """A normal null with caller-supplied mean and SD (for example ``sd=1.81``)."""

    mean: float = 0.0
    sd: float = 1.0

    def __post_init__(self):
        if not (np.isfinite(self.mean) and np.isfinite(self.sd) and self.sd > 0):
            raise ValueError("FixedNull needs a finite mean and a positive finite sd.")

    def parameters(self, z):
        n = len(_as_z(z)[0])
        return np.full(n, float(self.mean)), np.full(n, float(self.sd)), None

    def describe(self) -> dict:
        return {"kind": "fixed", "null_mean": float(self.mean), "null_sd": float(self.sd)}


@dataclass(frozen=True)
class EmpiricalNull(NullModel):
    """A null estimated from the z-statistics, overall or per group.

    Build with :meth:`fit`, or with :meth:`from_parameters` when the mean and SD
    were estimated elsewhere.

    Attributes
    ----------
    mean, sd : numpy.ndarray
        Per-provider null parameters (each provider carries its group's values).
    group : numpy.ndarray
        Group label per provider (all 1 for an overall null).
    diagnostics : pandas.DataFrame
        One row per group: ``group, n_providers, n_fitted, fitted_mean,
        null_mean, null_sd, converged, n_iter, fallback, note``.
    index : pandas.Index or None
        Providers the null was fitted on, when known; used to check alignment.
    """

    mean: np.ndarray
    sd: np.ndarray
    group: np.ndarray
    diagnostics: pd.DataFrame
    estimator: Any = None
    index: Optional[pd.Index] = field(default=None)

    @classmethod
    def fit(
        cls,
        z,
        *,
        groups=None,
        size=None,
        n_groups: int = 4,
        grouping: str = "quantile",
        order=None,
        fit_mask=None,
        common_mean: Union[None, bool, float] = None,
        estimator: Callable = DEFAULT_ESTIMATOR,
        min_group_size: int = 3,
        small_group: str = "error",
    ) -> "EmpiricalNull":
        """Estimate the null's mean and SD from z-statistics.

        Parameters
        ----------
        z : ZFrame, pandas.Series, or array-like
            Per-provider z-statistics. Missing values are not used in the fit.
        groups : array-like, optional
            Explicit group label per provider. Every provider receives its
            group's parameters, including providers with missing z or left out
            of the fit. Providers with a missing label get no parameters.
        size, n_groups, grouping, order : optional
            Build groups from a size variable instead (see
            :func:`assign_groups`); ``order`` breaks ties in size for
            ``grouping="rank"`` (for example provider IDs). With neither
            ``groups`` nor ``size`` (or ``n_groups=1``) the null is overall.
        fit_mask : array-like of bool, optional
            Providers eligible for the fit (``True``); the others are still
            calibrated with their group's parameters.
        common_mean : None, bool, or float
            ``None``/``False``: each group's own fitted mean. ``True``: the mean
            of all eligible finite z-values. A number: that value. Group SDs
            are kept as fitted either way.
        estimator : callable
            Called on each group's eligible z-values; returns a
            :class:`LocationScale` or a ``(mean, sd)`` tuple. The default is
            bisquare M-estimation to tight convergence (EmpiNull's default);
            :data:`HUBER_RLM`, :data:`BISQUARE_RLM`, and :data:`MM_RLM`
            reproduce ``MASS::rlm`` with MASS defaults.
        min_group_size : int
            Minimum number of eligible values needed to fit a group (at least 2).
        small_group : {"error", "theoretical"}
            Raise for a group below ``min_group_size``, or use N(0, 1) for it
            (with an :class:`EmpiricalNullWarning`).
        """
        if small_group not in ("error", "theoretical"):
            raise ValueError("small_group must be 'error' or 'theoretical'.")
        if int(min_group_size) != min_group_size or min_group_size < 2:
            raise ValueError("min_group_size must be an integer >= 2.")
        zz, index = _as_z(z)
        n = zz.size
        if groups is not None:
            labels = pd.Series(np.asarray(groups).ravel())
        elif size is not None and n_groups > 1:
            labels = pd.Series(np.asarray(assign_groups(size, n_groups, rule=grouping, order=order)))
        else:
            labels = pd.Series(np.ones(n, dtype=int))
        if len(labels) != n:
            raise ValueError("groups (or size) must have one entry per provider.")
        mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool).ravel()
        if mask.size != n:
            raise ValueError("fit_mask must have one entry per provider.")

        eligible = mask & np.isfinite(zz)
        missing_label = labels.isna().to_numpy()
        uniq = list(pd.unique(labels[~missing_label]))
        try:
            uniq = sorted(uniq)
        except TypeError:
            pass

        null_mean = np.full(n, np.nan)
        null_sd = np.full(n, np.nan)
        rows, problems = [], []
        for g in uniq:
            members = (labels == g).to_numpy()
            values = zz[members & eligible]
            row = {"group": g, "n_providers": int(members.sum()), "n_fitted": int(values.size),
                   "fitted_mean": np.nan, "null_mean": np.nan, "null_sd": np.nan,
                   "converged": None, "n_iter": None, "fallback": False, "note": ""}
            if values.size < min_group_size:
                if small_group == "error":
                    raise ValueError(f"Group {g} has {values.size} eligible providers; "
                                     f"at least {min_group_size} are required.")
                row.update(fitted_mean=0.0, null_mean=0.0, null_sd=1.0, fallback=True,
                           note=f"fewer than {min_group_size} eligible values; theoretical null used")
                problems.append(f"group {g}: {row['note']}")
            else:
                res = estimator(values)
                if isinstance(res, LocationScale):
                    loc, scl = res.location, res.scale
                    row.update(converged=res.converged, n_iter=res.n_iter)
                else:
                    loc, scl = float(res[0]), float(res[1])
                if not (np.isfinite(loc) and np.isfinite(scl) and scl > 0):
                    raise ValueError(f"Group {g} has a nonpositive or nonfinite fitted SD or mean.")
                row.update(fitted_mean=loc, null_mean=loc, null_sd=scl)
                if row["converged"] is False:
                    row["note"] = "did not converge"
                    problems.append(f"group {g}: fit did not converge in {row['n_iter']} iterations")
            rows.append(row)

        diagnostics = pd.DataFrame(rows)
        if common_mean is not None and common_mean is not False:
            if common_mean is True:
                if not eligible.any():
                    raise ValueError("No eligible finite z-values for a pooled mean.")
                pooled = float(np.mean(zz[eligible]))
            else:
                pooled = float(common_mean)
            diagnostics["null_mean"] = pooled
        for row in diagnostics.itertuples():
            members = (labels == row.group).to_numpy()
            null_mean[members] = row.null_mean
            null_sd[members] = row.null_sd

        if missing_label.any():
            problems.append(f"{int(missing_label.sum())} provider(s) have no group label and are not calibrated")
        if problems:
            warnings.warn("Empirical null: " + "; ".join(problems) + ".", EmpiricalNullWarning, stacklevel=2)
        return cls(mean=null_mean, sd=null_sd, group=labels.to_numpy(), diagnostics=diagnostics,
                   estimator=estimator, index=index)

    @classmethod
    def fitter(cls, **options) -> Callable:
        """A callable that fits this null to z-statistics with the given :meth:`fit` options.

        For methods that build the z-statistics themselves, for example
        ``model.test_standardized(..., null_model=EmpiricalNull.fitter(size=sizes, n_groups=4))``.
        """
        return functools.partial(cls.fit, **options)

    @classmethod
    def from_parameters(cls, mean, sd, *, group=None, index=None) -> "EmpiricalNull":
        """Wrap per-provider null means and SDs estimated elsewhere."""
        m = np.asarray(mean, dtype=np.float64).ravel()
        s = np.asarray(sd, dtype=np.float64).ravel()
        if m.shape != s.shape:
            raise ValueError("mean and sd must have the same length.")
        if np.any(np.isfinite(s) & ~(s > 0)):
            raise ValueError("sd values must be positive.")
        grp = np.ones(m.size, dtype=int) if group is None else np.asarray(group).ravel()
        return cls(mean=m, sd=s, group=grp, diagnostics=pd.DataFrame(), estimator="supplied",
                   index=None if index is None else pd.Index(index, name="provider_id"))

    def parameters(self, z):
        zz, index = _as_z(z)
        if zz.size != self.mean.size:
            raise ValueError(f"This empirical null was fitted on {self.mean.size} providers, "
                             f"but {zz.size} z-statistics were supplied.")
        if index is not None and self.index is not None and not index.equals(self.index):
            raise ValueError("The providers do not match those the empirical null was fitted on.")
        return self.mean, self.sd, self.group

    def describe(self) -> dict:
        return {"kind": "empirical", "estimator": repr(self.estimator),
                "groups": self.diagnostics.to_dict(orient="records")}
