"""Provider measures and their z-statistics.

A :class:`MeasureFrame` holds one estimate and standard error per provider.
:func:`z_statistic` turns it into a :class:`ZFrame`: the estimate is moved to a
working scale by a :class:`Transform` (with the delta-method standard error),
compared with a null value, and standardised.

The null value is always given on the measure's own scale (a rate, a ratio,
...) and transformed internally, so ``null_value=1.0`` for a ratio means a
ratio of 1 whatever the working scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import expit

__all__ = [
    "Transform",
    "IDENTITY",
    "LOGIT",
    "LOG",
    "MeasureFrame",
    "ZFrame",
    "z_statistic",
]


@dataclass(frozen=True)
class Transform:
    """A monotone increasing working-scale transform.

    Attributes
    ----------
    name : str
    forward : callable
        ``f(x)``.
    se_forward : callable
        ``(x, se) -> se of f(x)`` by the delta method, ``se * |f'(x)|``.
    inverse : callable
        ``f^{-1}(t)``, used to report intervals on the measure's own scale.
    """

    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    se_forward: Callable[[np.ndarray, np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]


IDENTITY = Transform("identity", lambda x: x, lambda x, se: se, lambda t: t)
LOGIT = Transform("logit", lambda x: np.log(x / (1.0 - x)), lambda x, se: se / (x * (1.0 - x)), expit)
LOG = Transform("log", np.log, lambda x, se: se / x, np.exp)

_BY_NAME = {"identity": IDENTITY, "logit": LOGIT, "log": LOG}
_AUTO = {
    "direct_rate": LOGIT, "direct_ratio": LOG,
    # Indirect measures: on the identity scale, with the variance of O_j at gamma_0,
    # the test is the score test of gamma_j = gamma_0 and zero-event providers stay testable.
    "indirect_rate": IDENTITY, "indirect_ratio": IDENTITY,
    "gamma": IDENTITY,
}


def _provider_index(provider, n: int) -> pd.Index:
    if provider is None:
        index = pd.RangeIndex(n, name="provider")
    else:
        index = pd.Index(np.asarray(provider).ravel(), name="provider")
        if len(index) != n:
            raise ValueError("provider must have one entry per estimate.")
    if not index.is_unique:
        raise ValueError("provider identifiers must be unique.")
    return index


@dataclass(frozen=True)
class MeasureFrame:
    """One estimate and standard error per provider.

    Attributes
    ----------
    estimate, se : numpy.ndarray
    index : pandas.Index
        Provider identifiers (named ``"provider"``).
    measure : str, optional
        Measure name (for example ``"direct_rate"``); used by ``transform="auto"``.
    reference_value : float, optional
        The measure's value when the provider effect equals the reference
        (for example 1.0 for an indirect ratio); used by ``null_value="reference"``.
        Set by the model-backed measure layer.
    """

    estimate: np.ndarray
    se: np.ndarray
    index: pd.Index
    measure: Optional[str] = None
    reference_value: Optional[float] = None

    @classmethod
    def from_arrays(cls, estimate, se, provider=None, *, measure=None, reference_value=None) -> "MeasureFrame":
        """Build from estimates and standard errors computed anywhere."""
        est = np.asarray(estimate, dtype=np.float64).ravel()
        s = np.asarray(se, dtype=np.float64).ravel()
        if est.shape != s.shape:
            raise ValueError("estimate and se must have the same length.")
        return cls(est, s, _provider_index(provider, est.size), measure, reference_value)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, estimate: str = "estimate", se: str = "se",
                   measure=None, reference_value=None) -> "MeasureFrame":
        """Build from a DataFrame indexed by provider."""
        return cls.from_arrays(frame[estimate].to_numpy(), frame[se].to_numpy(), frame.index,
                               measure=measure, reference_value=reference_value)

    def __len__(self) -> int:
        return self.estimate.size

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"estimate": self.estimate, "se": self.se}, index=self.index)


@dataclass(frozen=True)
class ZFrame:
    """Per-provider z-statistics, with the quantities needed for intervals.

    Built by :func:`z_statistic`, or by :meth:`from_arrays` for z-statistics
    computed elsewhere (in which case intervals are not available).
    ``z`` is NaN for providers that cannot be tested.
    """

    z: np.ndarray
    index: pd.Index
    estimate: Optional[np.ndarray] = None
    se: Optional[np.ndarray] = None
    transformed: Optional[np.ndarray] = None
    se_transformed: Optional[np.ndarray] = None
    null_value: Optional[float] = None
    null_transformed: Optional[float] = None
    transform: Transform = IDENTITY
    measure: Optional[str] = None

    @classmethod
    def from_arrays(cls, z, provider=None) -> "ZFrame":
        """Wrap z-statistics computed elsewhere (for example Poisson mid-p z-scores)."""
        zz = np.asarray(z, dtype=np.float64).ravel().copy()
        zz[~np.isfinite(zz)] = np.nan
        return cls(zz, _provider_index(provider, zz.size))

    @property
    def has_intervals(self) -> bool:
        return self.transformed is not None and self.se_transformed is not None

    def __len__(self) -> int:
        return self.z.size

    def to_frame(self) -> pd.DataFrame:
        n = self.z.size
        nan = np.full(n, np.nan)
        return pd.DataFrame({
            "estimate": nan if self.estimate is None else self.estimate,
            "se": nan if self.se is None else self.se,
            "null_value": np.full(n, np.nan if self.null_value is None else self.null_value),
            "transformed": nan if self.transformed is None else self.transformed,
            "se_transformed": nan if self.se_transformed is None else self.se_transformed,
            "null_transformed": np.full(n, np.nan if self.null_transformed is None else self.null_transformed),
            "z_raw": self.z,
        }, index=self.index)


def _resolve_transform(transform, measure: Optional[str]) -> Transform:
    if isinstance(transform, Transform):
        return transform
    if transform == "auto":
        if measure not in _AUTO:
            raise ValueError("transform='auto' needs a known measure "
                             f"({sorted(_AUTO)}); pass the transform explicitly.")
        return _AUTO[measure]
    if transform in _BY_NAME:
        return _BY_NAME[transform]
    raise ValueError("transform must be 'auto', 'identity', 'logit', 'log', or a Transform.")


def z_statistic(
    measure: Union[MeasureFrame, pd.DataFrame],
    *,
    null_value: Union[str, float, Callable[[np.ndarray], float]] = "reference",
    transform: Union[str, Transform] = "auto",
) -> ZFrame:
    """Standardise provider measures against a null value.

    ``z = (f(estimate) - f(null_value)) / se_f`` with ``se_f`` the delta-method
    standard error of ``f(estimate)``.

    Parameters
    ----------
    measure : MeasureFrame or DataFrame
        A DataFrame needs ``estimate`` and ``se`` columns and a provider index.
    null_value : "reference", "mean", "median", float, or callable
        The value under the null, on the measure's own scale.
        ``"reference"`` uses the measure's ``reference_value`` (the measure at
        the reference provider effect), so the test agrees with a test of the
        provider effect itself. ``"mean"`` and ``"median"`` summarise the
        finite estimates (unweighted); a callable receives the finite
        estimates and returns the null value.
    transform : "auto", "identity", "logit", "log", or Transform
        Working scale. ``"auto"`` uses logit for direct rates, log for direct
        ratios, and identity for indirect measures and provider effects. On the
        logit and log scales the delta method is evaluated at the estimate.

    Returns
    -------
    ZFrame
        Providers whose working-scale estimate or standard error is not finite,
        or whose standard error is not positive, get ``z = NaN`` (not tested).
    """
    m = measure if isinstance(measure, MeasureFrame) else MeasureFrame.from_frame(measure)
    tr = _resolve_transform(transform, m.measure)
    est, se = m.estimate, m.se
    finite = est[np.isfinite(est)]

    if isinstance(null_value, str):
        if null_value == "reference":
            if m.reference_value is None:
                raise ValueError("null_value='reference' needs a measure with a reference_value "
                                 "(as built by the model-backed measure layer); pass a number, "
                                 "'mean', or 'median' instead.")
            nv = float(m.reference_value)
        elif null_value in ("mean", "median"):
            if finite.size == 0:
                raise ValueError(f"null_value={null_value!r} needs at least one finite estimate.")
            nv = float(np.mean(finite) if null_value == "mean" else np.median(finite))
        else:
            raise ValueError("null_value must be 'reference', 'mean', 'median', a number, or a callable.")
    elif callable(null_value):
        nv = float(null_value(finite))
    else:
        nv = float(null_value)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.asarray(tr.forward(est), dtype=np.float64)
        se_t = np.asarray(tr.se_forward(est, se), dtype=np.float64)
        nt = float(tr.forward(np.float64(nv)))
        z = (t - nt) / se_t
    if not np.isfinite(nt):
        raise ValueError(f"null_value {nv!r} is outside the domain of the {tr.name} transform.")
    untestable = ~np.isfinite(t) | ~np.isfinite(se_t) | ~(se_t > 0) | ~np.isfinite(z)
    z = np.where(untestable, np.nan, z)
    return ZFrame(z=z, index=m.index, estimate=est, se=se, transformed=t, se_transformed=se_t,
                  null_value=nv, null_transformed=nt, transform=tr, measure=m.measure)
