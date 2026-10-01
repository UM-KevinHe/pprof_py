"""Deterministic synthetic provider data for documentation, the gallery and tests (never real data).

The data have planted structure (brief §10): log-normal provider volumes, overdispersion (between-provider SD 0.3 on
the log-odds scale), a few true outliers in each direction, and zero-event providers with 11 to 40 records (kept by
the default screening, without a finite fixed effect). A
second, correlated binary measure and a continuous outcome support the multi-measure and linear displays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["provider_data"]


def provider_data(n_providers: int = 200, *, seed: int = 20261001) -> pd.DataFrame:
    """Patient-level records for ``n_providers`` synthetic providers.

    Returns
    -------
    pandas.DataFrame
        ``provider_id`` (``"F001"``, ...), covariates ``x1`` (continuous) and ``x2`` (binary), binary outcomes ``y``
        and ``y2`` (a second, correlated measure) and the continuous outcome ``y_cont``. ``attrs["planted"]`` lists
        the providers with planted high and low effects and those given no events; ``attrs["seed"]`` the seed.
    """
    if n_providers < 12:
        raise ValueError("n_providers must be at least 12")
    rng = np.random.default_rng(seed)
    n = int(n_providers)
    size = np.clip(np.round(rng.lognormal(np.log(60.0), 0.8, n)), 8, 2000).astype(int)
    effect = rng.normal(0.0, 0.30, n)
    k = max(2, n // 40)
    order = rng.permutation(n)
    high, low = order[:k], order[k:2 * k]
    effect[high] += 0.9
    effect[low] -= 0.9
    # zero-event providers are small but above the default screening cutoff (10 records), so they reach the displays
    smallest = np.argsort(size, kind="stable")
    zero = [i for i in smallest if 11 <= size[i] <= 40 and i not in set(high) | set(low)][:max(1, n // 60)]
    width = len(str(n))
    ids = np.array([f"F{i + 1:0{width}d}" for i in range(n)])
    pid = np.repeat(np.arange(n), size)
    x1 = rng.normal(size=pid.size)
    x2 = rng.binomial(1, 0.4, pid.size)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-1.4 + effect[pid] + 0.5 * x1 + 0.3 * x2))))
    effect2 = 0.6 * effect + rng.normal(0.0, 0.25, n)
    y2 = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-2.0 + effect2[pid] + 0.3 * x1))))
    y_cont = 2.0 + 0.8 * effect[pid] + 0.4 * x1 + rng.normal(size=pid.size)
    y[np.isin(pid, zero)] = 0
    out = pd.DataFrame({"provider_id": ids[pid], "x1": x1, "x2": x2, "y": y, "y2": y2, "y_cont": y_cont})
    out.attrs["seed"] = seed
    out.attrs["planted"] = {"high": sorted(ids[high]), "low": sorted(ids[low]), "zero_events": sorted(ids[zero])}
    return out
