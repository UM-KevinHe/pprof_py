"""Data preparation for the three-stage logistic model (R's ``glmm.data.prep``).

:func:`glmm_data_prep` screens providers, adjusts the outcome of providers with no
or all events, and numbers the provider x cluster cells that Stage 1 fits. It
reproduces R's ``glmm.data.prep`` row for row. R's names map to these:

========================  =======================
R (``glmm.data.prep``)     pprof_py
========================  =======================
``fac.size``               ``provider_size``
``Y.adj``                  ``y_adj``
``prov_ID``                ``cell_id``
``included``               ``included``
``n.fac.hosp``             ``cell_sizes``
``fac``, ``hosp``          ``n_providers``, ``n_clusters``
========================  =======================
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GLMMPreparedData:
    """Output of :func:`glmm_data_prep`.

    Attributes
    ----------
    data : pandas.DataFrame
        The screened records, sorted by cluster and then provider, in data order within
        a provider x cluster cell. Added columns: ``provider_size`` (the provider's record
        count), ``y_adj``, ``cell_id`` (the cell's number, from 1, in that order) and
        ``included`` (1 when the cell has more than ``cutoff`` records). The provider
        and cluster columns become categorical.
    cell_sizes : numpy.ndarray
        Records in every provider x cluster combination, empty ones included, cluster-major
        with the provider varying fastest (R's ``as.integer(ftable(table(fac, hosp)))``).
    n_providers, n_clusters : int
        Providers and clusters after screening.
    """

    data: pd.DataFrame
    cell_sizes: np.ndarray
    n_providers: int
    n_clusters: int


def glmm_data_prep(data: pd.DataFrame, y_var: str, provider_var: str, cluster_var: str,
                   cutoff: int = 10) -> GLMMPreparedData:
    """Prepare data for the three-stage model, as R's ``glmm.data.prep``.

    1. Keep providers with more than ``cutoff`` records.
    2. ``y_adj``: the binary outcome, raised by ``0.01 / provider_size`` for providers with
       no events and lowered by that amount for providers with all events, so that every
       provider's effect is finite.
    3. Sort by cluster, then provider, keeping the data order within each cell.
    4. Number the provider x cluster cells in that order (``cell_id``) and mark the cells
       with more than ``cutoff`` records (``included``); Stage 1 fits the included cells.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per record.
    y_var : str
        Binary (0/1) outcome.
    provider_var, cluster_var : str
        Provider (facility) and cluster (hospital) IDs. IDs are ordered as pandas sorts
        them; R's ``factor()`` can order text IDs differently under some locales, which
        renumbers the cells but does not change the fits.
    cutoff : int, default 10
        Screening threshold for providers and for included cells.

    Returns
    -------
    GLMMPreparedData
    """
    missing = [c for c in (y_var, provider_var, cluster_var) if c not in data.columns]
    if missing:
        raise KeyError(f"Column(s) {missing} not found.")
    if data[[y_var, provider_var, cluster_var]].isna().any().any():
        raise ValueError("The outcome, provider and cluster columns must not have missing values.")
    if not set(pd.unique(data[y_var])).issubset({0, 1}):
        raise ValueError(f"'{y_var}' must be binary (0 or 1).")
    d = data.sort_values(provider_var, kind="stable").reset_index(drop=True)
    size = d.groupby(provider_var, sort=False)[provider_var].transform("size").to_numpy()
    d["provider_size"] = size
    keep = size > cutoff
    d, size = d[keep].reset_index(drop=True), size[keep]
    events = d.groupby(provider_var, sort=False)[y_var].transform("sum").to_numpy(np.float64)
    d["y_adj"] = d[y_var].to_numpy(np.float64) + (events == 0) * 0.01 / size - (events == size) * 0.01 / size
    d[cluster_var] = pd.Categorical(d[cluster_var])
    d[provider_var] = pd.Categorical(d[provider_var])
    d = d.sort_values([cluster_var, provider_var], kind="stable").reset_index(drop=True)
    # Number the cells from the sorted rows themselves: a new cell starts wherever the
    # (cluster, provider) pair changes, so cell_id and included cannot be attached to the
    # wrong rows. (Repeating groupby counts onto the rows assumed that the groupby returns
    # the cells in the rows' order; on production data it did not, and cell_id and
    # included landed on the wrong rows with every total unchanged.)
    c_codes = d[cluster_var].cat.codes.to_numpy(np.int64)
    p_codes = d[provider_var].cat.codes.to_numpy(np.int64)
    new_cell = np.r_[True, (c_codes[1:] != c_codes[:-1]) | (p_codes[1:] != p_codes[:-1])]
    cell_id = np.cumsum(new_cell)
    n_cell = np.bincount(cell_id)[1:]
    d["cell_id"] = cell_id
    d["included"] = (n_cell[cell_id - 1] > cutoff).astype(int)
    # R's n.fac.hosp: every provider x cluster combination, cluster-major, provider-minor
    # (as.integer(ftable(table(fac, hosp)))), counted from the category codes.
    n_prov_levels = len(d[provider_var].cat.categories)
    cell_sizes = np.bincount(c_codes * n_prov_levels + p_codes,
                             minlength=len(d[cluster_var].cat.categories) * n_prov_levels)
    return GLMMPreparedData(data=d, cell_sizes=cell_sizes, n_providers=int(d[provider_var].nunique()),
                            n_clusters=int(d[cluster_var].nunique()))
