"""Small classification metrics used by pprof_py (replacing scikit-learn's)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

__all__ = ["roc_auc_score", "cohen_kappa_score"]


def roc_auc_score(y_true, y_score) -> float:
    """Area under the ROC curve for 0/1 (or -1/1) labels, ties counted as 1/2 (Mann-Whitney)."""
    y = np.asarray(y_true).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    labels = np.unique(y)
    if not (np.array_equal(labels, [0, 1]) or np.array_equal(labels, [-1, 1])):
        raise ValueError(f"y_true takes values in {labels.tolist()}; expected {{0, 1}} or {{-1, 1}}.")
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    ranks = rankdata(s)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cohen_kappa_score(y1, y2, weights=None) -> float:
    """Cohen's kappa with scikit-learn's conventions: sorted union of labels, index-based weights."""
    a, b = np.asarray(y1).ravel(), np.asarray(y2).ravel()
    labels = np.unique(np.concatenate([a, b]))
    n = labels.size
    confusion = np.zeros((n, n), dtype=np.int64)
    np.add.at(confusion, (np.searchsorted(labels, a), np.searchsorted(labels, b)), 1)
    sum0, sum1 = confusion.sum(axis=0), confusion.sum(axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)
    if weights is None:
        w = np.ones((n, n), dtype=int)
        w.flat[:: n + 1] = 0
    else:
        w = np.zeros((n, n), dtype=int)
        w += np.arange(n)
        if weights == "linear":
            w = np.abs(w - w.T)
        elif weights == "quadratic":
            w = (w - w.T) ** 2
        else:
            raise ValueError("weights must be None, 'linear', or 'quadratic'.")
    return float(1 - np.sum(w * confusion) / np.sum(w * expected))
