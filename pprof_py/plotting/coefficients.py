"""Provider-effect caterpillar plot.

Provides :func:`plot_caterpillar`, a sorted point-and-interval display
of provider effects (gamma or alpha) used by the linear and logistic
fixed-effect and random-effect plotting mixins.
"""
import pandas as pd

from typing import Optional

from . import style as _style


def plot_caterpillar(df: pd.DataFrame, estimate_col: str = "estimate", ci_lower_col: Optional[str] = "ci_lower",
                     ci_upper_col: Optional[str] = "ci_upper", group_col: Optional[str] = None,
                     flag_col: Optional[str] = None, *, refline_value: float = 0.0,
                     plot_title: str = "Caterpillar Plot", save_path: Optional[str] = None,
                     dpi: int = _style.SAVE_DPI):
    """Interval plot from a DataFrame, drawn through :func:`pprof_py.presentation.caterpillar` (D45).

    Intervals are drawn from ``ci_lower_col`` to ``ci_upper_col``, the reference line at ``refline_value``, and
    flags (``flag_col``; without one, providers read "not tested") with the package's status encodings. Returns a
    :class:`~pprof_py.presentation.FigureResult` instead of ``None`` and no longer calls ``plt.show()``. The
    styling keywords, ``sort_by_estimate`` and ``orientation`` were removed in 0.7.0, and so was drawing without
    intervals or a reference line.
    """
    from ._standalone import caterpillar_from_frame

    return caterpillar_from_frame(df, estimate_col, ci_lower_col, ci_upper_col, group_col, flag_col,
                                  refline_value=refline_value, plot_title=plot_title, save_path=save_path, dpi=dpi)
