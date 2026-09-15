"""Centralized, publication-quality plotting for pprof_py.

All plots share one visual identity defined in `plotting.style`. Plotting
functions accept prepared results/DataFrames and return `(fig, ax)` so
callers can display, modify, or save the figure themselves rather than
the function forcing `plt.show()`.
"""
from .coefficients import plot_caterpillar
from .funnel import plot_funnel

__all__ = ["plot_caterpillar", "plot_funnel"]
