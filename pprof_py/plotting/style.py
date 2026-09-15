"""Package-wide visual style specification for pprof_py plots.

All plotting functions in :mod:`pprof_py.plotting` (and, over time, the
plotting methods still embedded in model classes) should source their
colors, sizes, and default figure options from this module rather than
choosing them independently, so that every figure produced by the package
shares one consistent, publication-ready, colorblind-friendly visual
identity.

The categorical palette below follows the Okabe-Ito colorblind-safe
palette (Okabe & Ito, 2008), assigned to semantic roles rather than to
arbitrary per-function positions.
"""
from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette, assigned to semantic roles.
# ---------------------------------------------------------------------------
COLOR_PRIMARY = "#56B4E9"      # sky blue -- primary estimate / "expected" group
COLOR_SECONDARY = "#009E73"    # bluish green -- secondary estimate / "higher" group
COLOR_WARNING = "#E69F00"      # orange -- warning / diagnostic / "lower" group
COLOR_REFERENCE = "black"      # reference / null line
COLOR_HIGHLIGHT = "#CC79A7"    # reddish purple -- highlighted point(s)
COLOR_NEUTRAL_GREY = "grey"    # error bars, secondary annotations

# Flag-coded categorical roles (e.g. -1 = lower, 0 = expected, 1 = higher).
FLAG_COLORS: Dict[int, str] = {-1: COLOR_WARNING, 0: COLOR_PRIMARY, 1: COLOR_SECONDARY}
FLAG_LABELS = ["Lower", "Expected", "Higher"]

# ---------------------------------------------------------------------------
# Figure/typography defaults.
# ---------------------------------------------------------------------------
FIGURE_SIZE = (8.0, 6.0)
FONT_SIZE = 12
TICK_LABEL_SIZE = 10

# ---------------------------------------------------------------------------
# Line/marker defaults.
# ---------------------------------------------------------------------------
POINT_SIZE = 2.0
POINT_ALPHA = 0.8
LINE_WIDTH = 0.8
ERRORBAR_SIZE = 0.5
ERRORBAR_ALPHA = 0.5

# ---------------------------------------------------------------------------
# Reference-line and grid conventions.
# ---------------------------------------------------------------------------
REFLINE_LINESTYLE = "--"
GRID_STYLE = ":"
GRID_ALPHA = 0.6
GRID_COLOR = "lightgrey"

# ---------------------------------------------------------------------------
# Confidence-interval fill defaults (funnel plots, etc.).
# ---------------------------------------------------------------------------
CI_FILL_COLOR = "#A6CEE3"    # light blue -- confidence band fill
CI_FILL_ALPHA = 0.25

# ---------------------------------------------------------------------------
# Coefficient / forest-plot defaults.
# ---------------------------------------------------------------------------
COLOR_ESTIMATE = "#34495E"   # dark slate -- unflagged point estimate
CAPSIZE = 5

# ---------------------------------------------------------------------------
# Spine and title defaults.
# ---------------------------------------------------------------------------
SPINE_WIDTH = 0.8
TITLE_PAD = 15

# ---------------------------------------------------------------------------
# Output defaults.
# ---------------------------------------------------------------------------
SAVE_DPI = 300


def remove_top_right_spines(ax) -> None:
    """Remove the top and right spines from `ax`, the package's default
    for a clean, publication-ready axes frame."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
