"""Colour and accessibility metrics for themes, in pure NumPy.

* Contrast: WCAG 2 relative luminance and contrast ratio.
* Colour difference: CIE76 Delta E*ab in CIELAB (D65 white).
* Colour-vision deficiency: the severity-1.0 matrices of Machado, Oliveira and Fernandes (2009), "A
  physiologically-based model for simulation of color vision deficiency", IEEE Transactions on Visualization and
  Computer Graphics 15(6), applied to linear sRGB.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Sequence, Union

import numpy as np

CONDITIONS = ("normal", "protan", "deutan", "tritan")
_MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}
_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])
_D65 = np.array([0.95047, 1.0, 1.08883])
Color = Union[str, Sequence[float], np.ndarray]


def hex_to_rgb(color: Color) -> np.ndarray:
    """``#RRGGBB`` (or an RGB triple in [0, 1]) as a float array in [0, 1]."""
    if not isinstance(color, str):
        return np.asarray(color, dtype=float)
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"expected a colour as #RRGGBB, got {color!r}")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(lin, dtype=float), 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1.0 / 2.4) - 0.055)


def relative_luminance(color: Color) -> float:
    """WCAG 2 relative luminance."""
    return float(srgb_to_linear(hex_to_rgb(color)) @ np.array([0.2126, 0.7152, 0.0722]))


def contrast_ratio(a: Color, b: Color = "#FFFFFF") -> float:
    """WCAG 2 contrast ratio between two colours (1 to 21)."""
    la, lb = relative_luminance(a), relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def simulate(color: Color, condition: str) -> np.ndarray:
    """The sRGB colour as seen with ``condition`` in :data:`CONDITIONS` (full severity)."""
    rgb = hex_to_rgb(color)
    if condition == "normal":
        return rgb
    if condition not in _MACHADO:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    return linear_to_srgb(_MACHADO[condition] @ srgb_to_linear(rgb))


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (D65) coordinates of an sRGB colour in [0, 1]."""
    xyz = (_SRGB_TO_XYZ @ srgb_to_linear(rgb)) / _D65
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def lightness(color: Color) -> float:
    """CIELAB L* (0 black to 100 white): what is left in grayscale."""
    return float(srgb_to_lab(hex_to_rgb(color))[0])


def delta_e(a: Color, b: Color, condition: str = "normal") -> float:
    """CIE76 colour difference between two colours as seen with ``condition``."""
    return float(np.linalg.norm(srgb_to_lab(simulate(a, condition)) - srgb_to_lab(simulate(b, condition))))


def accessibility_report(theme: Any) -> Dict[str, Any]:
    """Metrics the accessibility tests check for a theme."""
    st, bg = theme.status, theme.background
    pairs = {f"{a}/{b}": {c: delta_e(st[a].color, st[b].color, c) for c in CONDITIONS}
             for a, b in combinations(("above", "below", "not_different"), 2)}
    encodings = [(s.marker, s.filled) for s in st.values()]
    return {
        "status_contrast": {k: contrast_ratio(s.color, bg) for k, s in st.items()},
        "line_contrast": {k: contrast_ratio(getattr(theme, k), bg) for k in ("reference", "limit", "volume", "muted")},
        "text_contrast": contrast_ratio(theme.ink, bg),
        "delta_e": pairs,
        "lightness_gap_above_below": abs(lightness(st["above"].color) - lightness(st["below"].color)),
        "unique_encodings": len(set(encodings)) == len(encodings),
        "min_font_pt": theme.typography.minimum,
    }
