"""
Shared publication-quality Matplotlib style for the qwalk-topo figures.

Importing this module and calling ``apply()`` gives every example the same
serif, Computer-Modern-math look, tick discipline, and colour palette, so the
figures read as one coherent set. Kept dependency-free (Matplotlib only).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Consistent, colour-blind-friendly palette used across all figures.
RING = "#1f4e79"      # orientable / ring / periodic        (deep blue)
MOBIUS = "#c0392b"    # non-orientable / Mobius / anti-per.  (brick red)
BAND = "#7f8c8d"      # analytic continuum bands             (grey)
EDGE = "#e67e22"      # topological edge / boundary modes    (orange)
ACCENT = "#2c3e50"    # gap-closing lines, guides            (slate)


def apply() -> None:
    """Apply the shared rcParams. Call once at the top of an example."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "CMU Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 1.6,
    })


def panel_label(ax, text: str, loc: str = "upper left", pad: float = 0.04) -> None:
    """Put a bold (a)/(b) style panel label inside an axes."""
    x, ha = (pad, "left") if "left" in loc else (1 - pad, "right")
    y, va = (1 - pad, "top") if "upper" in loc else (pad, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, fontsize=12, fontweight="bold",
            ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))
