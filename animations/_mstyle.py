"""
Shared palette for the qwalk-topo Manim animations.

The hex values are copied from ``examples/_style.py`` so the animations and the
static figures read as one set. Manim accepts hex strings directly as colours,
so nothing needs converting.

Kept separate from ``examples/_style.py`` because that module imports Matplotlib
and calls ``rcParams.update`` on import, which an animation has no use for.
"""
from __future__ import annotations

# Same five colours as examples/_style.py -- keep them in sync by hand if that
# file's palette ever changes.
RING = "#1f4e79"      # orientable / ring / periodic          (deep blue)
MOBIUS = "#c0392b"    # non-orientable / Mobius / anti-per.    (brick red)
BAND = "#7f8c8d"      # analytic continuum bands               (grey)
EDGE = "#e67e22"      # topological edge / boundary modes      (orange)
ACCENT = "#2c3e50"    # gap-closing lines, guides, seam        (slate)

BG = "#ffffff"        # white background, matching the figures
INK = "#2c3e50"       # default text/axis colour on that background


# ---------------------------------------------------------------- notation
# Pango renders Greek and maths glyphs directly, so the scenes use real
# notation instead of spelling symbols out -- no LaTeX involved. Verified for
# nu, theta, subscripts, pi, psi, epsilon, the true minus sign and the arrow.
#
# Note there is no subscript y or z in Unicode, so n_y and n_z stay written
# that way; theta_1 and theta_2 do have subscripts and use them.
MINUS = "−"


def signed(x: float, places: int = 4) -> str:
    """Format a number with a true minus sign (U+2212), not a hyphen.

    A hyphen is narrower than the digits and sits at the wrong height, which
    looks wrong next to a column of numbers.
    """
    return format(x, "+." + str(places) + "f").replace("-", MINUS)
