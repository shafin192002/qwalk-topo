"""
Example: topological phase diagram of the split-step walk.

Computes the chiral winding number over the (theta1, theta2) plane and shows
the three regions (nu = -1, 0, +1) separated by the gap-closing lines
theta1 = +/- theta2. Reproduces the structure of Kitagawa et al.,
PRA 82, 033429 (2010).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Allow running as a plain script (python examples/...) without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matplotlib.colors import ListedColormap, BoundaryNorm

from qwalktopo import winding_number
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")


def main(n_grid=61):
    st.apply()
    ts = np.linspace(-np.pi / 2, np.pi / 2, n_grid)
    # sentinel 2 = "undefined" (on a gap-closing line), kept distinct from the
    # genuine trivial phase nu = 0 so the boundary lines cannot be misread as
    # thin trivial regions.
    UNDEF = 2
    Z = np.zeros((n_grid, n_grid), dtype=int)
    for i, t2 in enumerate(ts):
        for j, t1 in enumerate(ts):
            # winding_number now raises on the gap-closing lines rather than
            # returning a deceptively clean integer -- mark those points.
            try:
                Z[i, j] = winding_number(t1, t2, n_k=800)
            except ValueError:
                Z[i, j] = UNDEF

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.grid(False)
    # -1, 0, +1, undefined
    cmap = ListedColormap([st.RING, "#ecf0f1", st.MOBIUS, st.ACCENT])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    im = ax.imshow(Z, origin="lower", norm=norm, cmap=cmap,
                   extent=[-np.pi/2, np.pi/2, -np.pi/2, np.pi/2], aspect="equal")
    ax.set_xlabel(r"coin angle $\theta_1$")
    ax.set_ylabel(r"coin angle $\theta_2$")
    ax.set_title(r"Split-step winding number $\nu(\theta_1,\theta_2)$")

    # gap-closing lines theta1 = +- theta2
    line = np.array([-np.pi/2, np.pi/2])
    ax.plot(line, line, color=st.ACCENT, ls="--", lw=1.1, alpha=0.8)
    ax.plot(line, -line, color=st.ACCENT, ls="--", lw=1.1, alpha=0.8)

    # region labels
    lab = dict(fontsize=13, fontweight="bold", ha="center", va="center")
    ax.text(-1.05, 0.0, r"$\nu=+1$", color="white", **lab)
    ax.text(1.05, 0.0, r"$\nu=-1$", color="white", **lab)
    ax.text(0.0, 1.05, r"$\nu=0$", color="#2c3e50", **lab)
    ax.text(0.0, -1.05, r"$\nu=0$", color="#2c3e50", **lab)
    ax.text(0.62, 1.16, r"$\theta_1=\theta_2$", color=st.ACCENT, fontsize=8.5,
            rotation=45, ha="center")
    ax.text(-0.62, 1.16, r"$\theta_1=-\theta_2$", color=st.ACCENT, fontsize=8.5,
            rotation=-45, ha="center")

    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.82, pad=0.03)
    cbar.set_label(r"winding number $\nu$")
    ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
    ax.set_xticklabels([r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", "0",
                        r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"])
    ax.set_yticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
    ax.set_yticklabels([r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", "0",
                        r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"])
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "phase_diagram.png")
    fig.savefig(out)
    print(f"saved {out}")
    n_undef = int(np.count_nonzero(Z == UNDEF))
    print(f"gap-closing (undefined) grid points flagged: {n_undef}")
    print("distinct winding values found:", sorted(set(Z.flatten().tolist())))


if __name__ == "__main__":
    main()
