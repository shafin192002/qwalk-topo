"""
The topological invariant, visualised: the winding of n_hat(k).

For the 1D split-step walk in the symmetric time frame, chiral symmetry pins the
effective-Hamiltonian Bloch vector n_hat(k) to the (n_y, n_z) plane. The winding
number nu counts how many times n_hat(k) wraps the origin as k crosses the
Brillouin zone. This example shows that winding two ways for the SAME 1D walk:

  (a) the honest, quantitative view -- the closed curve (n_y(k), n_z(k)) in the
      plane. A non-trivial point (nu = -1) encircles the origin once; a trivial
      point (nu = 0) does not. This is the figure that actually proves the point.
  (b) the same n_hat(k) drawn on the Bloch sphere -- a great circle. Prettier,
      but strictly weaker than (a) (the winding is a planar fact), so it is
      included only as an aid to intuition.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo.invariants.winding import _bloch_vector
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")


def bloch_curve(theta1, theta2, n_k=400):
    ks = np.linspace(-np.pi, np.pi, n_k, endpoint=True)
    n = np.array([_bloch_vector(theta1, theta2, k)[1] for k in ks])
    return n


def _loop_panel(ax, n, color, label, winding_txt, inside):
    """Draw one planar (n_y,n_z) loop with the origin and an inside/outside verdict."""
    circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circ), np.sin(circ), color="#dddddd", lw=0.8, zorder=0)
    ax.plot(n[:, 1], n[:, 2], color=color, lw=2.2, label=label)
    ax.plot(0, 0, "o", color=st.ACCENT, ms=6, zorder=5)
    # arrows for the sense of traversal
    for i in (120, 300):
        ax.annotate("", xy=(n[i + 1, 1], n[i + 1, 2]),
                    xytext=(n[i, 1], n[i, 2]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
    ax.axhline(0, color="#eeeeee", lw=0.6, ls=":")
    ax.axvline(0, color="#eeeeee", lw=0.6, ls=":")
    verdict = "origin INSIDE" if inside else "origin OUTSIDE"
    nu_txt = label.split(":")[1].strip()
    # verdict placed OUTSIDE the loop (to the right), where there is free space
    ax.text(1.06, 0.5, f"{winding_txt}\n{verdict}\n$\\Rightarrow$ {nu_txt}",
            transform=ax.transAxes, ha="left", va="center", fontsize=8.5,
            color=color, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.95))
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel(r"$n_y(k)$", labelpad=1)
    ax.set_ylabel(r"$n_z(k)$", labelpad=1)


def main():
    st.apply()
    theta2 = 0.35
    trivial = (0.0, theta2)        # nu = 0  (deep trivial: small arc)
    topo = (0.60, theta2)          # nu = -1
    n_triv = bloch_curve(*trivial)
    n_topo = bloch_curve(*topo)

    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], hspace=0.45, wspace=0.25)

    # ---- (a),(b): the planar winding loops (the real proof) ----
    axa = fig.add_subplot(gs[0, 0])
    _loop_panel(axa, n_topo, st.MOBIUS,
                r"$\theta_1=0.60$:  $\nu=-1$", r"winds once", inside=True)
    axa.set_title(r"Topological loop", fontsize=11)
    st.panel_label(axa, "(a)")

    axb = fig.add_subplot(gs[1, 0])
    _loop_panel(axb, n_triv, st.RING,
                r"$\theta_1=0.00$:  $\nu=0$", r"no net wind", inside=False)
    axb.set_title(r"Trivial loop (an arc, retraced)", fontsize=11)
    st.panel_label(axb, "(b)")

    # ---- (c): same n_hat(k) on the Bloch sphere (intuition only) ----
    ax2 = fig.add_subplot(gs[:, 1], projection="3d")
    uu, vv = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    xs = np.cos(uu) * np.sin(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(vv)
    ax2.plot_surface(xs, ys, zs, color="#dfe6ec", alpha=0.28, linewidth=0,
                     antialiased=True, shade=False, zorder=0)
    for axis in "xyz":
        getattr(ax2, f"plot")([-1, 1] if axis == "x" else [0, 0],
                              [-1, 1] if axis == "y" else [0, 0],
                              [-1, 1] if axis == "z" else [0, 0],
                              color="#b0b8bf", lw=0.7)
    ax2.plot(n_topo[:, 0], n_topo[:, 1], n_topo[:, 2], color=st.MOBIUS, lw=2.6,
             label=r"$\nu=-1$ (full great circle)")
    ax2.plot(n_triv[:, 0], n_triv[:, 1], n_triv[:, 2], color=st.RING, lw=2.6,
             label=r"$\nu=0$ (arc, no wrap)")
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_xlabel(r"$n_x$", labelpad=-8)
    ax2.set_ylabel(r"$n_y$", labelpad=-8)
    ax2.set_zlabel(r"$n_z$", labelpad=-8)
    ax2.set_xticks([-1, 0, 1]); ax2.set_yticks([-1, 0, 1]); ax2.set_zticks([-1, 0, 1])
    ax2.tick_params(labelsize=7, pad=-2)
    ax2.view_init(elev=18, azim=-70)
    ax2.set_title(r"(c)  Same $\hat n(k)$ on the Bloch sphere (intuition)",
                  fontsize=11)
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(r"The winding number of the 1D split-step walk "
                 r"($\theta_2=0.35$)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "winding_sphere.png")
    fig.savefig(out)
    print(f"saved {out}")

    def wrap(n):
        a = np.unwrap(np.arctan2(n[:, 2], n[:, 1]))
        return (a[-1] - a[0]) / (2 * np.pi)
    print(f"(a) planar winding:  nu(topo)={wrap(n_topo):+.2f}   nu(triv)={wrap(n_triv):+.2f}")
    print(f"    max |n_x| on loops: {np.max(np.abs(np.r_[n_topo[:,0], n_triv[:,0]])):.1e} "
          "(planar, as chiral symmetry requires)")


if __name__ == "__main__":
    main()
