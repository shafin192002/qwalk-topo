"""
Illustration (not a result): the 1D walk drawn on its Mobius band.

The split-step walk is one-dimensional -- a chain of N sites. What makes the
Mobius walk special is only the boundary: the chain closes back on itself with
an orientation-reversing half-twist. This figure makes that geometry concrete by
embedding the 1D chain as the centre-line of a Mobius strip in 3D and painting
the *actual* walker probability density |psi(x)|^2 (a real time-evolved state of
the Mobius Floquet operator) along it. The seam -- where the strip glues with a
flip -- is highlighted.

This is a communication / hero figure for the README and talks. It is explicitly
an illustration of the geometry, not a piece of evidence; the quantitative
results live in the other four figures.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo import split_step_walk
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")


def walker_density(N, theta1, theta2, steps):
    """|psi(x)|^2 of a wavepacket launched at the seam and evolved on the Mobius band."""
    U = split_step_walk(N, theta1, theta2, "mobius")
    psi = np.zeros(2 * N, dtype=complex)
    x0 = 0                                   # start on the seam site
    psi[2 * x0] = 1 / np.sqrt(2)
    psi[2 * x0 + 1] = 1j / np.sqrt(2)        # balanced coin -> symmetric spread
    for _ in range(steps):
        psi = U @ psi
    return (np.abs(psi) ** 2).reshape(N, 2).sum(axis=1)


def main():
    st.apply()
    N = 60
    theta1, theta2 = 0.6, 0.35
    dens = walker_density(N, theta1, theta2, steps=15)
    dens = dens / dens.max()                 # normalise for colour

    # Mobius strip surface: u around the loop, v across the (half-twisting) width
    R, width = 1.0, 0.34
    nu, nv = N, 16
    u = np.linspace(0, 2 * np.pi, nu, endpoint=True)
    v = np.linspace(-width, width, nv)
    U_, V_ = np.meshgrid(u, v)
    X = (R + V_ * np.cos(U_ / 2)) * np.cos(U_)
    Y = (R + V_ * np.cos(U_ / 2)) * np.sin(U_)
    Z = V_ * np.sin(U_ / 2)

    # colour each ring by the walker density at that site (wrap to match u)
    dens_wrapped = np.concatenate([dens, dens[:1]])
    cvals = np.tile(dens_wrapped, (nv, 1))
    cmap = plt.colormaps["magma"]
    facecolors = cmap(0.12 + 0.88 * cvals)

    fig = plt.figure(figsize=(8.8, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)

    # highlight the seam (u = 0 line across the width) and the mid-line
    us = 0.0
    xs = (R + v * np.cos(us / 2)) * np.cos(us)
    ys = (R + v * np.cos(us / 2)) * np.sin(us)
    zs = v * np.sin(us / 2)
    ax.plot(xs, ys, zs, color="#00e5ff", lw=3.2, label="orientation-reversing seam")
    umid = np.linspace(0, 2 * np.pi, 200)
    ax.plot((R) * np.cos(umid), (R) * np.sin(umid), 0 * umid,
            color="white", lw=0.8, alpha=0.35)

    ax.set_box_aspect((1, 1, 0.42))
    ax.set_axis_off()
    ax.view_init(elev=38, azim=-58)
    ax.set_title("A split-step quantum walk on its Möbius band\n"
                 r"colour = walker density $|\psi(x)|^2$ (time-evolved, "
                 rf"$\theta_1,\theta_2={theta1},{theta2}$)", fontsize=12)

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    cbar = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.02, aspect=18)
    cbar.set_label(r"normalised density $|\psi(x)|^2$")
    ax.legend(loc="upper left", fontsize=9)

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "mobius_render.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"saved {out}")
    print("(illustration of the 1D walk's geometry; density is a real "
          "time-evolved Möbius state)")


if __name__ == "__main__":
    main()
