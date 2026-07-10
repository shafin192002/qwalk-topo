"""
Example: bulk-boundary correspondence -- topological edge states.

A non-zero bulk winding number is only physically meaningful if it *predicts*
protected boundary modes. This example makes that prediction visible.

We build a split-step walk on a ring whose coin angle theta1(x) is a spatial
domain wall: the left half sits in a topological phase (|theta1| > theta2, so
nu = -1) and the right half in the trivial phase (theta1 = 0, nu = 0). Because
the ring is periodic there are two walls; each binds a protected mode. For this
chiral walk the modes are pinned at quasi-energy eps = +-pi (the split-step
walk's hallmark "pi-modes"), and they are exponentially localised at the walls.

Panel (a): the full quasi-energy spectrum as the topological angle theta1 is
swept. Bulk bands (grey) open a gap around eps = pi; inside it, two edge
branches (orange) appear and lock to eps = pi exactly when |theta1| > theta2.
Panel (b): the probability density |psi(x)|^2 of one such pi-mode, sharply
peaked at the two domain walls.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo import split_step_walk
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")


def domain_wall_angles(N, theta_topo, theta_triv=0.0):
    """theta1(x): topological on the left half, trivial on the right half."""
    th = np.full(N, theta_triv)
    th[: N // 2] = theta_topo
    return th


def wall_weight(psi, N):
    """Fraction of |psi|^2 within +-3 sites of the two walls (x=0 and x=N/2)."""
    dens = (np.abs(psi) ** 2).reshape(N, 2).sum(axis=1)
    walls = list(range(0, 4)) + list(range(N - 3, N))
    walls += list(range(N // 2 - 3, N // 2 + 4))
    return dens[walls].sum()


def spectrum_with_localisation(N, theta_topo, theta2):
    """Return (quasi-energies, wall-weight per state) for one theta_topo."""
    th1 = domain_wall_angles(N, theta_topo)
    U = split_step_walk(N, th1, theta2, topology="ring")
    w, v = np.linalg.eig(U)
    eps = -np.angle(w)
    wts = np.array([wall_weight(v[:, i], N) for i in range(2 * N)])
    return eps, wts


def main():
    st.apply()
    N = 80
    theta2 = 0.30

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    # ---- panel (a): spectrum vs topological angle, edge states highlighted ----
    thetas = np.linspace(0.0, np.pi / 2, 90)
    bulk_k, bulk_e, edge_k, edge_e = [], [], [], []
    for th in thetas:
        eps, wts = spectrum_with_localisation(N, th, theta2)
        for e, wgt in zip(eps, wts):
            if wgt > 0.35:                      # localised at the walls -> edge
                edge_k.append(th); edge_e.append(e)
            else:
                bulk_k.append(th); bulk_e.append(e)

    ax[0].scatter(bulk_k, bulk_e, s=2.5, c=st.BAND, alpha=0.55,
                  edgecolors="none", rasterized=True, label="bulk states")
    ax[0].scatter(edge_k, edge_e, s=9, c=st.EDGE, edgecolors="none",
                  label="edge states")
    ax[0].axvline(theta2, color=st.ACCENT, lw=1.0, ls="--", alpha=0.8)
    ax[0].text(theta2 + 0.03, -2.6, r"$|\theta_1|=\theta_2$"
               "\n(gap closes)", fontsize=8.5, color=st.ACCENT)
    for yy, lab in [(np.pi, r"$+\pi$"), (-np.pi, r"$-\pi$"), (0, "0")]:
        ax[0].axhline(yy, color="#cccccc", lw=0.6, ls=":")
    ax[0].set_xlim(0, np.pi / 2)
    ax[0].set_ylim(-np.pi, np.pi)
    ax[0].set_xticks([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
    ax[0].set_xticklabels(["0", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"])
    ax[0].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[0].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax[0].set_xlabel(r"topological coin angle $\theta_1$ (left half)")
    ax[0].set_ylabel(r"quasi-energy $\varepsilon$")
    ax[0].set_title(r"Spectrum of a domain-wall walk ($\theta_2=0.30$, $N=80$)")
    ax[0].legend(loc="upper right", markerscale=1.4)
    st.panel_label(ax[0], "(a)")

    # ---- panel (b): spatial profile of a pi edge mode ----
    theta_topo = 1.2
    th1 = domain_wall_angles(N, theta_topo)
    U = split_step_walk(N, th1, theta2, topology="ring")
    w, v = np.linalg.eig(U)
    eps = -np.angle(w)
    order = np.argsort(np.abs(np.abs(eps) - np.pi))
    idx = order[0]
    dens = (np.abs(v[:, idx]) ** 2).reshape(N, 2).sum(axis=1)

    x = np.arange(N)
    ax[1].fill_between(x, dens, color=st.EDGE, alpha=0.28, step="mid")
    ax[1].plot(x, dens, color=st.EDGE, lw=1.6, drawstyle="steps-mid",
               label=r"$|\psi(x)|^2$ at $\varepsilon\simeq\pi$")
    ax[1].axvspan(0, N // 2 - 1, color=st.MOBIUS, alpha=0.05)
    for xw in (0, N // 2):
        ax[1].axvline(xw, color=st.ACCENT, lw=1.0, ls="--", alpha=0.75)
    ymax = ax[1].get_ylim()[1]
    ax[1].set_ylim(0, ymax * 1.18)          # headroom for labels + legend
    ymax = ax[1].get_ylim()[1]
    ax[1].text(N // 4, ymax * 0.72, "topological\nhalf",
               ha="center", va="top", fontsize=8.5, color=st.MOBIUS)
    ax[1].text(3 * N // 4, ymax * 0.72, "trivial\nhalf",
               ha="center", va="top", fontsize=8.5, color="#555555")
    ax[1].annotate("walls", xy=(N // 2, dens[N // 2] + ymax * 0.03),
                   xytext=(N // 2 + 9, ymax * 0.45), ha="left", fontsize=8.5,
                   color=st.ACCENT,
                   arrowprops=dict(arrowstyle="->", color=st.ACCENT, lw=0.8))
    ax[1].set_xlim(0, N - 1)
    ax[1].set_xlabel(r"position $x$")
    ax[1].set_ylabel(r"probability density $|\psi(x)|^2$")
    ax[1].set_title(r"A $\pi$-mode localised at the domain walls")
    ax[1].legend(loc="upper center")
    st.panel_label(ax[1], "(b)")

    fig.suptitle("Bulk-boundary correspondence: topological edge states of the "
                 "split-step walk", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "edge_states.png")
    fig.savefig(out)
    print(f"saved {out}")

    n_edge = int(np.sum(spectrum_with_localisation(N, theta_topo, theta2)[1] > 0.35))
    print(f"edge (wall-localised) states at theta1={theta_topo}: {n_edge}")
    print(f"pi-mode wall-localisation: {dens[list(range(4)) + list(range(N-3, N)) + list(range(N//2-3, N//2+4))].sum():.2%} "
          "of the weight sits on the walls")


if __name__ == "__main__":
    main()
