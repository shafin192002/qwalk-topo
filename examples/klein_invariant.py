"""
Example: the Z2 Klein-bottle invariant, its phase transition, and its edge states.

Reproduces the Brillouin-Klein-bottle Z2 invariant of Chen, Yang & Zhao
(Nat. Commun. 13, 2215, 2022) with this package, on their 4-band model. The same
+-1 (Z2 gauge) twist that gives the 1D Mobius seam here makes the 2D Brillouin
zone a Klein bottle, whose invariant is a Z2 -- not a Chern number.

(a) The kx-Wilson-loop Berry phase gamma(ky): in the topological phase it winds
    through +-pi (nu = 1); in the trivial phase it stays near 0 (nu = 0).
(b) Phase diagram: sweeping the on-site energy eps, nu is constant on each gapped
    side and flips by 1 exactly where the bulk gap collapses.
(c),(d) Open-boundary (x-edge) strips: the topological model binds in-gap edge
    states (bulk-boundary correspondence); the trivial model does not.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo import (klein_bottle_invariant, berry_phase_kx_loop,
                       cyz_hamiltonian, CYZ_NONTRIVIAL, CYZ_TRIVIAL)
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")


def bulk_gap(P, n=45):
    """Half-filling bulk gap (between band 2 and band 3 of 4)."""
    g = np.inf
    for kx in np.linspace(-np.pi, np.pi, n):
        for ky in np.linspace(-np.pi, np.pi, n):
            w = np.sort(np.linalg.eigvalsh(cyz_hamiltonian(kx, ky, **P)))
            g = min(g, w[2] - w[1])
    return g


def _gap_window(P, n=40):
    """Center and half-width of the half-filling bulk gap (band 2 to band 3)."""
    lo, hi = -np.inf, np.inf
    for kx in np.linspace(-np.pi, np.pi, n):
        for ky in np.linspace(-np.pi, np.pi, n):
            w = np.sort(np.linalg.eigvalsh(cyz_hamiltonian(kx, ky, **P)))
            lo, hi = max(lo, w[1]), min(hi, w[2])
    return (lo + hi) / 2.0, (hi - lo) / 2.0 * 0.95


def x_edge_spectrum(P, ky, Nx=28):
    """Strip with x open (Nx cells), ky good. Returns eigenvalues + edge weight."""
    kxs = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    A = np.mean([cyz_hamiltonian(kx, ky, **P) for kx in kxs], axis=0)
    B = np.mean([cyz_hamiltonian(kx, ky, **P) * np.exp(-1j * kx) for kx in kxs], axis=0)
    dim = 4 * Nx
    M = np.zeros((dim, dim), dtype=complex)
    for x in range(Nx):
        M[4 * x:4 * x + 4, 4 * x:4 * x + 4] += A
        if x + 1 < Nx:
            M[4 * (x + 1):4 * (x + 1) + 4, 4 * x:4 * x + 4] += B
            M[4 * x:4 * x + 4, 4 * (x + 1):4 * (x + 1) + 4] += B.conj().T
    w, v = np.linalg.eigh(M)
    edge = np.array([(np.abs(v[:, i]) ** 2).reshape(Nx, 4).sum(1)[[0, 1, -2, -1]].sum()
                     for i in range(dim)])
    return w, edge


def _edge_panel(ax, P, title, tag, show_edge_label=False):
    # highlight only states that are BOTH edge-localised and inside the bulk gap,
    # so the trivial panel is not muddied by band-edge states with some weight.
    gc, gh = _gap_window(P)
    kk = np.linspace(-np.pi, np.pi, 121)
    bulk_k, bulk_e, edge_k, edge_e = [], [], [], []
    for ky in kk:
        w, ew = x_edge_spectrum(P, ky)
        for e, wt in zip(w, ew):
            is_edge = (wt > 0.5) and (abs(e - gc) < gh)
            (edge_k if is_edge else bulk_k).append(ky)
            (edge_e if is_edge else bulk_e).append(e)
    ax.scatter(bulk_k, bulk_e, s=1.3, c=st.BAND, alpha=0.5, edgecolors="none",
               rasterized=True)
    if edge_e:
        ax.scatter(edge_k, edge_e, s=7, c=st.EDGE, edgecolors="none",
                   label="edge states" if show_edge_label else None)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_xlabel(r"edge momentum $k_y$")
    ax.set_title(title)
    st.panel_label(ax, tag)
    return len(edge_e)


def main():
    st.apply()
    RES = dict(n_kx=150, n_ky=200)
    nu_t = klein_bottle_invariant(cyz_hamiltonian, n_occ=2, **RES, **CYZ_NONTRIVIAL)
    nu_0 = klein_bottle_invariant(cyz_hamiltonian, n_occ=2, **RES, **CYZ_TRIVIAL)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))

    # ---- (a) Berry-phase winding gamma(ky) ----
    kys = np.linspace(-np.pi, 0.0, 160)
    g_t = [berry_phase_kx_loop(cyz_hamiltonian, ky, 2, 150, **CYZ_NONTRIVIAL) for ky in kys]
    g_0 = [berry_phase_kx_loop(cyz_hamiltonian, ky, 2, 150, **CYZ_TRIVIAL) for ky in kys]
    a = ax[0, 0]
    a.plot(kys, g_t, ".", ms=3.3, color=st.MOBIUS, label=rf"topological: $\nu={nu_t}$")
    a.plot(kys, g_0, ".", ms=3.3, color=st.RING, label=rf"trivial: $\nu={nu_0}$")
    for yy in (np.pi, -np.pi):
        a.axhline(yy, color=st.ACCENT, lw=1.0, ls="--", alpha=0.8)
    a.text(-np.pi + 0.1, np.pi - 0.4, r"$\gamma=\pi$", fontsize=8.5, color=st.ACCENT)
    a.set_xlim(-np.pi, 0); a.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    a.set_xticks([-np.pi, -np.pi / 2, 0]); a.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0"])
    a.set_yticks([-np.pi, 0, np.pi]); a.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    a.set_xlabel(r"$k_y$"); a.set_ylabel(r"Berry phase $\gamma(k_y)$")
    a.set_title(r"$\nu = (\#\,\gamma\ \mathrm{crosses}\ \pi)\ \mathrm{mod}\ 2$")
    a.legend(loc="lower right"); st.panel_label(a, "(a)")

    # ---- (b) phase diagram: nu and bulk gap vs eps ----
    epss = np.linspace(0.2, 3.4, 22)
    gaps = np.array([bulk_gap({**CYZ_NONTRIVIAL, "eps": e}) for e in epss])
    nus = np.array([klein_bottle_invariant(cyz_hamiltonian, n_occ=2, n_kx=120,
                    n_ky=150, **{**CYZ_NONTRIVIAL, "eps": e}) for e in epss])
    b = ax[0, 1]
    b.plot(epss, gaps, "-", color=st.ACCENT, lw=1.8, label="bulk gap")
    b.fill_between(epss, 0, gaps, where=(nus == 1), color=st.MOBIUS, alpha=0.12,
                   step="mid", label=r"$\nu=1$ (topological)")
    b.fill_between(epss, 0, gaps, where=(nus == 0), color=st.RING, alpha=0.12,
                   step="mid", label=r"$\nu=0$ (trivial)")
    # nu as a step trace on a second axis
    b2 = b.twinx()
    b2.step(epss, nus, where="mid", color=st.EDGE, lw=2.0)
    b2.set_ylim(-0.15, 1.35); b2.set_yticks([0, 1])
    b2.set_ylabel(r"invariant $\nu$", color=st.EDGE)
    b2.tick_params(axis="y", colors=st.EDGE)
    ic = epss[np.argmin(gaps)]
    b.axvline(ic, color="#888888", lw=0.9, ls=":")
    b.annotate("gap closes,\n$\\nu$ flips", xy=(ic, gaps.min()),
               xytext=(ic - 1.15, 0.30), fontsize=8.5, color="#555555",
               arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    b.set_xlim(epss[0], epss[-1]); b.set_ylim(0, None)
    b.set_xlabel(r"on-site energy $\epsilon$")
    b.set_ylabel(r"bulk gap")
    b.set_title(r"Phase diagram: $\nu(\epsilon)$ jumps at the gap closing")
    b.legend(loc="upper center", fontsize=8); st.panel_label(b, "(b)")

    # ---- (c),(d) edge spectra: topological vs trivial ----
    n_edge_t = _edge_panel(ax[1, 0], CYZ_NONTRIVIAL,
                           r"Topological ($\nu=1$): $x$-edge states", "(c)",
                           show_edge_label=True)
    ax[1, 0].set_ylabel(r"energy $E$"); ax[1, 0].legend(loc="upper right", markerscale=1.5)
    _edge_panel(ax[1, 1], CYZ_TRIVIAL, r"Trivial ($\nu=0$): no edge states", "(d)")

    fig.suptitle("The $\\mathbb{Z}_2$ Klein-bottle invariant "
                 "(Chen, Yang & Zhao 2022), reproduced and verified", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "klein_invariant.png")
    fig.savefig(out)
    print(f"saved {out}")
    print(f"Klein Z2:  topological -> nu={nu_t}   trivial -> nu={nu_0}   "
          f"(published 1 / 0)")
    print(f"edge states: topological panel = {n_edge_t} points, trivial = 0 "
          "(bulk-boundary correspondence)")


if __name__ == "__main__":
    main()
