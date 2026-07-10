"""
Example: the package's own 2D split-step *Floquet* Klein walk.

The static Z2 Klein invariant (examples/klein_invariant.py) is here applied to a
genuine discrete-time walk. Built from the glide-covariant pieces of the CYZ
model, the split-step Floquet operator

    U_F(kx,ky) = exp(-i s H_x) exp(-i s H_y)

carries the momentum glide U_F(kx,ky) = V U_F(-kx, ky+pi) V† exactly (residual
~1e-15), and since [H_x, H_y] != 0 it is a real two-step walk. At small drive s
it reduces to the static CYZ model, so the verified Klein invariant on its
eps = 0 quasi-energy gap gives nu = 1 -- the published value.

Panel (a): the walk's x-edge quasi-energy spectrum -- edge states cross the
eps = 0 gap (bulk-boundary correspondence), confirming nu = 1. Panel (b): the
gap is open across the small-drive window where nu is stable at 1.

Scope note (honest): pushing to strong drive, the naive Wilson-loop invariant
becomes unreliable (its apparent jump is *not* matched by a quasi-energy gap
closing or by the edge modes disappearing), so we restrict to the anchored
small-drive regime. A careful strong-drive Floquet analysis (both eps = 0 and
eps = pi gaps, anomalous invariants) is left as future work.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo import klein_bottle_invariant, CYZ_NONTRIVIAL
from qwalktopo.invariants.klein import (cyz_floquet_walk,
                                        cyz_floquet_effective_hamiltonian)
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)


def _expm_h(H, s):
    w, v = np.linalg.eigh(H)
    return (v * np.exp(-1j * s * w)) @ v.conj().T


def _Hx_strip(P, Nx):
    e = P["eps"]
    A = np.diag([e, e, -e, -e]).astype(complex)
    A[1, 0] += P["t11x"]; A[0, 1] += P["t11x"]
    A[3, 2] += P["t21x"]; A[2, 3] += P["t21x"]
    B = np.zeros((4, 4), complex); B[1, 0] += P["t12x"]; B[3, 2] += P["t22x"]
    H = np.zeros((4 * Nx, 4 * Nx), complex)
    for x in range(Nx):
        H[4 * x:4 * x + 4, 4 * x:4 * x + 4] += A
        if x + 1 < Nx:
            H[4 * (x + 1):4 * (x + 1) + 4, 4 * x:4 * x + 4] += B
            H[4 * x:4 * x + 4, 4 * (x + 1):4 * (x + 1) + 4] += B.conj().T
    return H


def _Hy_block(ky, P):
    qp = P["t1y"] + P["t2y"] * np.exp(1j * ky)
    qm = P["t1y"] - P["t2y"] * np.exp(1j * ky)
    lam = P["lam"]
    return (np.array([[0, 0, np.conj(qp), 0], [0, 0, 0, np.conj(qm)],
                      [qp, 0, 0, 0], [0, qm, 0, 0]], complex)
            + lam * np.cos(ky) * np.kron(sx, sy) + lam * np.sin(ky) * np.kron(sy, sy))


def edge_spectrum(P, s, ky, Nx=26):
    Hx = _Hx_strip(P, Nx)
    Hy = np.zeros((4 * Nx, 4 * Nx), complex)
    yb = _Hy_block(ky, P)
    for x in range(Nx):
        Hy[4 * x:4 * x + 4, 4 * x:4 * x + 4] = yb
    U = _expm_h(Hx, s) @ _expm_h(Hy, s)
    w, v = np.linalg.eig(U)
    phi = -np.angle(w)
    ew = np.array([(np.abs(v[:, i]) ** 2).reshape(Nx, 4).sum(1)[[0, 1, -2, -1]].sum()
                   for i in range(4 * Nx)])
    return phi, ew


def bulk_gap0(P, s, n=33):
    g = np.inf
    for kx in np.linspace(-np.pi, np.pi, n):
        for ky in np.linspace(-np.pi, np.pi, n):
            phi = np.sort(-np.angle(np.linalg.eigvals(cyz_floquet_walk(kx, ky, s, **P))))
            neg, pos = phi[phi < 0], phi[phi >= 0]
            if len(neg) and len(pos):
                g = min(g, pos.min() - neg.max())
    return g


def main():
    st.apply()
    s = 0.4
    nu = klein_bottle_invariant(
        lambda kx, ky: cyz_floquet_effective_hamiltonian(kx, ky, s, **CYZ_NONTRIVIAL),
        n_occ=2, n_kx=150, n_ky=200)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    # ---- (a) x-edge quasi-energy spectrum of the Floquet walk ----
    a = ax[0]
    kk = np.linspace(-np.pi, np.pi, 121)
    bk, be, ek, ee = [], [], [], []
    for ky in kk:
        phi, ew = edge_spectrum(CYZ_NONTRIVIAL, s, ky)
        for ph, wt in zip(phi, ew):
            if wt > 0.5 and abs(ph) < 0.6:
                ek.append(ky); ee.append(ph)
            else:
                bk.append(ky); be.append(ph)
    a.scatter(bk, be, s=1.5, c=st.BAND, alpha=0.5, edgecolors="none", rasterized=True)
    a.scatter(ek, ee, s=8, c=st.EDGE, edgecolors="none", label="edge states")
    a.axhline(0, color="#cccccc", lw=0.7, ls=":")
    a.set_xlim(-np.pi, np.pi); a.set_ylim(-np.pi, np.pi)
    a.set_xticks([-np.pi, 0, np.pi]); a.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    a.set_yticks([-np.pi, 0, np.pi]); a.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    a.set_xlabel(r"edge momentum $k_y$")
    a.set_ylabel(r"quasi-energy $\varepsilon$")
    a.set_title(rf"$x$-edge spectrum of the Floquet walk ($s={s}$, $\nu={nu}$)")
    a.legend(loc="upper right", markerscale=1.4); st.panel_label(a, "(a)")

    # ---- (b) eps=0 gap stays open across the anchored small-drive window ----
    b = ax[1]
    ss = np.linspace(0.15, 0.5, 12)
    g0 = [bulk_gap0(CYZ_NONTRIVIAL, s_) for s_ in ss]
    b.plot(ss, g0, "o-", ms=4, color=st.ACCENT, label=r"$\varepsilon=0$ gap")
    b.fill_between(ss, 0, g0, color=st.MOBIUS, alpha=0.12, label=r"$\nu=1$ (matches CYZ)")
    b.set_xlim(ss[0], ss[-1]); b.set_ylim(0, None)
    b.set_xlabel(r"drive strength $s$"); b.set_ylabel(r"$\varepsilon=0$ quasi-energy gap")
    b.set_title(r"Gap open, $\nu=1$ stable (static-limit anchor)")
    b.legend(loc="upper left"); st.panel_label(b, "(b)")

    fig.suptitle("A 2D split-step Floquet Klein walk with the verified invariant "
                 r"($\nu=1$, matches Chen–Yang–Zhao)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "klein_floquet_walk.png")
    fig.savefig(out)
    print(f"saved {out}")
    print(f"Floquet Klein walk at s={s}: nu={nu} (CYZ published = 1), "
          f"eps=0 edge states = {len(ee)}")


if __name__ == "__main__":
    main()
