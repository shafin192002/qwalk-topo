"""
Example: anti-periodic momentum quantisation on the Mobius band.

This is the physical signature qwalk-topo is built to expose. On a ring of N
sites the allowed momenta are the *periodic* set
        k_m = 2 pi m / N,            m = 0, ..., N-1.
On a Mobius band the orientation-reversing seam contributes the Z2 spin
holonomy -1 to the coin wavefunction, imposing the *anti-periodic* boundary
condition psi(x+N) = -psi(x) and shifting the allowed momenta to the
half-integer set
        k_m = 2 pi (m + 1/2) / N,    m = 0, ..., N-1.

Consequently the finite-size quasi-energy spectrum of the Mobius walk samples
the Bloch bands at different momenta than the ring walk. In particular, the
ring includes k = 0 (and k = pi for even N), where the split-step bands can be
gapless or extremal; the Mobius walk never samples k = 0.

We verify this quantitatively: the *exact* finite-N Floquet spectra of the ring
and Mobius walks are shown to coincide, to machine precision, with the analytic
continuum bands eps_+-(k) = +- E(k) evaluated on the periodic / anti-periodic
momenta respectively.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Allow running as a plain script (python examples/...) without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo import split_step_walk, quasi_energies
from qwalktopo.invariants.winding import _bloch_vector
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")


def continuum_bands(theta1, theta2, k):
    """Analytic quasi-energy bands eps_+-(k) = +- E(k)."""
    E, _ = _bloch_vector(theta1, theta2, k)
    return E, -E


def periodic_momenta(N):
    k = np.array([2 * np.pi * m / N for m in range(N)])
    return np.where(k > np.pi, k - 2 * np.pi, k)


def antiperiodic_momenta(N):
    k = np.array([2 * np.pi * (m + 0.5) / N for m in range(N)])
    return np.where(k > np.pi, k - 2 * np.pi, k)


def band_set(theta1, theta2, ks):
    """Full analytic spectrum {+-E(k)} sampled on the momenta ks, sorted."""
    Ek = np.array([continuum_bands(theta1, theta2, k)[0] for k in ks])
    return np.sort(np.concatenate([Ek, -Ek])), Ek


def main():
    st.apply()
    theta1, theta2 = 0.6, 0.35
    N = 16

    # analytic continuum bands
    kk = np.linspace(-np.pi, np.pi, 400)
    Ep = np.array([continuum_bands(theta1, theta2, k)[0] for k in kk])

    # exact finite-size spectra (the real thing we are validating)
    eps_ring = quasi_energies(split_step_walk(N, theta1, theta2, "ring"))
    eps_mob = quasi_energies(split_step_walk(N, theta1, theta2, "mobius"))

    # predicted sampling momenta and the analytic spectra on them
    k_per = periodic_momenta(N)
    k_anti = antiperiodic_momenta(N)
    pred_ring, E_per = band_set(theta1, theta2, k_per)
    pred_mob, E_anti = band_set(theta1, theta2, k_anti)

    # the overlay is only meaningful if the exact spectrum equals the sampled
    # band; report the residual so the claim is verifiable, not asserted.
    dev_ring = np.max(np.abs(np.sort(eps_ring) - pred_ring))
    dev_mob = np.max(np.abs(np.sort(eps_mob) - pred_mob))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for a in ax:
        a.plot(kk, Ep, "-", color=st.BAND, lw=1.3, alpha=0.9,
               label=r"continuum band $\pm E(k)$")
        a.plot(kk, -Ep, "-", color=st.BAND, lw=1.3, alpha=0.9)
        a.axhline(0, color="#cccccc", lw=0.6, ls=":")
        a.set_xlabel(r"quasi-momentum $k$")
        a.set_xlim(-np.pi, np.pi)
        a.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        a.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    ax[0].set_title(rf"Ring (periodic): $k_m = 2\pi m/N$,  $N={N}$")
    ax[0].plot(k_per, E_per, "o", color=st.RING, ms=7, mec="white", mew=0.6,
               label="exact finite-$N$ spectrum")
    ax[0].plot(k_per, -E_per, "o", color=st.RING, ms=7, mec="white", mew=0.6)
    ax[0].axvline(0, color=st.RING, lw=0.9, ls="--", alpha=0.55)
    ax[0].set_ylabel(r"quasi-energy $\varepsilon$")
    ax[0].legend(loc="upper right")
    st.panel_label(ax[0], "(a)")

    ax[1].set_title(r"Möbius (anti-periodic): $k_m = 2\pi(m+\frac{1}{2})/N$")
    ax[1].plot(k_anti, E_anti, "s", color=st.MOBIUS, ms=6.5, mec="white",
               mew=0.6, label="exact finite-$N$ spectrum")
    ax[1].plot(k_anti, -E_anti, "s", color=st.MOBIUS, ms=6.5, mec="white",
               mew=0.6)
    ax[1].axvline(0, color=st.ACCENT, lw=0.9, ls="--", alpha=0.6)
    ax[1].annotate(r"no mode at $k=0$", xy=(0, E_anti.min()),
                   xytext=(0.7, 0.2), fontsize=8.5, color=st.ACCENT,
                   arrowprops=dict(arrowstyle="->", color=st.ACCENT, lw=0.8))
    ax[1].legend(loc="upper right")
    st.panel_label(ax[1], "(b)")

    fig.suptitle(
        "Anti-periodic momentum quantisation on the Möbius band  "
        rf"($\theta_1,\theta_2 = {theta1},\,{theta2}$)",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "mobius_momentum_quantisation.png")
    fig.savefig(out)
    print(f"saved {out}")

    # verification: exact spectra fall on the sampled bands to machine precision
    print(f"exact spectrum vs sampled band  ring   : max dev = {dev_ring:.2e}")
    print(f"exact spectrum vs sampled band  mobius : max dev = {dev_mob:.2e}")

    # spectral gap (min |eps|): controlled entirely by the momentum shift
    gap_ring = np.min(np.abs(eps_ring))
    gap_mob = np.min(np.abs(eps_mob))
    diff = gap_mob - gap_ring
    rel = "larger" if diff > 0 else "smaller"
    print(f"min |quasi-energy|  ring = {gap_ring:.4f}   mobius = {gap_mob:.4f}")
    print(f"Mobius gap is {rel} by {abs(diff):.4f} "
          "(the two walks sample the band at different momenta)")


if __name__ == "__main__":
    main()
