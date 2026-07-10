"""
Example: the Mobius walk keeps a finite-size gap the ring loses.

At quasi-energy eps = 0 the split-step bulk gap closes at momentum k = 0 when
theta1 = -theta2 (there U(k=0) = C(theta1+theta2) has eigenphases +-(theta1+theta2)).
The ring *always* samples k = 0, so its eps = 0 gap slams shut on that line. The
Mobius walk samples the anti-periodic momenta k = 2*pi*(m+1/2)/N and therefore
never touches k = 0, so it keeps a finite gap at finite N.

That gap is genuinely a finite-size effect: the smallest anti-periodic momentum
is |k| = pi/N, which drifts to 0 as N grows, so the Mobius gap closes as ~1/N.
Panel (a) shows the gap across the closing line at fixed N; panel (b) shows the
1/N scaling exactly on the line.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo import split_step_walk, quasi_energies
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")


def gap_at_zero(N, t1, t2, topology):
    """Spectral gap around quasi-energy 0 = min |eps|."""
    return np.min(np.abs(quasi_energies(split_step_walk(N, t1, t2, topology))))


def main():
    st.apply()
    theta2 = 0.35

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    # ---- panel (a): gap vs theta1 across the k=0 closing line theta1 = -theta2 ----
    N = 24
    t1s = np.linspace(-1.0, 0.3, 261)
    g_ring = np.array([gap_at_zero(N, t1, theta2, "ring") for t1 in t1s])
    g_mob = np.array([gap_at_zero(N, t1, theta2, "mobius") for t1 in t1s])

    ax[0].plot(t1s, g_ring, color=st.RING, lw=1.8, label="ring (periodic)")
    ax[0].plot(t1s, g_mob, color=st.MOBIUS, lw=1.8, label="Möbius (anti-periodic)")
    ax[0].axvline(-theta2, color=st.ACCENT, lw=1.0, ls="--", alpha=0.85)
    ax[0].annotate(r"ring gap $\to 0$ at $\theta_1=-\theta_2$"
                   "\n($k=0$ closes)",
                   xy=(-theta2, 0.0), xytext=(-0.93, 0.28), fontsize=8.5,
                   color=st.ACCENT,
                   arrowprops=dict(arrowstyle="->", color=st.ACCENT, lw=0.8))
    ax[0].annotate("Möbius stays\nopen (no $k=0$)",
                   xy=(-theta2, g_mob[np.argmin(np.abs(t1s + theta2))]),
                   xytext=(-0.30, 0.42), fontsize=8.5, color=st.MOBIUS,
                   arrowprops=dict(arrowstyle="->", color=st.MOBIUS, lw=0.8))
    ax[0].set_xlim(t1s[0], t1s[-1])
    ax[0].set_ylim(0, None)
    ax[0].set_xlabel(r"coin angle $\theta_1$ ($\theta_2=0.35$)")
    ax[0].set_ylabel(r"spectral gap at $\varepsilon=0$,  $\min|\varepsilon|$")
    ax[0].set_title(rf"Finite-size gap across the closing line ($N={N}$)")
    ax[0].legend(loc="upper left")
    st.panel_label(ax[0], "(a)", loc="upper right")

    # ---- panel (b): on the line theta1 = -theta2, gap vs N (1/N scaling) ----
    Ns = np.arange(8, 129, 2)
    g_ring_line = np.array([gap_at_zero(N, -theta2, theta2, "ring") for N in Ns])
    g_mob_line = np.array([gap_at_zero(N, -theta2, theta2, "mobius") for N in Ns])

    ax[1].plot(Ns, g_mob_line, "o", ms=4.5, color=st.MOBIUS,
               label="Möbius gap (data)")
    # analytic guide: gap ~ E(pi/N) ~ (dE/dk|_0) * pi/N ; fit the constant.
    c = np.median(g_mob_line * Ns)
    ax[1].plot(Ns, c / Ns, color=st.MOBIUS, lw=1.2, ls="--", alpha=0.8,
               label=r"$\propto 1/N$ guide")
    ax[1].plot(Ns, g_ring_line, "s", ms=4.5, color=st.RING,
               label="ring gap (data)")
    ax[1].axhline(0, color="#cccccc", lw=0.6, ls=":")
    ax[1].set_xlim(Ns[0], Ns[-1])
    ax[1].set_ylim(-0.01, None)
    ax[1].set_xlabel(r"system size $N$")
    ax[1].set_ylabel(r"gap at $\theta_1=-\theta_2$,  $\min|\varepsilon|$")
    ax[1].set_title(r"On the line: ring $\equiv 0$, Möbius $\sim 1/N$")
    ax[1].legend(loc="upper right")
    st.panel_label(ax[1], "(b)")

    fig.suptitle("Non-orientability opens a finite-size gap: Möbius vs ring",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "finite_size_gap.png")
    fig.savefig(out)
    print(f"saved {out}")
    print(f"on the line theta1=-theta2:  ring gap (N=24) = {g_ring_line[Ns.tolist().index(24)]:.2e}"
          f"   Mobius gap (N=24) = {g_mob_line[Ns.tolist().index(24)]:.4f}")
    print(f"Mobius gap * N is ~constant ({c:.3f}) -> confirms the 1/N closing")


if __name__ == "__main__":
    main()
