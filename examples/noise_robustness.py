"""
Example: does the topology survive decoherence? Symmetry decides.

Evolves the split-step walk as a density matrix under coin noise and tracks the
mean chiral displacement (MCD, -> winding number ν), the position distribution,
and the spreading exponent. Every panel shows the clean (p = 0) walk together
with the noisy (p > 0) walk.

The message: it is not the amount of noise that matters but the symmetry it
respects. Bit-flip noise (σ_x = the chiral operator Γ) leaves the topological MCD
essentially pinned at ν even at strong p; dephasing (σ_z) and depolarizing noise
break the protecting chiral symmetry and drive C -> 0, while turning ballistic
quantum spreading (Var ~ t²) into diffusive classical spreading (Var ~ t).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo import (run_noisy_walk, mean_chiral_displacement, coin_channel,
                       symmetric_walk, winding_number)
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")

CH = {"bitflip": (st.EDGE, "bit-flip $\\sigma_x$ (chiral-symmetric)"),
      "dephasing": (st.MOBIUS, "dephasing $\\sigma_z$ (breaks chiral)"),
      "depolarizing": (st.RING, "depolarizing (breaks all)")}


def final_distribution(N, t1, t2, steps, channel, p):
    """Position probability distribution P(x) after `steps` noisy steps."""
    U = symmetric_walk(N, t1, t2, "ring")
    Ks = [np.kron(np.eye(N, dtype=complex), k) for k in coin_channel(channel, p)]
    x0 = N // 2
    rho = np.zeros((2 * N, 2 * N), dtype=complex)
    rho[2 * x0, 2 * x0] = rho[2 * x0 + 1, 2 * x0 + 1] = 0.5
    for _ in range(steps):
        rho = U @ rho @ U.conj().T
        rho = sum(K @ rho @ K.conj().T for K in Ks)
    d = np.real(np.diag(rho)).reshape(N, 2).sum(1)
    return np.arange(N) - x0, d


def main():
    st.apply()
    N, t1, t2, T = 61, 0.6, 0.35, 26
    nu = winding_number(t1, t2)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))

    # ---- (a) MCD(t): clean and each channel at p = 0.3 ----
    a = ax[0, 0]
    clean = run_noisy_walk(N, t1, t2, T, "none", 0.0)
    a.plot(clean["t"], clean["mcd"], color=st.ACCENT, lw=2.0, label="no noise")
    for ch, (col, lab) in CH.items():
        o = run_noisy_walk(N, t1, t2, T, ch, 0.3)
        a.plot(o["t"], o["mcd"], color=col, lw=1.6, alpha=0.9,
               label=lab.split(" (")[0])
    a.axhline(nu, color="#999999", lw=0.9, ls="--")
    a.text(1.5, nu + 0.12, rf"$\nu={nu}$", fontsize=9, color="#666666")
    a.set_xlim(1, T); a.set_ylim(nu - 0.9, 0.4)
    a.set_xlabel("time step $t$"); a.set_ylabel(r"mean chiral displacement $C(t)$")
    a.set_title(r"$C(t)\to\nu$; noise at $p=0.3$")
    a.legend(loc="lower right", fontsize=8); st.panel_label(a, "(a)")

    # ---- (b) MCD (time-averaged) vs noise strength p -- the robustness curve ----
    b = ax[0, 1]
    ps = np.linspace(0, 0.5, 11)
    for ch, (col, lab) in CH.items():
        cs = [mean_chiral_displacement(N, t1, t2, T, ch, p) for p in ps]
        b.plot(ps, cs, "o-", ms=4, color=col, label=lab)
    b.axhline(nu, color="#999999", lw=0.9, ls="--")
    b.text(0.02, nu + 0.06, rf"$\nu={nu}$ (topological)", fontsize=8.5, color="#666666")
    b.axhline(0, color="#cccccc", lw=0.8, ls=":")
    b.set_xlim(0, 0.5); b.set_ylim(nu - 0.25, 0.3)
    b.set_xlabel(r"noise strength $p$")
    b.set_ylabel(r"time-averaged $C\ (\to \nu)$")
    b.set_title(r"Topology vs noise: symmetry decides")
    b.legend(loc="upper right", fontsize=8); st.panel_label(b, "(b)")

    # ---- (c) position distribution: quantum (ballistic) -> classical (diffusive) ----
    c = ax[1, 0]
    x, d0 = final_distribution(N, t1, t2, T, "none", 0.0)
    _, dd = final_distribution(N, t1, t2, T, "dephasing", 0.3)
    c.plot(x, d0, color=st.ACCENT, lw=1.8, label="no noise (ballistic)")
    c.fill_between(x, d0, color=st.ACCENT, alpha=0.08)
    c.plot(x, dd, color=st.MOBIUS, lw=1.8, label="dephasing $p=0.3$ (diffusive)")
    c.fill_between(x, dd, color=st.MOBIUS, alpha=0.12)
    c.set_xlim(x.min(), x.max())
    c.set_xlabel(r"position $x-x_0$"); c.set_ylabel(r"probability $P(x)$")
    c.set_title(r"Quantum-to-classical transition")
    c.legend(loc="upper right", fontsize=8.5); st.panel_label(c, "(c)")

    # ---- (d) spreading exponent: Var ~ t^a (2 ballistic, 1 diffusive) ----
    d = ax[1, 1]
    for ch, p, col, lab in [("none", 0.0, st.ACCENT, "no noise"),
                            ("dephasing", 0.3, st.MOBIUS, "dephasing $p=0.3$")]:
        o = run_noisy_walk(N, t1, t2, T, ch, p)
        tt, vv = o["t"], o["variance"]
        d.plot(tt, vv, "o", ms=3.5, color=col, label=lab)
        sl = np.polyfit(np.log(tt[4:]), np.log(vv[4:]), 1)[0]
        d.plot(tt[4:], np.exp(np.polyval(np.polyfit(np.log(tt[4:]), np.log(vv[4:]), 1),
               np.log(tt[4:]))), color=col, lw=1.2, alpha=0.7,
               label=rf"fit: Var$\sim t^{{{sl:.2f}}}$")
    d.set_xscale("log"); d.set_yscale("log")
    d.set_xlabel(r"time step $t$"); d.set_ylabel(r"position variance")
    d.set_title(r"Ballistic ($t^2$) $\to$ diffusive ($t^1$)")
    d.legend(loc="lower right", fontsize=8); st.panel_label(d, "(d)")

    fig.suptitle("Does the topology survive decoherence? "
                 "Symmetry-respecting noise protects it", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "noise_robustness.png")
    fig.savefig(out)
    print(f"saved {out}")
    print(f"nu={nu}. time-avg MCD at p=0.3:  "
          f"bitflip={mean_chiral_displacement(N,t1,t2,T,'bitflip',0.3):+.3f}  "
          f"dephasing={mean_chiral_displacement(N,t1,t2,T,'dephasing',0.3):+.3f}  "
          f"depol={mean_chiral_displacement(N,t1,t2,T,'depolarizing',0.3):+.3f}")


if __name__ == "__main__":
    main()
