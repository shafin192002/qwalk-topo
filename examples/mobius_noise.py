"""
Example: does non-orientability survive decoherence? (No -- it washes out.)

The Mobius walk differs from the ring only because of the coherent -1 seam twist.
We start the same walker on both, let it reach the seam, and add coin noise.
Panel (a): the ring vs Mobius position distributions -- clearly different at
p = 0, essentially identical once dephased. Panel (b): the total-variation
distance D(p) between them decays to 0 as noise grows. So the non-orientable
signature is a coherent-quantum feature with no decoherence robustness: modest
noise erases the difference between orientable and non-orientable.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo import topology_distinguishability
from qwalktopo.noise import _final_distribution
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")


def main():
    st.apply()
    N, t1, t2, T = 16, 0.6, 0.35, 22
    x0 = N // 2
    x = np.arange(N) - x0

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    # ---- (a) ring vs mobius distributions, clean and dephased ----
    a = ax[0]
    for p, alpha, tag in [(0.0, 1.0, "no noise"), (0.25, 0.5, "$p=0.25$")]:
        pr = _final_distribution(N, t1, t2, T, "dephasing", p, "ring", x0)
        pm = _final_distribution(N, t1, t2, T, "dephasing", p, "mobius", x0)
        a.plot(x, pr, "-o", ms=3, color=st.RING, alpha=alpha,
               label=f"ring, {tag}")
        a.plot(x, pm, "-s", ms=3, color=st.MOBIUS, alpha=alpha,
               label=f"Möbius, {tag}")
    a.set_xlabel(r"position $x-x_0$"); a.set_ylabel(r"probability $P(x)$")
    a.set_title(r"Ring vs Möbius distributions ($N=16$, seam reached)")
    a.legend(fontsize=7.5, loc="upper right"); st.panel_label(a, "(a)")

    # ---- (b) distinguishability D(p) -> 0 ----
    b = ax[1]
    ps = np.linspace(0, 0.5, 21)
    for ch, col in [("dephasing", st.MOBIUS), ("depolarizing", st.RING)]:
        D = [topology_distinguishability(N, t1, t2, T, ch, p) for p in ps]
        b.plot(ps, D, "o-", ms=3.5, color=col, label=ch)
    b.axhline(0, color="#cccccc", lw=0.8, ls=":")
    b.set_xlim(0, 0.5); b.set_ylim(-0.01, None)
    b.set_xlabel(r"noise strength $p$")
    b.set_ylabel(r"ring$\,\leftrightarrow\,$Möbius distance  $D(p)$")
    b.set_title(r"Non-orientable signature decoheres away")
    b.legend(loc="upper right"); st.panel_label(b, "(b)")

    fig.suptitle("Does non-orientability survive noise? "
                 "The ring/Möbius difference washes out", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "mobius_noise.png")
    fig.savefig(out)
    print(f"saved {out}")
    d0 = topology_distinguishability(N, t1, t2, T, "dephasing", 0.0)
    d4 = topology_distinguishability(N, t1, t2, T, "dephasing", 0.4)
    print(f"ring<->mobius distance:  p=0.0 -> {d0:.3f}   p=0.4 -> {d4:.3f} "
          "(non-orientability offers no decoherence robustness)")


if __name__ == "__main__":
    main()
