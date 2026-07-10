"""
Example: the split-step walk as a quantum circuit (PennyLane).

The same walk that the rest of the package builds by exact diagonalisation is run
here as a gate circuit -- a coin qubit plus n = log2(N) position qubits -- and
sampled on a simulator. The circuit reproduces the exact walk to machine
precision (verified in tests/test_circuit.py); here we overlay the exact
distribution with a finite-shot sampled one, for both the ring and the Mobius
walk (the -1 seam becomes two FlipSign gates).

Requires the optional backend:  pip install qwalk-topo[circuit]
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo import split_step_walk
from qwalktopo.circuit import run_walk
import _style as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")


def ed_distribution(N, t1, t2, T, topology):
    U = split_step_walk(N, t1, t2, topology)
    x0 = N // 2
    psi = np.zeros(2 * N, dtype=complex)
    psi[2 * x0:2 * x0 + 2] = np.array([1, 1j]) / np.sqrt(2)
    for _ in range(T):
        psi = U @ psi
    return np.arange(N) - x0, (np.abs(psi) ** 2).reshape(N, 2).sum(1)


def main():
    st.apply()
    N, t1, t2, T, shots = 16, 0.6, 0.35, 6, 20000

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for a, topo, col in [(ax[0], "ring", st.RING), (ax[1], "mobius", st.MOBIUS)]:
        x, p_ed = ed_distribution(N, t1, t2, T, topo)
        p_shots = run_walk(N, t1, t2, T, topology=topo, shots=shots)
        a.plot(x, p_ed, "-", color=st.ACCENT, lw=1.8,
               label="exact diagonalisation")
        a.bar(x, p_shots, width=0.7, color=col, alpha=0.45,
              label=f"circuit, {shots} shots")
        a.set_xlabel(r"position $x-x_0$")
        a.set_title(f"{'Ring' if topo=='ring' else 'Möbius'} walk "
                    f"({int(np.log2(N))}+1 qubits)")
        a.legend(loc="upper right", fontsize=8.5)
    ax[0].set_ylabel(r"probability $P(x)$")
    st.panel_label(ax[0], "(a)"); st.panel_label(ax[1], "(b)")

    fig.suptitle("The walk as a quantum circuit: circuit sampling reproduces "
                 "exact diagonalisation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "circuit_backend.png")
    fig.savefig(out)
    print(f"saved {out}")
    x, p_ed = ed_distribution(N, t1, t2, T, "mobius")
    p_ex = run_walk(N, t1, t2, T, topology="mobius")
    print(f"circuit(exact) vs ED, mobius: max|dP| = {np.max(np.abs(p_ex - p_ed)):.2e}")


if __name__ == "__main__":
    main()
