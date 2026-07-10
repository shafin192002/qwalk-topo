"""
Quantum-circuit backend: the split-step walk as a gate circuit (PennyLane).

Everything else in this package is exact linear algebra on the (2N)-dimensional
walk operator. This module maps the *same* walk onto a qubit circuit -- a coin
qubit plus ``n = log2(N)`` position qubits -- so it can run on a simulator or on
hardware. It is verified to reproduce the exact-diagonalisation walk to machine
precision (see tests/test_circuit.py):

  * ``walk_unitary`` equals ``split_step_walk`` (ring and Mobius) up to global
    phase;
  * the sampled position distribution matches the exact dynamics.

Circuit dictionary
------------------
  coin C(theta) = exp(-i theta sigma_y)        ->  RY(2 theta) on the coin qubit
  S_+  (move coin=0 by +1)                     ->  increment(mod 2^n) ctrl on coin=0
  S_-  (move coin=1 by -1)                     ->  decrement(mod 2^n) ctrl on coin=1
  Mobius seam (-1 on the wrap bond)            ->  FlipSign on the wrapping state

PennyLane is an *optional* dependency (``pip install qwalk-topo[circuit]``); the
rest of the package never imports it.
"""
from __future__ import annotations

import numpy as np


def _pennylane():
    try:
        import pennylane as qml
        return qml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the circuit backend needs PennyLane: pip install qwalk-topo[circuit] "
            "(or pip install pennylane)") from exc


def _n_qubits(N: int) -> int:
    n = int(round(np.log2(N)))
    if 2 ** n != N:
        raise ValueError("circuit backend requires N to be a power of 2")
    return n


def walk_gates(n: int, theta1: float, theta2: float, steps: int = 1,
               topology: str = "ring") -> None:
    """Apply ``steps`` of the split-step walk to position wires 0..n-1 and coin wire n.

    Call inside a PennyLane QNode. N = 2^n position sites.
    """
    qml = _pennylane()
    pos = list(range(n))
    coin = n
    N = 2 ** n

    def increment(cval):        # +1 mod 2^n gated on coin==cval; MSB flip first
        for i in reversed(range(n)):
            ctrls = [coin] + [pos[n - 1 - j] for j in range(i)]
            qml.ctrl(qml.PauliX(n - 1 - i), control=ctrls,
                     control_values=[cval] + [1] * i)

    def decrement(cval):        # -1 = inverse of increment
        for i in range(n):
            ctrls = [coin] + [pos[n - 1 - j] for j in range(i)]
            qml.ctrl(qml.PauliX(n - 1 - i), control=ctrls,
                     control_values=[cval] + [1] * i)

    for _ in range(steps):
        qml.RY(2 * theta1, wires=coin)
        if topology == "mobius":
            qml.FlipSign(2 * N - 2, wires=pos + [coin])   # -1 on |x=N-1, coin=0>
        increment(0)
        qml.RY(2 * theta2, wires=coin)
        if topology == "mobius":
            qml.FlipSign(1, wires=pos + [coin])           # -1 on |x=0, coin=1>
        decrement(1)


def walk_unitary(N: int, theta1: float, theta2: float,
                 topology: str = "ring") -> np.ndarray:
    """(2N)x(2N) unitary of one circuit step -- should equal split_step_walk."""
    qml = _pennylane()
    n = _n_qubits(N)
    return qml.matrix(lambda: walk_gates(n, theta1, theta2, 1, topology),
                      wire_order=list(range(n)) + [n])()


def run_walk(N: int, theta1: float, theta2: float, steps: int,
             x0: int | None = None, topology: str = "ring",
             shots: int | None = None) -> np.ndarray:
    """Run the circuit and return the position probability distribution P(x).

    Starts a walker localised at ``x0`` (default N//2) with the balanced coin
    (|0>+i|1>)/sqrt(2). With ``shots`` set, samples on the simulator; otherwise
    returns exact probabilities.
    """
    qml = _pennylane()
    n = _n_qubits(N)
    if x0 is None:
        x0 = N // 2
    dev = qml.device("default.qubit", wires=n + 1, shots=shots)

    @qml.qnode(dev)
    def circuit():
        for b in range(n):                              # prepare |x0>
            if (x0 >> (n - 1 - b)) & 1:
                qml.PauliX(b)
        qml.RY(np.pi / 2, wires=n)                      # balanced coin
        qml.S(wires=n)
        walk_gates(n, theta1, theta2, steps, topology)
        return qml.probs(wires=list(range(n)) + [n])

    probs = np.asarray(circuit())
    return probs.reshape(N, 2).sum(axis=1)
