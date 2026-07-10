"""
Coin operators and the split-step Floquet walk operator.

A single-step DTQW is U = S (I (x) C), where C is the coin rotation.
The split-step walk of Kitagawa et al. (PRA 82, 033429, 2010) uses two
half-shifts and two coins:

    U = S_- C(theta_2) S_+ C(theta_1)

and is the minimal walk hosting a non-trivial topological phase diagram
in 1D (winding number / chiral symmetry class BDI/AIII). We build the
Floquet operator U; its eigenphases (quasi-energies) eps in (-pi, pi]
come from U|psi> = e^{-i eps} |psi>.
"""

from __future__ import annotations
import numpy as np
from .shift import _basis_shift as _ring_basis


def coin(theta, N: int) -> np.ndarray:
    """Coin operator C(theta) = exp(-i theta sigma_y), tensored over N sites.

    C(theta) = [[cos theta, -sin theta],[sin theta, cos theta]] on the coin,
    identity on position.

    `theta` may be a scalar (position-independent coin, the usual split-step
    walk) or a length-N array giving a *position-dependent* coin angle. The
    latter lets you build spatial domain walls between topological phases, whose
    walls bind protected edge states -- see examples/edge_states.py.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    if theta.size == 1:
        theta = np.full(N, theta.item())
    assert theta.size == N, "theta must be a scalar or a length-N array"
    C = np.zeros((2 * N, 2 * N), dtype=complex)
    for x in range(N):
        t = theta[x]
        C[2 * x:2 * x + 2, 2 * x:2 * x + 2] = [[np.cos(t), -np.sin(t)],
                                               [np.sin(t),  np.cos(t)]]
    # VERIFY: each 2x2 block is SO(2) (det = 1), so C is unitary.
    assert np.allclose(C @ C.conj().T, np.eye(2 * N)), "coin not unitary"
    return C


def _half_shift(N: int, topology: str, sign: int) -> np.ndarray:
    """Half conditional shift used in the split-step construction.

    S_+ moves the |0> (right-mover) component by +1 and leaves |1> fixed;
    S_- moves the |1> (left-mover) component by -1 and leaves |0> fixed.
    Each is a unitary permutation. On the Mobius band the anti-periodic
    boundary condition psi(x+N) = -psi(x) is realised by negating the single
    seam bond of each half-shift (see shift.mobius_shift), so each half-shift
    stays unitary on its own and the twist is confined to the seam.
    """
    proj0 = np.kron(np.eye(N), np.array([[1, 0], [0, 0]], dtype=complex))
    proj1 = np.kron(np.eye(N), np.array([[0, 0], [0, 1]], dtype=complex))

    # Bulk ring half-shifts (permutations, hence unitary).
    Pp = _ring_basis(N, +1)   # |x+1><x|
    Pm = _ring_basis(N, -1)   # |x-1><x|

    if topology == "mobius":
        # Anti-periodic (-1) twist on the seam bond only, matching mobius_shift.
        Pp[0, N - 1] *= -1.0   # right-mover crossing the seam  (N-1) -> 0
        Pm[N - 1, 0] *= -1.0   # left-mover  crossing the seam    0 -> (N-1)

    Hp = np.kron(Pp, np.array([[1, 0], [0, 0]], dtype=complex)) + proj1  # move |0>
    Hm = np.kron(Pm, np.array([[0, 0], [0, 1]], dtype=complex)) + proj0  # move |1>

    H = Hp if sign == +1 else Hm
    # VERIFY: each half-shift is unitary on its own.
    assert np.allclose(H @ H.conj().T, np.eye(2 * N), atol=1e-10), \
        "half shift not unitary"
    return H


def split_step_walk(N: int, theta1: float, theta2: float,
                    topology: str = "ring") -> np.ndarray:
    """Floquet operator of the split-step walk on the chosen topology.

    U = S_- C(theta2) S_+ C(theta1).

    Parameters
    ----------
    N : number of position sites.
    theta1, theta2 : coin angles (rad), each a scalar or a length-N array. The
        (theta1, theta2) plane carries the topological phase diagram; winding
        number jumps across the gap-closing lines theta1 +/- theta2 = 0, pi
        (Kitagawa 2010). Passing a position-dependent array builds a spatial
        domain wall between phases, binding edge states (examples/edge_states.py).
    topology : 'ring' (orientable) or 'mobius' (anti-periodic, non-orientable).

    Returns
    -------
    U : (2N, 2N) unitary Floquet operator.
    """
    C1 = coin(theta1, N)
    C2 = coin(theta2, N)
    Sp = _half_shift(N, topology, +1)
    Sm = _half_shift(N, topology, -1)
    U = Sm @ C2 @ Sp @ C1
    # VERIFY: unitarity of the composed Floquet operator.
    assert np.allclose(U @ U.conj().T, np.eye(2 * N), atol=1e-10), \
        "Floquet operator not unitary"
    return U


def quasi_energies(U: np.ndarray) -> np.ndarray:
    """Quasi-energies eps in (-pi, pi] from U eigenvalues e^{-i eps}.

    Sorted ascending. Real by construction (U unitary => |eig|=1).
    """
    eig = np.linalg.eigvals(U)
    # VERIFY: all eigenvalues on the unit circle.
    assert np.allclose(np.abs(eig), 1.0, atol=1e-8), "non-unitary spectrum"
    eps = -np.angle(eig)
    return np.sort(eps)
