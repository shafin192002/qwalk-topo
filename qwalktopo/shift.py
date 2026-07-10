"""
Conditional shift operators for discrete-time quantum walks on
orientable and non-orientable surfaces.

The walk Hilbert space is H = H_pos (x) H_coin, with H_coin = C^2.
The shift S moves the walker conditioned on the coin state:

    S = sum_x  |x+1><x| (x) |0><0|  +  |x-1><x| (x) |1><1|

What distinguishes the topologies is how the position index wraps at the
boundary, and whether wrapping flips the coin (the orientation-reversing
"deck transformation" of a non-orientable surface).

Boundary conditions encoded here
--------------------------------
- ring (periodic):           x = N-1 --(coin 0)--> x = 0,   amplitude unchanged
- mobius (anti-periodic):    x = N-1 --(coin 0)--> x = 0,   amplitude * (-1)
                             i.e. the orientation-reversing gluing of the Mobius
                             band contributes the Z2 spin holonomy -1 to the coin
                             wavefunction: psi(x + N) = -psi(x). This is the
                             anti-periodic boundary condition that shifts the
                             allowed momenta to the half-integer set
                             k = 2*pi*(m + 1/2)/N.
- klein:                     two-dimensional generalisation; one direction
                             periodic, the other glued with an x-reflection.

References for the construction:
  Kitagawa et al., PRA 82, 033429 (2010)  -- split-step DTQW, topology.
  Li & Zhang, J. Phys. A 45, 285301 (2012) -- non-orientable lattices (CTQW).
The discrete-time + non-orientable + Floquet-invariant combination is the
gap this package targets.
"""

from __future__ import annotations
import numpy as np


def _basis_shift(N: int, direction: int) -> np.ndarray:
    """Cyclic position-shift matrix P with P|x> = |x+direction mod N>.

    Implements  sum_x |x+d><x|.  This is the *orientable* (ring) shift.
    """
    P = np.zeros((N, N), dtype=complex)
    for x in range(N):
        P[(x + direction) % N, x] = 1.0
    # VERIFY: P must be a permutation (unitary). In the ring limit
    # P @ P.conj().T == I_N exactly.
    return P


def ring_shift(N: int) -> np.ndarray:
    """Conditional shift on a periodic ring (orientable).

    S = P_+ (x) |0><0| + P_- (x) |1><1|.
    Returns a (2N, 2N) unitary in the basis |x> (x) |c|.
    """
    Pp = _basis_shift(N, +1)
    Pm = _basis_shift(N, -1)
    proj0 = np.array([[1, 0], [0, 0]], dtype=complex)
    proj1 = np.array([[0, 0], [0, 1]], dtype=complex)
    S = np.kron(Pp, proj0) + np.kron(Pm, proj1)
    # VERIFY: unitarity, S S^dagger = I_{2N}
    assert np.allclose(S @ S.conj().T, np.eye(2 * N)), "ring shift not unitary"
    return S


def mobius_shift(N: int) -> np.ndarray:
    """Conditional shift on a Mobius band (non-orientable, anti-periodic).

    The Mobius band is obtained from the ring by an orientation-reversing
    gluing at the seam. A spin-1/2 coin transported once around such a loop
    picks up the Z2 spin holonomy of the reversed frame, a factor -1, so the
    seam enforces the anti-periodic boundary condition

        psi(x + N) = -psi(x)

    -- the discrete-time analogue of the twisted PBC used for non-orientable
    lattices by Li & Zhang, J. Phys. A 45, 285301 (2012). This is exactly the
    condition that shifts the allowed momenta from the periodic set
    k = 2*pi*m/N to the anti-periodic set k = 2*pi*(m + 1/2)/N.

    Construction (important):  the twist must be applied to the SEAM BOND ONLY,
    i.e. the single wrap-around hopping element that connects site N-1 to site 0.
    A tempting shortcut -- multiplying the whole ring shift by a site-local
    operator (sigma_x on the coin of site 0) -- is unitary but WRONG: it also
    twists the interior bond 1 -> 0 that happens to land on site 0, turning the
    clean anti-periodic closure into a localised defect and destroying the
    half-integer momentum quantisation. Here we negate only the two genuine
    seam matrix elements, one per mover, leaving every bulk bond untouched.
    """
    Pp = _basis_shift(N, +1)   # right-mover permutation
    Pm = _basis_shift(N, -1)   # left-mover permutation
    # Anti-periodic (-1) twist on the seam bond only:
    Pp[0, N - 1] *= -1.0       # right-mover crossing the seam  (N-1) -> 0
    Pm[N - 1, 0] *= -1.0       # left-mover  crossing the seam    0 -> (N-1)
    proj0 = np.array([[1, 0], [0, 0]], dtype=complex)
    proj1 = np.array([[0, 0], [0, 1]], dtype=complex)
    S = np.kron(Pp, proj0) + np.kron(Pm, proj1)
    # VERIFY: unitarity. Negating a matrix element of a permutation keeps every
    # column a distinct unit vector, so S stays unitary.
    assert np.allclose(S @ S.conj().T, np.eye(2 * N)), "mobius shift not unitary"
    # VERIFY: the twist touches the seam only -- S_mobius and S_ring agree on
    # every bulk bond and differ (by the sign) on exactly the two seam bonds.
    diff = np.abs(S - ring_shift(N))
    assert np.count_nonzero(diff > 1e-12) == 2, "mobius twist not confined to seam"
    return S


def klein_shift(Nx: int, Ny: int) -> np.ndarray:
    """Conditional shift on a Klein bottle (2D non-orientable).

    Position space is an Nx x Ny grid; coin is 4-dimensional encoding
    (+x, -x, +y, -y). The x-direction is periodic; the y-direction is
    anti-periodic with an x-reflection on the seam (the Klein gluing).

    Coin basis order: 0:+x, 1:-x, 2:+y, 3:-y. Returns (4 Nx Ny, 4 Nx Ny).
    """
    dimc = 4
    Npos = Nx * Ny
    dim = dimc * Npos
    S = np.zeros((dim, dim), dtype=complex)

    def pidx(x: int, y: int) -> int:
        return x * Ny + y

    def idx(x: int, y: int, c: int) -> int:
        return pidx(x, y) * dimc + c

    for x in range(Nx):
        for y in range(Ny):
            # +x / -x : periodic, no flip
            S[idx((x + 1) % Nx, y, 0), idx(x, y, 0)] = 1.0
            S[idx((x - 1) % Nx, y, 1), idx(x, y, 1)] = 1.0

            # +y : crossing top seam (y == Ny-1) reflects x -> Nx-1-x
            if y == Ny - 1:
                xr = (Nx - 1 - x) % Nx
                S[idx(xr, 0, 2), idx(x, y, 2)] = 1.0
            else:
                S[idx(x, y + 1, 2), idx(x, y, 2)] = 1.0

            # -y : crossing bottom seam (y == 0) reflects x -> Nx-1-x
            if y == 0:
                xr = (Nx - 1 - x) % Nx
                S[idx(xr, Ny - 1, 3), idx(x, y, 3)] = 1.0
            else:
                S[idx(x, y - 1, 3), idx(x, y, 3)] = 1.0

    # VERIFY: permutation -> unitary
    assert np.allclose(S @ S.conj().T, np.eye(dim)), "klein shift not unitary"
    return S
