"""
Topological invariants for 1D split-step quantum walks.

The split-step walk has chiral symmetry: there exists Gamma with
Gamma U Gamma^{-1} = U^{-1}. In a symmetric time frame the effective
Hamiltonian H(k) (defined by U(k) = exp(-i H(k))) takes the form
H(k) = E(k) [ n(k) . sigma ], with n(k) lying in a great circle of the
Bloch sphere (a consequence of chiral symmetry). In the symmetric time frame
chiral symmetry Gamma = sigma_x pins n(k) to the (n_y, n_z) plane, and the
topological invariant is the winding number of n(k) around the origin as k
traverses the Brillouin zone:

    nu = (1 / 2pi) oint  (n_y dn_z - n_z dn_y) / (n_y^2 + n_z^2).

For the ring (periodic) walk, k is a good quantum number and nu is the
standard 1D winding number (Kitagawa 2010). For the Mobius walk the
anti-periodic boundary condition shifts the allowed momenta to the
half-integer (anti-periodic) set k = 2pi (m + 1/2)/N, which is exactly the
physical signature this package is built to expose.
"""

from __future__ import annotations
import numpy as np

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def bloch_floquet(theta1: float, theta2: float, k: float,
                  frame: str = "symmetric") -> np.ndarray:
    """2x2 Bloch Floquet operator U(k) for the split-step walk.

    The conditional shift is diagonal in momentum space:
        s_+(k) = diag(e^{+ik}, 1)   (shift |0> right)
        s_-(k) = diag(1, e^{-ik})   (shift |1> left).

    frame='asymmetric' :  U = s_- C(theta2) s_+ C(theta1)   (lab frame)
    frame='symmetric'  :  U = C(t1/2) s_+ C(t2) s_- C(t1/2)-type symmetric
        time frame in which chiral symmetry Gamma = sigma_x is manifest and
        the effective Hamiltonian's Bloch vector lies in the (n_y, n_z) plane,
        giving an integer winding number (Asboth & Obuse, PRB 88, 121406 (2013)).
    """
    def C(t):
        return np.array([[np.cos(t), -np.sin(t)],
                         [np.sin(t),  np.cos(t)]], dtype=complex)
    sp = np.array([[np.exp(1j * k), 0], [0, 1]], dtype=complex)
    sm = np.array([[1, 0], [0, np.exp(-1j * k)]], dtype=complex)
    if frame == "asymmetric":
        return sm @ C(theta2) @ sp @ C(theta1)
    # symmetric frame: split the first coin in half around the period
    h1 = C(theta1 / 2.0)
    return h1 @ sm @ C(theta2) @ sp @ h1


def _bloch_vector(theta1: float, theta2: float, k: float):
    """Return (E, n_hat) with U(k) = exp(-i E n_hat . sigma).

    n_hat is the unit Bloch vector of the effective Hamiltonian.
    """
    U = bloch_floquet(theta1, theta2, k)
    # H(k) = i log U ; extract via eig:  U = e^{-iE} P_+ + e^{+iE} P_-
    eig, vecs = np.linalg.eig(U)
    phases = -np.angle(eig)                # quasi-energies +/- E
    E = np.max(np.abs(phases))
    if E < 1e-12:
        return 0.0, np.array([0.0, 0.0, 1.0])
    # Build H = sum eps |v><v|, then n = (1/E) * (coeffs of sigma).
    H = np.zeros((2, 2), dtype=complex)
    for ev, vp in zip(phases, vecs.T):
        vp = vp / np.linalg.norm(vp)
        H += ev * np.outer(vp, vp.conj())
    nx = np.real(np.trace(H @ SX)) / 2.0 / E
    ny = np.real(np.trace(H @ SY)) / 2.0 / E
    nz = np.real(np.trace(H @ SZ)) / 2.0 / E
    n = np.array([nx, ny, nz])
    nrm = np.linalg.norm(n)
    return E, (n / nrm if nrm > 1e-12 else n)


def winding_number(theta1: float, theta2: float, n_k: int = 2000) -> int:
    """Chiral winding number of the split-step walk (symmetric time frame).

    In the symmetric frame the chiral symmetry Gamma = sigma_x forces the
    effective-Hamiltonian Bloch vector into the (n_y, n_z) plane; the
    invariant is the winding of (n_y, n_z) around the origin across the
    Brillouin zone k in [-pi, pi).

    Returns the nearest integer. Near gap-closing lines (theta1 +/- theta2
    = 0, pi) the gap closes and the invariant is ill-defined.
    """
    ks = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    ny = np.empty(n_k + 1)
    nz = np.empty(n_k + 1)
    for i, k in enumerate(ks):
        _, nhat = _bloch_vector(theta1, theta2, k)
        ny[i], nz[i] = nhat[1], nhat[2]
    # Close the loop explicitly by repeating the first point at the end,
    # then unwrap the polar angle over the closed contour. The total change
    # divided by 2pi is the winding number.
    ny[-1], nz[-1] = ny[0], nz[0]
    ang = np.unwrap(np.arctan2(nz, ny))
    nu = (ang[-1] - ang[0]) / (2 * np.pi)
    # VERIFY: invariant should be (near-)integer for a gapped point.
    return int(np.round(nu))
