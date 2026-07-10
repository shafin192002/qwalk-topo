"""
The Z2 Klein-bottle topological invariant.

Non-orientability changes the classification: an ordinary (orientable) Brillouin
*torus* carries an integer Chern number, but a non-orientable Brillouin *Klein
bottle* carries only a Z2 invariant. The Klein bottle appears when a Z2 gauge
field -- hopping phases +-1, the same "-1" twist this package uses for the Mobius
seam (see shift.mobius_shift) -- makes momentum space acquire a glide reflection

    U H(kx, ky) U^dagger = H(-kx, ky + pi).                    (glide symmetry)

Under this glide the Brillouin zone folds into a Klein bottle, and (because the
glide is orientation-reversing on the BZ, forcing the Chern number to vanish) the
invariant is the Z2

    nu = W_pi  mod 2,

where W_pi is the number of times the kx-Wilson-loop Berry phase gamma(ky) crosses
pi as ky sweeps a half-period (ky: -pi -> 0). This is the construction of

    Z. Y. Chen, S. A. Yang, Y. X. Zhao,
    "Brillouin Klein bottle from artificial gauge fields",
    Nature Communications 13, 2215 (2022); arXiv:2204.12438.

This module ports their invariant to reusable code and ships their 4-band model
as the correctness anchor. The implementation is *verified* against their paper:
``klein_bottle_invariant`` returns 1 on their non-trivial parameter set and 0 on
their trivial set (their Figs. 3-4), the value is resolution-stable, it jumps only
across gap closings, and an open-boundary (x-edge) strip of the non-trivial model
hosts in-gap edge states while the trivial model does not (bulk-boundary
correspondence). See tests/test_klein.py and examples/klein_invariant.py.

Note: this reproduces established (indeed experimentally realised) physics; it is
a validated reference implementation, not a novel result. The novelty of this
package remains the 1D discrete-time Mobius result (winding.py).
"""
from __future__ import annotations
import numpy as np

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)

# Glide operator U with U H(kx,ky) U^dagger = H(-kx, ky+pi):  U = tau_0 (x) sigma_1.
GLIDE_U = np.kron(_I2, _SX)

# Chen-Yang-Zhao parameter sets (their Figs. 3-4). Known answers: nu = 1 and 0.
CYZ_NONTRIVIAL = dict(t11x=1.0, t22x=1.0, t12x=3.5, t21x=3.5,
                      t1y=2.0, t2y=1.5, eps=1.0, lam=1.0)
CYZ_TRIVIAL = dict(t11x=1.0, t12x=1.0, t21x=3.5, t22x=1.7,
                   t1y=2.0, t2y=1.5, eps=0.6, lam=0.0)


def cyz_hamiltonian(kx: float, ky: float, *, t11x, t22x, t12x, t21x,
                    t1y, t2y, eps, lam) -> np.ndarray:
    """4x4 Bloch Hamiltonian of Chen, Yang & Zhao, Nat. Commun. 13, 2215 (2022).

    Eq. (10): H0 with q_a^x(kx) = t_{a1}^x + t_{a2}^x e^{ikx} (a = 1, 2) and
    q_pm^y(ky) = t_1^y +- t_2^y e^{iky}, in the basis (tau (x) sigma) with diagonal
    (eps, eps, -eps, -eps); plus the time-reversal-breaking term
    H1 = lam cos(ky) (tau_1 (x) sigma_2) + lam sin(ky) (tau_2 (x) sigma_2).

    The result satisfies the glide symmetry GLIDE_U @ H @ GLIDE_U^dagger =
    H(-kx, ky+pi).
    """
    q1 = t11x + t12x * np.exp(1j * kx)
    q2 = t21x + t22x * np.exp(1j * kx)
    qp = t1y + t2y * np.exp(1j * ky)
    qm = t1y - t2y * np.exp(1j * ky)
    H0 = np.array([
        [eps,          np.conj(q1),  np.conj(qp),  0.0        ],
        [q1,           eps,          0.0,          np.conj(qm)],
        [qp,           0.0,         -eps,          np.conj(q2)],
        [0.0,          qm,           q2,          -eps        ]], dtype=complex)
    H1 = (lam * np.cos(ky) * np.kron(_SX, _SY)
          + lam * np.sin(ky) * np.kron(_SY, _SY))
    return H0 + H1


def _expm_herm(H: np.ndarray, s: float) -> np.ndarray:
    """exp(-i s H) for Hermitian H, via eigendecomposition (NumPy only, no SciPy)."""
    w, v = np.linalg.eigh(H)
    return (v * np.exp(-1j * s * w)) @ v.conj().T


def cyz_pieces(kx: float, ky: float, **params) -> tuple[np.ndarray, np.ndarray]:
    """Split the CYZ Hamiltonian into two glide-covariant pieces (H_x, H_y).

    Each piece separately satisfies GLIDE_U H GLIDE_U† = H(-kx, ky+pi), so any
    product of their exponentials inherits the glide symmetry.
    """
    H = cyz_hamiltonian(kx, ky, **params)
    eps, lam = params["eps"], params["lam"]
    q1 = params["t11x"] + params["t12x"] * np.exp(1j * kx)
    q2 = params["t21x"] + params["t22x"] * np.exp(1j * kx)
    Hx = np.array([[eps, np.conj(q1), 0, 0], [q1, eps, 0, 0],
                   [0, 0, -eps, np.conj(q2)], [0, 0, q2, -eps]], dtype=complex)
    Hy = H - Hx                       # the remaining (y-hopping + lam) terms
    return Hx, Hy


def cyz_floquet_walk(kx: float, ky: float, s: float = 0.4,
                     **params) -> np.ndarray:
    """A 2D split-step *Floquet* walk with the Klein glide, built from CYZ pieces.

    U_F(kx,ky) = exp(-i s H_x) exp(-i s H_y). Because H_x and H_y are each
    glide-covariant, U_F satisfies GLIDE_U U_F GLIDE_U† = U_F(-kx, ky+pi), and
    since [H_x, H_y] != 0 it is a genuine two-step walk (its quasi-energy bands
    differ from the static spectrum). As s -> 0 it reduces to the static CYZ
    model, so its Klein invariant recovers the published nu there.
    """
    Hx, Hy = cyz_pieces(kx, ky, **params)
    return _expm_herm(Hx, s) @ _expm_herm(Hy, s)


def cyz_floquet_effective_hamiltonian(kx: float, ky: float, s: float = 0.4,
                                      **params) -> np.ndarray:
    """Effective Hamiltonian H_eff = i log U_F (quasi-energies in (-pi, pi]).

    Hermitian; feed it to klein_bottle_invariant to get the Z2 invariant of the
    quasi-energy gap at eps = 0 of the Floquet Klein walk.

    RELIABLE ONLY AT SMALL DRIVE. Quasi-energy is periodic (mod 2pi), so H_eff has
    a branch cut at eps = pi. As s grows the bands are pushed against that branch
    cut (the eps = pi gap shrinks); the "bands below eps = 0" selection used by the
    invariant then reorders discontinuously as a band wraps across +-pi, and the
    invariant flips *without any real gap closing* -- a numerical artifact, not a
    Floquet transition (the eps = 0 gap stays open and the edge modes persist). A
    proper strong-drive invariant is the Rudner-Lindner-Berg-Levin per-gap Floquet
    winding number (Phys. Rev. X 3, 031005 (2013)), which uses the full-period
    micromotion and is branch-cut-free; that is left as future work. Use this only
    in the anchored small-drive regime, and cross-check gap closings / edge modes.
    """
    U = cyz_floquet_walk(kx, ky, s, **params)
    w, v = np.linalg.eig(U)
    H = (v * (-np.angle(w))) @ np.linalg.inv(v)
    return 0.5 * (H + H.conj().T)


def _occupied_frame(Hk: np.ndarray, n_occ: int) -> np.ndarray:
    """Columns = the n_occ lowest-energy eigenvectors of a Hermitian matrix."""
    _, vecs = np.linalg.eigh(Hk)
    return vecs[:, :n_occ]


def berry_phase_kx_loop(H, ky: float, n_occ: int, n_kx: int = 300,
                        **params) -> float:
    """Wilson-loop Berry phase of the occupied bands around the kx circle.

    gamma(ky) = arg det [ prod_i <U(kx_i)|U(kx_{i+1})> ] in (-pi, pi], with U the
    frame of the n_occ lowest bands. Uses the multiband overlap determinant, so it
    is gauge invariant and robust for degenerate bands.
    """
    ks = np.linspace(-np.pi, np.pi, n_kx, endpoint=False)
    frames = [_occupied_frame(H(k, ky, **params), n_occ) for k in ks]
    M = np.eye(n_occ, dtype=complex)
    for i in range(n_kx):
        M = M @ (frames[i].conj().T @ frames[(i + 1) % n_kx])
    return float(np.angle(np.linalg.det(M)))


def klein_bottle_invariant(H, n_occ: int, n_kx: int = 300, n_ky: int = 400,
                           **params) -> int:
    """Z2 Klein-bottle invariant nu in {0, 1} (Chen, Yang & Zhao, 2022).

    nu = (number of times the kx-Wilson-loop Berry phase gamma(ky) crosses pi as
    ky runs from -pi to 0) mod 2. Requires the glide symmetry
    U H(kx, ky) U^dagger = H(-kx, ky+pi).

    Parameters
    ----------
    H : callable H(kx, ky, **params) -> Hermitian (2n_occ)x(2n_occ) ndarray.
    n_occ : number of occupied (lower) bands (half filling: n_occ = dim/2).
    n_kx, n_ky : Wilson-loop / sweep resolution.
    **params : forwarded to H (e.g. the CYZ hopping parameters).

    Verified on cyz_hamiltonian: returns 1 for CYZ_NONTRIVIAL, 0 for CYZ_TRIVIAL.
    """
    kys = np.linspace(-np.pi, 0.0, n_ky)
    gamma = np.unwrap([berry_phase_kx_loop(H, ky, n_occ, n_kx, **params)
                       for ky in kys])
    crossings = 0
    for a, b in zip(gamma[:-1], gamma[1:]):
        lo, hi = sorted((a, b))
        crossings += int(np.floor((hi - np.pi) / (2 * np.pi))
                         - np.floor((lo - np.pi) / (2 * np.pi)))
    return crossings % 2
