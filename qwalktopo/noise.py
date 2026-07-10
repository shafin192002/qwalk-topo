"""
Decoherence for the split-step walk: does the topology survive noise?

The winding number of §invariants is a *pure-state* quantity; under decoherence
the walk is a mixed state ρ, so we track topology through an observable that
survives mixing: the **mean chiral displacement** (MCD)

    C(t) = -2 Tr[ ρ(t) Γ (X - x0) ] ,        Γ = σ_x (chiral operator),

which converges to the winding number ν in the long-time limit for the chiral
walk in its symmetric time frame (Cardano et al., Nat. Commun. 8, 15516 (2017);
Maffei et al., New J. Phys. 20, 013023 (2018)). Here C → ν to a few 1e-3 at p=0.

Noise is applied as a completely-positive channel on the coin, once per step:

    ρ -> Σ_j (I ⊗ K_j)  U ρ U†  (I ⊗ K_j)† ,

with U the symmetric-frame Floquet operator and {K_j} the coin Kraus operators.
Three channels are provided:

    'dephasing'    : {√(1-p) I, √p σ_z}           -- breaks chiral symmetry
    'depolarizing' : {√(1-p) I, √(p/3) σ_{x,y,z}} -- breaks everything
    'bitflip'      : {√(1-p) I, √p σ_x}           -- σ_x = Γ, chiral-symmetric

The point is not "how much noise" but **which symmetry the noise respects**:
symmetry-respecting noise (bit-flip, along the chiral axis) leaves the MCD -- and
hence the topological invariant -- essentially intact even at large p, while
symmetry-breaking noise (dephasing, depolarizing) drives C -> 0 and turns the
ballistic quantum spreading (Var ~ t²) into diffusive classical spreading
(Var ~ t). See examples/noise_robustness.py.

Dependency-free (NumPy only): ρ is (2N)x(2N), cheap for the sizes used here.
"""
from __future__ import annotations
import numpy as np

from .walk import coin, _half_shift

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def coin_channel(channel: str, p: float) -> list[np.ndarray]:
    """Kraus operators (2x2) of a single-coin noise channel of strength p in [0,1]."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("noise strength p must be in [0, 1]")
    if channel == "none" or p == 0.0:
        return [_I2]
    if channel == "dephasing":
        return [np.sqrt(1 - p) * _I2, np.sqrt(p) * _SZ]
    if channel == "bitflip":
        return [np.sqrt(1 - p) * _I2, np.sqrt(p) * _SX]
    if channel == "depolarizing":
        return [np.sqrt(1 - p) * _I2, np.sqrt(p / 3) * _SX,
                np.sqrt(p / 3) * _SY, np.sqrt(p / 3) * _SZ]
    raise ValueError(f"unknown channel {channel!r}")


def symmetric_walk(N: int, theta1: float, theta2: float,
                   topology: str = "ring") -> np.ndarray:
    """Symmetric-frame split-step Floquet operator (chiral symmetry Γ=σ_x manifest).

    U = C(θ1/2) S₋ C(θ2) S₊ C(θ1/2). In this frame Γ U Γ = U⁻¹ exactly, which is
    what makes the mean chiral displacement converge to the winding number.
    """
    Sp = _half_shift(N, topology, +1)
    Sm = _half_shift(N, topology, -1)
    h = coin(theta1 / 2.0, N)
    U = h @ Sm @ coin(theta2, N) @ Sp @ h
    return U


def _operators(N: int, x0: int):
    G = np.kron(np.eye(N, dtype=complex), _SX)                       # chiral Γ
    disp = (np.arange(N) - x0).astype(float)
    X = np.kron(np.diag(disp).astype(complex), _I2)
    X2 = np.kron(np.diag(disp ** 2).astype(complex), _I2)
    return G, X, X2


def run_noisy_walk(N: int, theta1: float, theta2: float, steps: int,
                   channel: str = "dephasing", p: float = 0.0,
                   topology: str = "ring", x0: int | None = None) -> dict:
    """Evolve the density matrix under the noisy symmetric walk and record observables.

    Starts from a walker localised at ``x0`` (default: the central site) with a
    maximally mixed coin, ρ0 = |x0⟩⟨x0| ⊗ I/2.

    Returns a dict of length-``steps`` arrays:
      't'        : step index 1..steps
      'mcd'      : mean chiral displacement C(t) -> winding number ν (at p=0)
      'variance' : position variance (Var ~ t² ballistic, ~ t diffusive)
      'purity'   : Tr[ρ²] (1 = pure, 1/2N = maximally mixed)
      'trace'    : Tr[ρ] (must stay 1 -- a CPTP check)
    """
    if x0 is None:
        x0 = N // 2
    U = symmetric_walk(N, theta1, theta2, topology)
    G, X, X2 = _operators(N, x0)
    Ks = [np.kron(np.eye(N, dtype=complex), k) for k in coin_channel(channel, p)]

    rho = np.zeros((2 * N, 2 * N), dtype=complex)
    rho[2 * x0, 2 * x0] = 0.5
    rho[2 * x0 + 1, 2 * x0 + 1] = 0.5

    t, mcd, var, pur, tr = [], [], [], [], []
    for step in range(1, steps + 1):
        rho = U @ rho @ U.conj().T
        rho = sum(K @ rho @ K.conj().T for K in Ks)
        mx = np.real(np.trace(rho @ X))
        t.append(step)
        mcd.append(-2.0 * np.real(np.trace(rho @ G @ X)))
        var.append(np.real(np.trace(rho @ X2)) - mx ** 2)
        pur.append(np.real(np.trace(rho @ rho)))
        tr.append(np.real(np.trace(rho)))
    return {"t": np.array(t), "mcd": np.array(mcd), "variance": np.array(var),
            "purity": np.array(pur), "trace": np.array(tr)}


def mean_chiral_displacement(N: int, theta1: float, theta2: float, steps: int,
                             channel: str = "none", p: float = 0.0,
                             topology: str = "ring") -> float:
    """Time-averaged (second-half) mean chiral displacement; -> winding number ν."""
    C = run_noisy_walk(N, theta1, theta2, steps, channel, p, topology)["mcd"]
    return float(np.mean(C[steps // 2:]))


def _final_distribution(N, theta1, theta2, steps, channel, p, topology, x0):
    U = symmetric_walk(N, theta1, theta2, topology)
    Ks = [np.kron(np.eye(N, dtype=complex), k) for k in coin_channel(channel, p)]
    rho = np.zeros((2 * N, 2 * N), dtype=complex)
    rho[2 * x0, 2 * x0] = rho[2 * x0 + 1, 2 * x0 + 1] = 0.5
    for _ in range(steps):
        rho = U @ rho @ U.conj().T
        rho = sum(K @ rho @ K.conj().T for K in Ks)
    return np.real(np.diag(rho)).reshape(N, 2).sum(axis=1)


def topology_distinguishability(N: int, theta1: float, theta2: float, steps: int,
                                channel: str = "dephasing", p: float = 0.0,
                                x0: int | None = None) -> float:
    """Total-variation distance between the ring and Mobius position distributions.

    D = ½ Σ_x |P_ring(x) − P_mobius(x)|, after `steps` noisy steps from the same
    localised start. This measures how distinguishable non-orientability makes the
    walk. Choose N and steps so the walker actually reaches the seam (e.g. N=16,
    steps≈22); otherwise the two are trivially identical.

    As the noise p grows D → 0: the non-orientable signature is a coherent-quantum
    feature that decoherence washes out (non-orientability offers no robustness).
    """
    if x0 is None:
        x0 = N // 2
    pr = _final_distribution(N, theta1, theta2, steps, channel, p, "ring", x0)
    pm = _final_distribution(N, theta1, theta2, steps, channel, p, "mobius", x0)
    return float(0.5 * np.sum(np.abs(pr - pm)))
