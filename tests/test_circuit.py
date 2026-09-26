"""Tests for the quantum-circuit backend (skipped if PennyLane is absent).

The circuit must reproduce the exact-diagonalisation walk: its one-step unitary
equals split_step_walk (up to global phase) and its sampled dynamics match.
"""
import numpy as np
import pytest

pytest.importorskip("pennylane")

from qwalktopo import split_step_walk
from qwalktopo.circuit import walk_unitary, run_walk


def _phase_aligned_diff(U, E):
    ov = np.vdot(U.flatten(), E.flatten())
    return np.max(np.abs(U * (ov / abs(ov)) - E))


@pytest.mark.parametrize("topology", ["ring", "mobius"])
@pytest.mark.parametrize("N", [8, 16])
def test_circuit_unitary_matches_ed(topology, N):
    U = walk_unitary(N, 0.6, 0.35, topology)
    E = split_step_walk(N, 0.6, 0.35, topology)
    assert _phase_aligned_diff(U, E) < 1e-10


@pytest.mark.parametrize("topology", ["ring", "mobius"])
def test_circuit_dynamics_match_ed(topology):
    N, t1, t2, T = 16, 0.6, 0.35, 5
    p_circ = run_walk(N, t1, t2, T, topology=topology)      # exact probabilities
    # ED reference with the same balanced-coin initial state
    U = split_step_walk(N, t1, t2, topology)
    x0 = N // 2
    psi = np.zeros(2 * N, dtype=complex)
    psi[2 * x0:2 * x0 + 2] = np.array([1, 1j]) / np.sqrt(2)
    for _ in range(T):
        psi = U @ psi
    p_ed = (np.abs(psi) ** 2).reshape(N, 2).sum(1)
    assert np.max(np.abs(p_circ - p_ed)) < 1e-9


def test_circuit_requires_power_of_two():
    with pytest.raises(ValueError):
        walk_unitary(12, 0.6, 0.35)


def test_sampled_run_is_reproducible_with_a_seed():
    # an unseeded shot-based run gives a different histogram every time, which
    # silently makes any figure built from it unreproducible.
    a = run_walk(8, 0.6, 0.35, 4, shots=2000, seed=7)
    b = run_walk(8, 0.6, 0.35, 4, shots=2000, seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, run_walk(8, 0.6, 0.35, 4, shots=2000, seed=8))
