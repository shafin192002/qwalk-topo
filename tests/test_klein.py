"""Known-answer tests for the Z2 Klein-bottle invariant.

The correctness anchor is the model of Chen, Yang & Zhao, Nat. Commun. 13, 2215
(2022): the invariant must return 1 on their non-trivial parameter set and 0 on
their trivial set, be resolution-stable, and respect the glide symmetry.
"""
import numpy as np
import pytest

from qwalktopo import (klein_bottle_invariant, cyz_hamiltonian,
                       CYZ_NONTRIVIAL, CYZ_TRIVIAL)
from qwalktopo.invariants.klein import (GLIDE_U, cyz_floquet_walk,
                                        cyz_floquet_effective_hamiltonian)


def test_cyz_hamiltonian_hermitian_and_glide():
    U = GLIDE_U
    for P in (CYZ_NONTRIVIAL, CYZ_TRIVIAL):
        for kx in np.linspace(-3, 3, 5):
            for ky in np.linspace(-3, 3, 5):
                H = cyz_hamiltonian(kx, ky, **P)
                assert np.allclose(H, H.conj().T)                 # Hermitian
                # glide: U H(kx,ky) U^dagger = H(-kx, ky+pi)
                lhs = U @ H @ U.conj().T
                rhs = cyz_hamiltonian(-kx, ky + np.pi, **P)
                assert np.allclose(lhs, rhs, atol=1e-10)


def test_klein_invariant_matches_published_values():
    # the whole point: reproduce Chen-Yang-Zhao's known nu = 1 and nu = 0.
    # (150,200) is in the resolution-stable regime; keeps the test fast.
    assert klein_bottle_invariant(cyz_hamiltonian, n_occ=2, n_kx=150, n_ky=200,
                                  **CYZ_NONTRIVIAL) == 1
    assert klein_bottle_invariant(cyz_hamiltonian, n_occ=2, n_kx=150, n_ky=200,
                                  **CYZ_TRIVIAL) == 0


@pytest.mark.parametrize("n_kx,n_ky", [(150, 200), (300, 400)])
def test_klein_invariant_resolution_stable(n_kx, n_ky):
    assert klein_bottle_invariant(cyz_hamiltonian, n_occ=2, n_kx=n_kx,
                                  n_ky=n_ky, **CYZ_NONTRIVIAL) == 1
    assert klein_bottle_invariant(cyz_hamiltonian, n_occ=2, n_kx=n_kx,
                                  n_ky=n_ky, **CYZ_TRIVIAL) == 0


def test_klein_invariant_jumps_only_across_gap_closing():
    # sweeping eps in the non-trivial family, nu stays 1 while gapped near eps~1
    # and has flipped to 0 by eps~2.6 (after the gap dips) -- it is not constant,
    # confirming the invariant tracks a real transition rather than a fixed label.
    below = klein_bottle_invariant(cyz_hamiltonian, n_occ=2, n_kx=150, n_ky=200,
                                   **{**CYZ_NONTRIVIAL, "eps": 0.6})
    above = klein_bottle_invariant(cyz_hamiltonian, n_occ=2, n_kx=150, n_ky=200,
                                   **{**CYZ_NONTRIVIAL, "eps": 2.6})
    assert below == 1
    assert above == 0


# ---------- the package's own 2D split-step Floquet Klein walk ----------
def test_floquet_walk_has_glide_symmetry():
    s = 0.4
    for kx in np.linspace(-2, 2, 4):
        for ky in np.linspace(-2, 2, 4):
            U = cyz_floquet_walk(kx, ky, s, **CYZ_NONTRIVIAL)
            assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10)   # unitary
            lhs = GLIDE_U @ U @ GLIDE_U.conj().T
            rhs = cyz_floquet_walk(-kx, ky + np.pi, s, **CYZ_NONTRIVIAL)
            assert np.allclose(lhs, rhs, atol=1e-9)                     # glide


def test_floquet_walk_invariant_matches_static_limit():
    # at small drive the Floquet Klein walk recovers the published CYZ nu = 1 / 0
    s = 0.3
    nn = klein_bottle_invariant(
        lambda kx, ky: cyz_floquet_effective_hamiltonian(kx, ky, s, **CYZ_NONTRIVIAL),
        n_occ=2, n_kx=120, n_ky=150)
    nt = klein_bottle_invariant(
        lambda kx, ky: cyz_floquet_effective_hamiltonian(kx, ky, s, **CYZ_TRIVIAL),
        n_occ=2, n_kx=120, n_ky=150)
    assert nn == 1
    assert nt == 0
