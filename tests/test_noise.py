"""Tests for the decoherence module.

Anchors: the density-matrix walk must (i) stay a valid state (trace 1, CPTP),
(ii) reproduce the pure walk at p=0 with the mean chiral displacement converging
to the winding number, and (iii) show the physical hierarchy -- symmetry-respecting
(bit-flip) noise protects the topological MCD while symmetry-breaking (dephasing,
depolarizing) noise degrades it.
"""
import numpy as np
import pytest

from qwalktopo import (run_noisy_walk, mean_chiral_displacement, coin_channel,
                       symmetric_walk, winding_number, topology_distinguishability)


def test_symmetric_walk_is_chiral_and_unitary():
    N = 21
    U = symmetric_walk(N, 0.6, 0.35, "ring")
    assert np.allclose(U @ U.conj().T, np.eye(2 * N), atol=1e-10)
    G = np.kron(np.eye(N), np.array([[0, 1], [1, 0]], dtype=complex))
    # chiral symmetry: Gamma U Gamma = U^{-1}
    assert np.allclose(G @ U @ G, U.conj().T, atol=1e-10)


@pytest.mark.parametrize("channel", ["dephasing", "depolarizing", "bitflip"])
def test_channels_are_cptp(channel):
    # sum_j K_j^dagger K_j = I  (trace preservation / CPTP)
    for p in (0.0, 0.25, 0.7):
        Ks = coin_channel(channel, p)
        S = sum(K.conj().T @ K for K in Ks)
        assert np.allclose(S, np.eye(2))


def test_density_matrix_stays_valid():
    out = run_noisy_walk(31, 0.6, 0.35, steps=12, channel="depolarizing", p=0.3)
    assert np.allclose(out["trace"], 1.0, atol=1e-9)          # trace preserved
    assert np.all(out["purity"] <= 1.0 + 1e-9)               # purity in (0,1]
    assert np.all(out["purity"] > 0)


def test_mcd_converges_to_winding_number_no_noise():
    # at p = 0 the mean chiral displacement recovers the winding number
    for t1, expect in [(0.6, -1), (-0.6, 1), (0.1, 0)]:
        C = mean_chiral_displacement(51, t1, 0.35, steps=24)
        assert abs(C - expect) < 0.15


def test_symmetry_respecting_noise_protects_topology():
    # bit-flip (sigma_x = Gamma) keeps MCD near nu; dephasing/depol degrade it.
    N, t1, t2, T, p = 51, 0.6, 0.35, 24, 0.3
    nu = winding_number(t1, t2)
    c_clean = mean_chiral_displacement(N, t1, t2, T, "none", 0.0)
    c_bit = mean_chiral_displacement(N, t1, t2, T, "bitflip", p)
    c_dep = mean_chiral_displacement(N, t1, t2, T, "depolarizing", p)
    assert abs(c_clean - nu) < 0.15
    assert abs(c_bit - nu) < 0.15                     # protected
    assert abs(c_dep - nu) > abs(c_bit - nu) + 0.1    # degraded, and worse than bit-flip


def test_non_orientability_decoheres_away():
    # ring vs Mobius are distinguishable at p=0 but the difference washes out
    N, t1, t2, T = 16, 0.6, 0.35, 22
    d0 = topology_distinguishability(N, t1, t2, T, "dephasing", 0.0)
    dmid = topology_distinguishability(N, t1, t2, T, "dephasing", 0.1)
    dhi = topology_distinguishability(N, t1, t2, T, "dephasing", 0.4)
    assert d0 > 0.1            # non-orientability is visible with no noise
    assert dmid < d0           # ... and monotonically fades
    assert dhi < 0.02          # ... to nothing (no robustness)
