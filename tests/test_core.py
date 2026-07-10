"""Known-answer tests for qwalk-topo.

Run with:  python -m pytest tests/  (or python tests/test_core.py)

The split-step winding diagram is taken from Kitagawa, Rudner, Berg, Demler,
PRA 82, 033429 (2010) and Asboth & Obuse, PRB 88, 121406(R) (2013):
the (theta1, theta2) plane is divided by the gap-closing lines
theta1 = +/- theta2 (mod pi) into regions of constant integer winding.
"""
import numpy as np
import pytest

from qwalktopo import (ring_shift, mobius_shift, klein_shift,
                       split_step_walk, quasi_energies, winding_number)
from qwalktopo.invariants.winding import _bloch_vector


def _periodic_momenta(N):
    return np.array([2 * np.pi * m / N for m in range(N)])


def _antiperiodic_momenta(N):
    return np.array([2 * np.pi * (m + 0.5) / N for m in range(N)])


def _band_spectrum(t1, t2, ks):
    """Full analytic spectrum {+-E(k)} sampled on the momenta ks, sorted."""
    Ek = np.array([_bloch_vector(t1, t2, k)[0] for k in ks])
    return np.sort(np.concatenate([Ek, -Ek]))


# ---------- unitarity ----------
@pytest.mark.parametrize("N", [5, 8, 13, 21])
def test_ring_shift_unitary(N):
    S = ring_shift(N)
    assert np.allclose(S @ S.conj().T, np.eye(2 * N))


@pytest.mark.parametrize("N", [5, 8, 13, 21])
def test_mobius_shift_unitary(N):
    S = mobius_shift(N)
    assert np.allclose(S @ S.conj().T, np.eye(2 * N))


@pytest.mark.parametrize("Nx,Ny", [(3, 3), (4, 3), (4, 4)])
def test_klein_shift_unitary(Nx, Ny):
    S = klein_shift(Nx, Ny)
    assert np.allclose(S @ S.conj().T, np.eye(4 * Nx * Ny))


@pytest.mark.parametrize("topo", ["ring", "mobius"])
def test_floquet_unitary(topo):
    U = split_step_walk(16, 0.7, 0.4, topo)
    assert np.allclose(U @ U.conj().T, np.eye(32), atol=1e-10)


# ---------- known-answer: Mobius differs from ring ----------
def test_mobius_distinct_from_ring():
    e_ring = quasi_energies(split_step_walk(12, 0.8, 0.5, "ring"))
    e_mob = quasi_energies(split_step_walk(12, 0.8, 0.5, "mobius"))
    assert np.max(np.abs(e_ring - e_mob)) > 1e-3


# ---------- flagship: exact spectra sit on the (anti-)periodic bands ----------
def test_ring_spectrum_matches_periodic_bands():
    # the ring walk samples the analytic bands on the periodic momenta 2*pi*m/N
    t1, t2, N = 0.6, 0.35, 16
    eps = np.sort(quasi_energies(split_step_walk(N, t1, t2, "ring")))
    pred = _band_spectrum(t1, t2, _periodic_momenta(N))
    assert np.max(np.abs(eps - pred)) < 1e-8


def test_mobius_spectrum_matches_antiperiodic_bands():
    # the flagship result: the Mobius walk samples the SAME bands but on the
    # half-integer (anti-periodic) momenta 2*pi*(m+1/2)/N -- never k = 0.
    t1, t2, N = 0.6, 0.35, 16
    eps = np.sort(quasi_energies(split_step_walk(N, t1, t2, "mobius")))
    pred = _band_spectrum(t1, t2, _antiperiodic_momenta(N))
    assert np.max(np.abs(eps - pred)) < 1e-8


# ---------- finite-size gap: Mobius keeps a gap the ring loses ----------
def test_ring_gap_closes_mobius_open_on_line():
    # on theta1 = -theta2 the ring's k=0 mode makes gap -> 0; Mobius avoids k=0.
    t2, N = 0.35, 24
    g_ring = np.min(np.abs(quasi_energies(split_step_walk(N, -t2, t2, "ring"))))
    g_mob = np.min(np.abs(quasi_energies(split_step_walk(N, -t2, t2, "mobius"))))
    assert g_ring < 1e-9          # ring gap closes at k = 0
    assert g_mob > 1e-2           # Mobius stays gapped at finite N


# ---------- position-dependent coin + bulk-boundary edge states ----------
def test_position_dependent_coin_unitary():
    N = 10
    theta = np.linspace(-1.0, 1.0, N)          # a different angle on every site
    from qwalktopo.walk import coin
    C = coin(theta, N)
    assert C.shape == (2 * N, 2 * N)
    assert np.allclose(C @ C.conj().T, np.eye(2 * N))


def test_domain_wall_binds_pi_edge_modes():
    # a topological/trivial domain wall on a ring binds two protected pi-modes,
    # each localised at the two walls (bulk-boundary correspondence).
    N, theta2 = 48, 0.30
    th1 = np.where(np.arange(N) < N // 2, 1.2, 0.0)   # topological | trivial
    U = split_step_walk(N, th1, theta2, "ring")
    w, v = np.linalg.eig(U)
    eps = -np.angle(w)
    walls = list(range(0, 3)) + list(range(N - 2, N)) + \
        list(range(N // 2 - 2, N // 2 + 3))
    n_edge = 0
    for i in range(2 * N):
        if np.abs(np.abs(eps[i]) - np.pi) < 1e-2:           # pinned at eps = pi
            dens = (np.abs(v[:, i]) ** 2).reshape(N, 2).sum(axis=1)
            if dens[walls].sum() > 0.5:                      # localised at walls
                n_edge += 1
    assert n_edge == 2


# ---------- known-answer: winding quantisation & plateaus ----------
def test_winding_integer_quantised():
    # within a region the winding is a constant integer
    for t1 in [0.5, 0.7, 0.9, 1.1]:
        assert winding_number(t1, 0.3, n_k=2000) == -1


def test_winding_jumps_across_gap_closing():
    # crossing the line theta1 = theta2 (=0.3) changes the winding by 1
    below = winding_number(0.1, 0.3, n_k=2000)   # |t1| < t2  -> trivial (0)
    above = winding_number(0.6, 0.3, n_k=2000)   # t1 > t2    -> nu = -1
    assert below == 0
    assert above == -1
    assert below != above


def test_winding_parity_symmetry():
    # theta1 -> -theta1 flips the sign of the winding
    assert winding_number(0.8, 0.3) == -winding_number(-0.8, 0.3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
