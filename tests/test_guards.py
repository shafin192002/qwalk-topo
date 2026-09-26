"""Tests for the correctness guards, not the physics.

These cover the failure modes that are silent rather than loud -- the ones that
hand back a plausible-looking wrong number instead of raising. Each test here
corresponds to a bug that a passing physics suite did not catch.
"""
import numpy as np
import pytest

from qwalktopo import (split_step_walk, mobius_shift, ring_shift, klein_shift,
                       winding_number, klein_bottle_invariant, check_topology,
                       check_glide, cyz_hamiltonian, symmetric_walk,
                       CYZ_NONTRIVIAL)
from qwalktopo.walk import _half_shift, coin


# ---------- a mistyped topology must never fall back to the ring ----------
@pytest.mark.parametrize("bad", ["Mobius", "MOBIUS", "torus", "klein", "ring ", ""])
def test_unknown_topology_raises(bad):
    # the whole package is a ring-vs-Mobius comparison, so a silent fallback to
    # the ring would return a plausible, wrong answer.
    with pytest.raises(ValueError, match="unknown topology"):
        split_step_walk(8, 0.6, 0.35, bad)
    with pytest.raises(ValueError, match="unknown topology"):
        _half_shift(8, bad, +1)


@pytest.mark.parametrize("good", ["ring", "mobius"])
def test_known_topologies_accepted(good):
    assert check_topology(good) == good
    assert split_step_walk(8, 0.6, 0.35, good).shape == (16, 16)


def test_ring_and_mobius_really_differ():
    a = split_step_walk(8, 0.6, 0.35, "ring")
    b = split_step_walk(8, 0.6, 0.35, "mobius")
    assert not np.allclose(a, b)


# ---------- the seam twist has exactly one implementation ----------
@pytest.mark.parametrize("N", [4, 8, 13, 16])
def test_half_shifts_compose_to_mobius_shift(N):
    # walk._half_shift and shift.mobius_shift apply the same twist via the same
    # helper; composing the half-shifts must reproduce the full shift exactly.
    Sp = _half_shift(N, "mobius", +1)
    Sm = _half_shift(N, "mobius", -1)
    assert np.allclose(Sm @ Sp, mobius_shift(N))


@pytest.mark.parametrize("N", [4, 8, 13, 16])
def test_mobius_twist_confined_to_two_seam_bonds(N):
    diff = np.abs(mobius_shift(N) - ring_shift(N))
    assert np.count_nonzero(diff > 1e-12) == 2


# ---------- guards survive `python -O` (no bare asserts in the package) ----------
def test_package_has_no_assert_based_validation():
    import pathlib
    import qwalktopo
    root = pathlib.Path(qwalktopo.__file__).parent
    offenders = []
    for f in root.rglob("*.py"):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("assert "):
                offenders.append(f"{f.name}:{n}")
    # `python -O` strips assert statements; correctness guards must not be one.
    assert offenders == [], f"assert-based validation found: {offenders}"


def test_unitarity_guard_raises_not_asserts():
    from qwalktopo.shift import _require_unitary
    with pytest.raises(RuntimeError, match="not unitary"):
        _require_unitary(np.array([[2.0, 0.0], [0.0, 1.0]]), "test matrix")


def test_coin_length_mismatch_raises_valueerror():
    with pytest.raises(ValueError, match="scalar or a length-8 array"):
        coin(np.zeros(5), 8)


# ---------- the winding number is flagged, not silently returned ----------
@pytest.mark.parametrize("t1,t2", [(0.3, 0.3), (-0.3, 0.3), (0.0, 0.0),
                                   (np.pi / 2, np.pi / 2), (0.5, -0.5)])
def test_winding_raises_on_gap_closing_line(t1, t2):
    # the raw sweep still yields a clean-looking integer here, which is exactly
    # why the gap has to be checked rather than trusted.
    with pytest.raises(ValueError, match="undefined"):
        winding_number(t1, t2)


@pytest.mark.parametrize("t1,t2,expect", [(0.5, 0.3, -1), (0.9, 0.3, -1),
                                          (0.1, 0.3, 0), (-0.8, 0.3, 1),
                                          (0.6, 0.35, -1)])
def test_winding_unchanged_away_from_gap_closing(t1, t2, expect):
    assert winding_number(t1, t2) == expect


def test_winding_vectorised_matches_reference_bloch_vector():
    # the fast batched sweep must agree with the per-k eigendecomposition path
    from qwalktopo.invariants.winding import _bloch_vector, _bloch_su2
    ks = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    _, m = _bloch_su2(0.6, 0.35, ks)
    for i, k in enumerate(ks):
        _, nhat = _bloch_vector(0.6, 0.35, k)
        mi = m[i] / np.linalg.norm(m[i])
        assert np.allclose(mi, nhat, atol=1e-9)


# ---------- the Klein Z2 requires its glide precondition ----------
def test_klein_invariant_rejects_non_glide_hamiltonian():
    def not_glide(kx, ky):
        rng = np.random.default_rng(int(abs(kx * 1000) + abs(ky * 97)))
        A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        return A + A.conj().T
    with pytest.raises(ValueError, match="glide symmetry"):
        klein_bottle_invariant(not_glide, n_occ=2, n_kx=40, n_ky=50,
                               verify_glide=True)


def test_check_glide_accepts_cyz():
    check_glide(cyz_hamiltonian, **CYZ_NONTRIVIAL)          # must not raise


# ---------- klein_shift is a genuine Klein gluing, not merely unitary ----------
@pytest.mark.parametrize("Nx,Ny", [(4, 3), (4, 4), (6, 5)])
def test_klein_shift_y_movers_are_mutual_inverses(Nx, Ny):
    # unitarity alone does not say the +y and -y sectors glue consistently;
    # a genuine Klein identification makes them exact inverses of each other.
    S = klein_shift(Nx, Ny)
    dimc, dim = 4, 4 * Nx * Ny
    idx = lambda x, y, c: (x * Ny + y) * dimc + c
    Py = np.zeros((Nx * Ny, Nx * Ny))
    Qy = np.zeros((Nx * Ny, Nx * Ny))
    for x in range(Nx):
        for y in range(Ny):
            src = x * Ny + y
            Py[np.argmax(np.abs(S[:, idx(x, y, 2)])) // dimc, src] = 1.0
            Qy[np.argmax(np.abs(S[:, idx(x, y, 3)])) // dimc, src] = 1.0
    assert np.allclose(Py @ Qy, np.eye(Nx * Ny))
    assert np.allclose(Qy @ Py, np.eye(Nx * Ny))


@pytest.mark.parametrize("Nx,Ny", [(4, 3), (4, 4), (6, 5)])
def test_klein_shift_y_translation_has_order_2Ny(Nx, Ny):
    # going around the y-cycle once reflects x; going around twice must return
    # every site to itself -- the defining property of the Klein gluing.
    S = klein_shift(Nx, Ny)
    dimc = 4
    idx = lambda x, y, c: (x * Ny + y) * dimc + c
    Py = np.zeros((Nx * Ny, Nx * Ny))
    for x in range(Nx):
        for y in range(Ny):
            Py[np.argmax(np.abs(S[:, idx(x, y, 2)])) // dimc, x * Ny + y] = 1.0
    once = np.linalg.matrix_power(Py, Ny)
    assert not np.allclose(once, np.eye(Nx * Ny))        # reflected, not identity
    assert np.allclose(once @ once, np.eye(Nx * Ny))     # twice = identity


# ---------- quasi-energy range is [-pi, pi) as documented ----------
def test_quasi_energy_range_is_half_open():
    from qwalktopo import quasi_energies
    eps = quasi_energies(split_step_walk(32, 0.6, 0.35, "ring"))
    assert eps.min() >= -np.pi - 1e-12
    assert eps.max() < np.pi


def test_symmetric_walk_rejects_bad_topology():
    with pytest.raises(ValueError, match="unknown topology"):
        symmetric_walk(8, 0.6, 0.35, "Mobius")



# ---------- the docs must not drift from the code they describe ----------
def _repo_root():
    import pathlib
    import qwalktopo
    return pathlib.Path(qwalktopo.__file__).parent.parent


def test_docs_state_the_real_test_count():
    """The stated test count must match the collected one.

    It is typed by hand into two files and nothing checked it, so it drifted
    twice (44 -> 85 -> 86) before a person happened to notice. Prose
    consistency needs a reader; a number does not.
    """
    import re
    import subprocess
    import sys
    root = _repo_root()
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         cwd=root, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    if m is None:                      # older/quieter pytest prints a bare total
        m = re.search(r"^(\d+)$", out.strip().splitlines()[-1])
    assert m, "could not determine the collected test count"
    real = m.group(1)

    for doc in ("README.md", "DOCUMENTATION.md"):
        text = (root / doc).read_text(encoding="utf-8")
        stated = re.findall(r"(\d+) tests", text)
        assert stated, f"{doc} no longer states a test count"
        for v in stated:
            assert v == real, (
                f"{doc} says {v} tests but {real} are collected - "
                "update the doc, or drop the number so it cannot go stale")
