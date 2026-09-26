"""
Physics data for the qwalk-topo animations.

The point of this module: **no physics lives in a Manim scene file.** The scenes
import arrays from here and do nothing but draw them. That keeps the animation
code about animation, and it means the figures and the films are driven by the
same verified package rather than by a re-implementation.

Everything here is cheap -- the whole set computes in about 0.02 s -- so nothing
is cached to disk. If a future scene needs a heavy sweep (a full phase-diagram
grid, say), cache *that* rather than adding a cache layer for these.

Run ``python animations/_data.py`` to print a summary and sanity-check the
numbers without rendering anything.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Allow running as a plain script and being imported by a Manim scene file,
# without installing the package -- same pattern as examples/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwalktopo import split_step_walk                      # noqa: E402
from qwalktopo.invariants.winding import _bloch_su2         # noqa: E402

# Shared coin angles: the topological point used throughout the repo's figures.
THETA1, THETA2 = 0.6, 0.35
THETA1_TRIVIAL = 0.1            # |theta1| < theta2  ->  nu = 0


# ----------------------------------------------------------------- scene 1
def seam_walk(N: int = 16, steps: int = 24, theta1: float = THETA1,
              theta2: float = THETA2) -> dict:
    """Ring and Mobius walks from the same localised start.

    Returns a dict with, for each topology, the per-step site density
    ``P[step, x]`` (shape ``(steps+1, N)``), plus the step at which the two
    first differ and the per-step total-variation distance between them.

    The walker starts at the centre with the balanced coin, so it takes a few
    steps to reach the seam. Until it does, the ring and Mobius walks are
    *bit-identical* -- the twist sits on one bond and nothing has touched it
    yet. The step where they part company is the seam crossing, and it is the
    beat the animation is built around.
    """
    x0 = N // 2
    dens, amps = {}, {}
    for topo in ("ring", "mobius"):
        U = split_step_walk(N, theta1, theta2, topo)
        psi = np.zeros(2 * N, dtype=complex)
        psi[2 * x0:2 * x0 + 2] = np.array([1.0, 1j]) / np.sqrt(2.0)
        frames = [psi.copy()]
        for _ in range(steps):
            psi = U @ psi
            frames.append(psi.copy())
        a = np.array(frames).reshape(steps + 1, N, 2)
        amps[topo] = a
        dens[topo] = (np.abs(a) ** 2).sum(axis=2)

    tvd = 0.5 * np.abs(dens["ring"] - dens["mobius"]).sum(axis=1)
    adiff = np.abs(amps["ring"] - amps["mobius"]).sum(axis=(1, 2))
    hit = lambda a: int(np.argmax(a > 1e-9)) if np.any(a > 1e-9) else -1

    # These differ by one step, and the gap is the interesting part: the seam
    # stamps the -1 onto the amplitude at `first_amplitude`, but a sign on one
    # component is not yet visible in |psi|^2 -- it only shows up once it
    # interferes, at `first_density`. The sign is there a step before you can
    # measure it.
    return dict(N=N, steps=steps, x0=x0, density=dens, amplitude=amps,
                tvd=tvd, amp_diff=adiff,
                first_amplitude=hit(adiff), first_density=hit(tvd),
                theta1=theta1, theta2=theta2)


def seam_signed(N: int = 16, steps: int = 20, coin: int = 0,
                theta1: float = THETA1, theta2: float = THETA2) -> dict:
    """Signed amplitudes for the seam scene, plus the exact flip event.

    Returns ``re[topo][step, x]`` -- the real part of the amplitude on coin
    component ``coin`` (0 = right-mover, the one that crosses the seam) -- for
    the ring and Mobius walks, together with ``flip_step`` and ``flip_site``
    locating the single component where they first differ.

    Why the real part and not the density: the seam multiplies an amplitude by
    -1, and |psi|^2 cannot see that, because |-a|^2 = |+a|^2. A density plot of
    this walk is mathematically incapable of showing the effect it is meant to
    illustrate -- at the moment of crossing the two densities are *identical*.
    The sign only becomes visible in a signed quantity.

    The event is remarkably clean: at ``flip_step`` exactly one of the 2N
    complex components differs between the two walks, and it differs by exactly
    a factor -1. Every other number is bit-identical. One step later that lone
    sign has interfered with its neighbours and the two walks are genuinely
    different everywhere.
    """
    w = seam_walk(N=N, steps=steps, theta1=theta1, theta2=theta2)
    a = w["amplitude"]
    re = {t: a[t][:, :, coin].real.copy() for t in ("ring", "mobius")}

    t0 = w["first_amplitude"]
    d = np.abs(a["mobius"][t0] - a["ring"][t0])
    site, comp = (int(v) for v in np.argwhere(d > 1e-10)[0])
    return dict(re=re, N=N, steps=steps, coin=coin,
                flip_step=t0, flip_site=site, flip_coin=comp,
                ring_value=float(a["ring"][t0, site, comp].real),
                mobius_value=float(a["mobius"][t0, site, comp].real),
                n_components=int(a["ring"][t0].size),
                n_differing=int(np.count_nonzero(d > 1e-10)),
                ymax=float(max(np.abs(re["ring"]).max(),
                               np.abs(re["mobius"]).max())),
                tvd=w["tvd"], first_density=w["first_density"])


# ----------------------------------------------------------------- scene 2
MOBIUS_R = 1.0
MOBIUS_W = 0.34


def mobius_point(u: float, v: float, R: float = MOBIUS_R) -> np.ndarray:
    """A point on the Mobius band. Same parametrisation as examples/mobius_render.py."""
    return np.array([(R + v * np.cos(u / 2.0)) * np.cos(u),
                     (R + v * np.cos(u / 2.0)) * np.sin(u),
                     v * np.sin(u / 2.0)])


def mobius_width_dir(u: float) -> np.ndarray:
    """Unit vector along the band's width at parameter ``u`` (that is, dP/dv).

    This is the frame that the animation transports. It is already unit length
    for every ``u``, and after one circuit it comes back negated:

        w(u = 2 pi) = -w(u = 0)

    That sign is the whole modelling assumption of the package made visible --
    a frame carried once around the orientation-reversing loop returns
    inverted, which is why the seam contributes the Z2 holonomy -1 rather
    than +1. See DOCUMENTATION.md section 2.1.
    """
    return np.array([np.cos(u / 2.0) * np.cos(u),
                     np.cos(u / 2.0) * np.sin(u),
                     np.sin(u / 2.0)])


def mobius_normal(u: float) -> np.ndarray:
    """Unit surface normal of the band along its centre line (v = 0).

    Obtained as (dP/du) x (dP/dv) at v = 0 and simplified; it is unit length for
    every u, and perpendicular to both tangents (checked numerically below).

    Scene 2 needs this because the frame it transports is the *width* direction,
    which is tangent to the surface. Drawn on the centre line the arrow is
    therefore coplanar with the band, so the band renders over it and the marker
    reads as a spear stuck through the strip. Lifting it a little way along the
    normal puts it on top of the surface, which is what "a frame carried along
    the band" should look like.
    """
    return np.array([np.sin(u / 2.0) * np.cos(u),
                     np.sin(u / 2.0) * np.sin(u),
                     -np.cos(u / 2.0)])


# ----------------------------------------------------------------- scene 4
def closure_condition(N: int = 8, k_max: float = np.pi,
                      n_k: int = 900) -> dict:
    """Which waves can live on the ring, and which on the Mobius band.

    A wave has to meet itself after one lap. Writing it as cos(k x) and carrying
    it from x = 0 round to x = N, the two gluings demand

        ring    :  psi(x + N) = +psi(x)   ->  cos(k N) = +1
        mobius  :  psi(x + N) = -psi(x)   ->  cos(k N) = -1

    so a wave is allowed exactly where its end value lands on the target. Those
    conditions pick out

        ring    :  k = 2 pi m / N
        mobius  :  k = 2 pi (m + 1/2) / N

    which is the half-spacing shift the whole package is about -- here derived
    from nothing but "the wave has to close up".

    The k = 0 case is the one worth watching. A flat wave is the same everywhere,
    so coming back negated would need c = -c, i.e. c = 0. The flat wave does not
    merely shift on the Mobius band; it cannot exist at all. Since that is the
    momentum where the split-step gap closes, removing it is what leaves the
    Mobius walk gapped where the ring is not.

    Returns the sweep ``ks``, the end value ``end`` = cos(k N) it produces, and
    the exact allowed momenta for each topology below ``k_max``.
    """
    ks = np.linspace(0.0, k_max, n_k)
    end = np.cos(ks * N)

    m = np.arange(0, int(np.ceil(k_max * N / (2 * np.pi))) + 2)
    ring = 2 * np.pi * m / N
    mobius = 2 * np.pi * (m + 0.5) / N
    tol = 1e-9
    return dict(N=N, ks=ks, end=end, k_max=k_max,
                ring_allowed=ring[ring <= k_max + tol],
                mobius_allowed=mobius[mobius <= k_max + tol],
                spacing=2 * np.pi / N)


# ----------------------------------------------------------------- scene 3
def winding_loop(theta1: float = THETA1, theta2: float = THETA2,
                 n_k: int = 400) -> dict:
    """The chiral Bloch vector's path around the Brillouin zone.

    Returns ``ks``, the unit Bloch vector components ``ny``, ``nz``, the
    unwrapped polar angle ``angle``, and the running winding
    ``turns = (angle - angle[0]) / 2 pi`` whose final value is the invariant.

    Note for the animation: ``n_hat`` is a *unit* vector pinned to the
    (n_y, n_z) plane by chiral symmetry, so it rides the unit circle in **both**
    phases -- the loop's shape and its distance from the origin are the same
    either way. What separates the phases is purely how far around it gets:
    a full lap for nu = -1, out-and-back for nu = 0. So the honest thing to put
    on screen is the accumulated angle, not the loop's outline.
    """
    ks = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    _, m = _bloch_su2(theta1, theta2, ks)
    n = m / np.linalg.norm(m, axis=1)[:, None]
    ny, nz = n[:, 1], n[:, 2]
    angle = np.unwrap(np.arctan2(nz, ny))
    return dict(ks=ks, ny=ny, nz=nz, angle=angle,
                turns=(angle - angle[0]) / (2.0 * np.pi),
                nu=int(round((angle[-1] - angle[0]) / (2.0 * np.pi))),
                nx_max=float(np.max(np.abs(n[:, 0]))),
                theta1=theta1, theta2=theta2)


# ----------------------------------------------------------------- self-check
def _summary() -> None:
    w = seam_walk()
    print("scene 1  seam walk")
    print(f"   density shape        : {w['density']['ring'].shape}  (steps+1, N)")
    print(f"   amplitudes differ at : step {w['first_amplitude']}  <- seam stamps the -1")
    print(f"   density differs at   : step {w['first_density']}  <- when it becomes visible")
    print(f"   final ring<->Mobius  : {w['tvd'][-1]:.4f}")
    print(f"   probability conserved: "
          f"{np.allclose(w['density']['mobius'].sum(axis=1), 1.0)}")

    print("scene 2  Mobius frame transport")
    w0, w1 = mobius_width_dir(0.0), mobius_width_dir(2 * np.pi)
    print(f"   w(0)                 : {np.round(w0, 6)}")
    print(f"   w(2 pi)              : {np.round(w1, 6)}")
    print(f"   returns negated      : {np.allclose(w1, -w0)}  <- the -1 holonomy")
    nn = [mobius_normal(u) for u in np.linspace(0, 2 * np.pi, 9)]
    print(f"   normal is unit       : "
          f"{np.allclose([np.linalg.norm(x) for x in nn], 1.0)}")
    print(f"   normal _|_ width dir : "
          f"{np.allclose([x @ mobius_width_dir(u) for x, u in zip(nn, np.linspace(0, 2 * np.pi, 9))], 0.0)}")
    print(f"   closes as a Mobius   : "
          f"{np.allclose(mobius_point(0.0, 0.3), mobius_point(2 * np.pi, -0.3))}")

    c = closure_condition()
    print("scene 4  which waves close up")
    print(f"   ring allowed k/pi    : {np.round(c['ring_allowed'] / np.pi, 4)}")
    print(f"   mobius allowed k/pi  : {np.round(c['mobius_allowed'] / np.pi, 4)}")
    print(f"   offset is half a step: "
          f"{np.allclose(c['mobius_allowed'][0] - c['ring_allowed'][0], c['spacing'] / 2)}")
    print(f"   ring includes k = 0  : {np.any(np.abs(c['ring_allowed']) < 1e-12)}")
    print(f"   mobius includes k = 0: {np.any(np.abs(c['mobius_allowed']) < 1e-12)}")

    print("scene 3  winding loop")
    for lbl, t1 in (("topological", THETA1), ("trivial", THETA1_TRIVIAL)):
        d = winding_loop(t1, THETA2)
        print(f"   {lbl:11s} theta1={t1}: nu = {d['nu']:+d}, "
              f"total turns = {d['turns'][-1]:+.3f}, max|n_x| = {d['nx_max']:.1e}")


if __name__ == "__main__":
    _summary()
