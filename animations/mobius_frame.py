"""
Scene 2: why the seam twist is -1.

    manim -pql animations/mobius_frame.py MobiusFrame
    manim -qm --format=gif animations/mobius_frame.py MobiusFrame

The package's central modelling assumption is that a spin-1/2 coin carried once
around the Mobius band's orientation-reversing loop comes back with a factor
-1 (DOCUMENTATION.md section 2.1). In the code that is one negated matrix
element. Here it is the obvious thing it actually is: a frame carried around the
band returns inverted.

The transported vector is the band's own width direction, dP/dv, which is unit
length for every u and satisfies

    w(u = 2 pi) = -w(u = 0)

exactly -- verified numerically in _data.py. A ghost of the starting frame stays
on screen so the return orientation can be compared against it directly.

Uses Text (Pango) only; no LaTeX needed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from manim import (Arrow3D, Create, Dot3D, FadeIn, Surface, Text, VGroup,
                   ThreeDScene, ValueTracker, always_redraw, config,
                   DEGREES, DOWN, LEFT, UP, TAU)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mstyle as st                                       # noqa: E402
from _data import (MOBIUS_W, mobius_point, mobius_normal,   # noqa: E402
                   mobius_width_dir)

config.background_color = st.BG

SCALE = 2.15         # world units per unit of the parametrisation
# The arrow spans the band rather than overshooting it: at 1.05 it was longer
# than the strip is wide (2 * MOBIUS_W = 0.68), so it stuck out both edges and,
# being coplanar with the surface, got sliced by it -- reading as a spear driven
# through the strip instead of a marker carried along it.
ARROW_LEN = 0.58     # slightly inside the band's full width
ARROW_THICK = 0.05   # Arrow3D defaults are far too thin to see in 3D
ARROW_LIFT = 0.095   # float above the surface along its normal, so it sits ON it
LIFT_EASE = 3.5      # how sharply the lift eases through zero (see _seat)
GHOST_U = -0.34      # offset so the start and return frames sit side by side


def _P(u: float, v: float) -> np.ndarray:
    return mobius_point(u, v) * SCALE


class MobiusFrame(ThreeDScene):
    TRANSPORT_TIME = 9.0

    def construct(self):
        self.set_camera_orientation(phi=58 * DEGREES, theta=-52 * DEGREES,
                                    zoom=1.25)

        # Manim lights a 3D surface by angle, so the faces turning toward the
        # light wash out. With a pale checkerboard at low opacity on a white
        # background that went far enough to dissolve the band's edge exactly
        # where it twists. Giving the surface more body and a visible outline
        # keeps the silhouette readable all the way round, while staying light
        # enough that the arrows on top remain the brightest thing on screen.
        band = Surface(lambda u, v: _P(u, v),
                       u_range=[0, TAU], v_range=[-MOBIUS_W, MOBIUS_W],
                       resolution=(48, 8),
                       fill_opacity=0.92, stroke_width=0.8,
                       stroke_color="#8d9a9f",
                       checkerboard_colors=["#aab6bb", "#c2ccd0"])

        head = Text("Carry a frame once around the band", font_size=30,
                    color=st.INK).to_edge(UP, buff=0.36)
        self.add_fixed_in_frame_mobjects(head)
        head.set_opacity(0)

        self.play(Create(band), run_time=2.2)
        self.play(head.animate.set_opacity(1.0), run_time=0.7)

        u = ValueTracker(0.0)

        def _cam_dir():
            """Unit vector from the origin toward the camera."""
            phi, th = self.camera.get_phi(), self.camera.get_theta()
            return np.array([np.sin(phi) * np.cos(th),
                             np.sin(phi) * np.sin(th),
                             np.cos(phi)])

        def _seat(uu):
            """Arrow endpoints for the frame at u, lifted clear of the surface.

            The lift follows the surface normal, but which way along it is not a
            fixed choice: the band is non-orientable, so there is no consistent
            "outer" side to sit on -- the normal that points up at one place
            points down half a turn later. Picking a fixed sign buries the arrow
            under the band for half the loop, so the direction has to follow the
            camera.

            Doing that with a hard ``if n . camera < 0: n = -n`` is wrong,
            though: the sign flips in a single frame and the arrow jumps clean
            across the strip -- a quarter of the band's width, instantly. It
            reads as a teleport. Easing the lift through zero with ``tanh``
            instead keeps it continuous. Where it passes through zero the
            marker is momentarily flush with the surface, but that is exactly
            the moment the surface is edge-on to the camera, so there is
            nothing to see there anyway.
            """
            n = mobius_normal(uu)
            ease = np.tanh(LIFT_EASE * float(n @ _cam_dir()))
            base = _P(uu, 0.0) + n * (ARROW_LIFT * ease) * SCALE
            half = mobius_width_dir(uu) * ARROW_LEN * 0.5 * SCALE
            return base - half, base + half

        def frame_arrow():
            a, b = _seat(u.get_value())
            return Arrow3D(start=a, end=b, color=st.MOBIUS,
                           thickness=ARROW_THICK, base_radius=0.115)

        def walker_dot():
            a, b = _seat(u.get_value())
            return Dot3D((a + b) * 0.5, radius=0.062, color=st.ACCENT)

        # A stationary ghost of the starting frame, to compare against. It is
        # placed a short way *back* along the band (GHOST_U) rather than exactly
        # at u = 0: the returning frame lands on u = 2 pi, which is the same
        # physical point, so drawn together the two arrows overlap into a single
        # ambiguous shape. Offsetting by a few degrees puts them side by side,
        # where pointing opposite ways is unmistakable.
        # Same weight as the transported arrow: these two are the comparison the
        # scene exists to make, so neither should look like the junior partner.
        def ghost_arrow():
            ga, gb = _seat(GHOST_U)
            g = Arrow3D(start=ga, end=gb, color=st.RING,
                        thickness=ARROW_THICK, base_radius=0.115)
            g.set_opacity(0.9)
            return g

        ghost = always_redraw(ghost_arrow)

        arrow = always_redraw(frame_arrow)
        dot = always_redraw(walker_dot)

        self.add(ghost)
        self.wait(0.3)
        self.add(arrow, dot)
        self.wait(0.6)

        legend = VGroup(
            Text("blue = starting frame (fixed)", font_size=21, color=st.RING),
            Text("red  = transported frame", font_size=21, color=st.MOBIUS),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        self.add_fixed_in_frame_mobjects(legend)
        # left edge, low: clear of the title above and the closing line below
        legend.to_edge(LEFT, buff=0.40).shift(DOWN * 1.75)
        self.play(FadeIn(legend), run_time=0.5)

        self.begin_ambient_camera_rotation(rate=0.10)
        self.play(u.animate.set_value(TAU), run_time=self.TRANSPORT_TIME,
                  rate_func=lambda x: x)
        self.stop_ambient_camera_rotation()

        # the transported frame now sits on top of the ghost, pointing the
        # other way -- move in so the opposition is unmistakable
        self.move_camera(phi=54 * DEGREES, theta=-16 * DEGREES, zoom=1.55,
                         frame_center=_P(0.0, 0.0) * 0.72, run_time=2.0)
        self.wait(0.6)
        verdict = Text("...and it comes back inverted.   That is the  −1.",
                       font_size=30, color=st.MOBIUS)
        self.add_fixed_in_frame_mobjects(verdict)
        verdict.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(verdict), run_time=0.8)
        self.wait(2.0)


class MobiusFrameShort(MobiusFrame):
    """Tighter cut for a README GIF."""
    TRANSPORT_TIME = 5.5

if __name__ == "__main__":
    # Running `python mobius_frame.py` directly would otherwise just define the
    # class and exit without rendering anything -- Manim renders through its own
    # CLI, not through the module. This forwards to it so the obvious command
    # works. For more control, call the CLI yourself:
    #     manim -pqh animations/mobius_frame.py MobiusFrame
    import subprocess
    import sys as _sys
    _sys.exit(subprocess.call([_sys.executable, "-m", "manim", "-pql",
                               __file__, "MobiusFrame"]))
