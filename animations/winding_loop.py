"""
Scene 3: the winding number, as an angle that accumulates.

    manim -pql animations/winding_loop.py WindingLoop
    manim -qm --format=gif animations/winding_loop.py WindingLoop

Chiral symmetry pins the effective Hamiltonian's Bloch vector to the (n_y, n_z)
plane, so n_hat is a unit vector riding the unit circle. As k sweeps the
Brillouin zone it drags that vector around, and the winding number is simply

    nu = (total angle swept) / 2 pi.

One thing worth being careful about, because it is easy to animate misleadingly:
n_hat rides the *same* unit circle in both phases, and the loop's distance from
the origin is identical either way. So "the loop encircles the origin" is not
something a viewer can see here. What actually separates the phases is how far
around the vector gets -- a full lap for nu = -1, out-and-back for nu = 0 --
which is why the accumulated angle is the thing on screen.

Uses Text (Pango) only, so no LaTeX is required -- note this includes passing
``mob_class=Text`` to DecimalNumber, which otherwise renders through MathTex and
would silently reintroduce a LaTeX dependency.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from manim import (Arc, Arrow, Axes, Circle, Create, Dot, FadeIn,
                   FadeOut, Scene, Text, TracedPath, VGroup, ValueTracker,
                   always_redraw, config, DOWN, LEFT, RIGHT, UP)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mstyle as st                                       # noqa: E402
from _data import THETA1, THETA1_TRIVIAL, THETA2, winding_loop   # noqa: E402

config.background_color = st.BG


class WindingLoop(Scene):
    SWEEP_TIME = 5.0

    def construct(self):
        topo = winding_loop(THETA1, THETA2)
        triv = winding_loop(THETA1_TRIVIAL, THETA2)

        axes = Axes(x_range=[-1.45, 1.45, 0.5], y_range=[-1.45, 1.45, 0.5],
                    x_length=5.0, y_length=5.0,
                    axis_config=dict(stroke_color=st.BAND, stroke_width=2,
                                     include_ticks=False),
                    tips=False).shift(LEFT * 2.5)
        guide = Circle(radius=1.0, stroke_color=st.BAND, stroke_width=1.6,
                       stroke_opacity=0.45)
        guide.move_to(axes.c2p(0, 0))
        guide.set(width=2 * abs(axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0]))

        xlab = Text("n_y", font_size=24, color=st.BAND).next_to(axes.x_axis,
                                                               RIGHT, buff=0.16)
        ylab = Text("n_z", font_size=24, color=st.BAND).next_to(axes.y_axis,
                                                               UP, buff=0.16)
        head = Text("The winding number is an angle you accumulate",
                    font_size=28, color=st.INK).to_edge(UP, buff=0.34)

        self.play(FadeIn(head, shift=DOWN * 0.2), run_time=0.8)
        self.play(Create(axes), FadeIn(xlab), FadeIn(ylab), Create(guide),
                  run_time=1.1)

        readout = RIGHT * 3.75          # explicit point, not an empty VGroup
        last = "trivial"
        for phase, data, colour in (("topological", topo, st.MOBIUS),
                                    ("trivial", triv, st.RING)):
            # keep the final panel on screen -- a looping GIF that ends on a
            # blank frame reads as broken.
            self._sweep(axes, data, colour, phase, readout,
                        clear=(phase != last))

        self.wait(1.0)

    # ------------------------------------------------------------------
    def _sweep(self, axes, data, colour, phase, anchor_pt, clear=True):
        """Sweep k once, tracing n_hat and reading off the accumulated turns."""
        ks, ny, nz = data["ks"], data["ny"], data["nz"]
        turns = data["turns"]
        n = len(ks)
        t = ValueTracker(0.0)

        def nhat_point():
            s = float(np.clip(t.get_value(), 0, n - 1))
            lo = int(np.floor(s)); hi = min(lo + 1, n - 1); f = s - lo
            y = (1 - f) * ny[lo] + f * ny[hi]
            z = (1 - f) * nz[lo] + f * nz[hi]
            return axes.c2p(y, z)

        vec = always_redraw(lambda: Arrow(
            axes.c2p(0, 0), nhat_point(), buff=0,
            stroke_width=5, color=colour, max_tip_length_to_length_ratio=0.16))
        tip = always_redraw(lambda: Dot(nhat_point(), radius=0.06, color=colour))
        trail = TracedPath(nhat_point, stroke_color=colour, stroke_width=4)

        title = Text(f"{phase}", font_size=27, color=colour)
        sub = Text(f"θ₁ = {data['theta1']},  θ₂ = {data['theta2']}",
                   font_size=20, color=st.BAND)
        head_blk = VGroup(title, sub).arrange(DOWN, buff=0.13)
        head_blk.move_to(anchor_pt + UP * 2.05)

        turn_lbl = Text("turns", font_size=22, color=st.INK)

        def turns_now():
            return float(np.interp(np.clip(t.get_value(), 0, n - 1),
                                   np.arange(n), turns))

        turn_val = always_redraw(lambda: Text(
            st.signed(turns_now(), 2), font_size=44, color=colour
        ).move_to(anchor_pt + DOWN * 0.18))
        def k_now():
            s = float(np.clip(t.get_value(), 0, n - 1))
            return float(np.interp(s, np.arange(n), ks))

        kval = always_redraw(lambda: Text(
            "k = " + st.signed(k_now() / np.pi, 2) + " π",
            font_size=21, color=st.BAND).move_to(anchor_pt + DOWN * 0.72))

        turn_lbl.move_to(anchor_pt + UP * 0.42)
        block = VGroup(turn_lbl)
        self.add(kval, turn_val)

        arc = always_redraw(lambda: Arc(
            radius=abs(axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0]) * 0.34,
            start_angle=float(np.arctan2(nz[0], ny[0])),
            angle=float(np.interp(np.clip(t.get_value(), 0, n - 1),
                                  np.arange(n), data["angle"]))
            - float(np.arctan2(nz[0], ny[0])),
            arc_center=axes.c2p(0, 0), stroke_color=colour,
            stroke_width=7, stroke_opacity=0.5))

        self.play(FadeIn(head_blk), FadeIn(block), run_time=0.6)
        self.add(arc)
        self.add(trail, vec, tip)
        self.play(t.animate.set_value(n - 1), run_time=self.SWEEP_TIME,
                  rate_func=lambda x: x)

        nu = data["nu"]
        verdict = Text("ν = 0" if nu == 0 else f"ν = " + st.signed(nu, 0),
                       font_size=42, color=colour)
        verdict.move_to(anchor_pt + DOWN * 1.60)
        self.play(FadeIn(verdict), run_time=0.6)
        self.wait(1.1)

        if clear:
            kval.clear_updaters()
            arc.clear_updaters()
            turn_val.clear_updaters()
            self.play(FadeOut(trail), FadeOut(vec), FadeOut(tip), FadeOut(arc),
                      FadeOut(head_blk), FadeOut(block), FadeOut(kval),
                      FadeOut(turn_val), FadeOut(verdict), run_time=0.6)

class WindingLoopShort(WindingLoop):
    """Tighter cut for a README GIF."""
    SWEEP_TIME = 3.2


if __name__ == "__main__":
    # Running `python winding_loop.py` directly would otherwise just define the
    # class and exit without rendering anything -- Manim renders through its own
    # CLI, not through the module. This forwards to it so the obvious command
    # works. For more control, call the CLI yourself:
    #     manim -pqh animations/winding_loop.py WindingLoop
    import subprocess
    import sys as _sys
    _sys.exit(subprocess.call([_sys.executable, "-m", "manim", "-pql",
                               __file__, "WindingLoop"]))
