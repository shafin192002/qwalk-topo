"""
Scene 1: the Mobius seam, and the single sign that changes.

    manim -pql animations/seam_flip.py SeamFlip        # draft
    manim -qm --format=gif animations/seam_flip.py SeamFlipShort   # README GIF
    python animations/seam_flip.py                     # same as the draft line

Two split-step walks, ring and Mobius, from the same localised start. Every bond
is identical except one: the wrap-around bond, which the Mobius seam multiplies
by -1.

What is on screen is the *signed* amplitude, not the density. That choice is the
whole scene. The seam's effect is a factor -1, and |psi|^2 cannot see a sign,
because |-a|^2 = |+a|^2 -- at the moment of crossing the two densities are
exactly equal. An earlier version of this scene plotted density, and was
therefore showing a quantity mathematically incapable of revealing the effect it
was meant to illustrate.

Shown as a signed quantity the event is startlingly sharp: for eight steps all
32 complex components agree to the last bit; at step 8 exactly *one* of them
changes sign; one step later that lone sign has interfered outward and the two
walks differ everywhere.

Uses Text (Pango) only -- no LaTeX required.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from manim import (Axes, Create, Dot, FadeIn, FadeOut, Flash, Line, Rectangle,
                   Scene, Text, VGroup, ValueTracker, always_redraw, config,
                   DOWN, RIGHT, UP)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mstyle as st                                       # noqa: E402
from _data import seam_signed                              # noqa: E402

config.background_color = st.BG


class SeamFlip(Scene):
    STEP_TIME = 0.40          # seconds per walk step
    HOLD = 1.0                # beat length for the callouts

    def construct(self):
        d = seam_signed(N=16, steps=20)
        N, re = d["N"], d["re"]
        kf, xf = d["flip_step"], d["flip_site"]
        ymax = d["ymax"] * 1.12

        axes = Axes(x_range=[-0.8, N - 0.2, 1], y_range=[-ymax, ymax, ymax / 2],
                    x_length=10.2, y_length=4.0,
                    axis_config=dict(stroke_color=st.BAND, stroke_width=2,
                                     include_ticks=False),
                    y_axis_config=dict(stroke_opacity=0.0),
                    tips=False).shift(DOWN * 0.40)

        head = Text("One bond differs. Watch what it costs.",
                    font_size=29, color=st.INK).to_edge(UP, buff=0.22)
        ylab = Text("amplitude of the right-mover   (signed, not |ψ|²)",
                    font_size=20, color=st.BAND)
        ylab.next_to(axes, DOWN, buff=0.48)

        step = ValueTracker(0.0)

        def value(topo, x):
            s = float(np.clip(step.get_value(), 0, d["steps"]))
            lo = int(np.floor(s))
            hi = min(lo + 1, d["steps"])
            f = s - lo
            return (1 - f) * re[topo][lo, x] + f * re[topo][hi, x]

        def stems(topo, colour, dx):
            """Signed stem plot, rebuilt each frame from the walk data."""
            def build():
                g = VGroup()
                for x in range(N):
                    v = value(topo, x)
                    p0 = axes.c2p(x + dx, 0.0)
                    p1 = axes.c2p(x + dx, v)
                    g.add(Line(p0, p1, stroke_color=colour, stroke_width=5.5))
                    g.add(Dot(p1, radius=0.048, color=colour))
                return g
            return always_redraw(build)

        # Offset the two sets slightly in x so they read as adjacent pairs:
        # identical values give parallel pairs, a flipped sign gives an
        # opposing pair, which is the one thing the eye should catch.
        ring = stems("ring", st.RING, -0.16)
        mob = stems("mobius", st.MOBIUS, +0.16)

        legend = VGroup(
            Text("ring  (periodic)", font_size=22, color=st.RING),
            Text("Möbius  (anti-periodic)", font_size=22, color=st.MOBIUS),
        ).arrange(RIGHT, buff=0.7).next_to(head, DOWN, buff=0.20)

        # the wrap-around bond: one bond, drawn at both ends of the unrolled line
        seam_l = Line(axes.c2p(-0.62, -ymax * 0.92),
                      axes.c2p(-0.62, ymax * 0.92),
                      stroke_color=st.ACCENT, stroke_width=3).set_opacity(0.5)
        seam_r = Line(axes.c2p(N - 0.38, -ymax * 0.92),
                      axes.c2p(N - 0.38, ymax * 0.92),
                      stroke_color=st.ACCENT, stroke_width=3).set_opacity(0.5)
        seam_lbl = Text("the seam: one bond, joining x = N−1 back to x = 0",
                        font_size=19, color=st.ACCENT)
        seam_lbl.next_to(axes, UP, buff=0.08)

        site_lo = Text("x = 0", font_size=19, color=st.BAND)
        site_lo.next_to(axes.c2p(0, -ymax), DOWN, buff=0.10)
        site_hi = Text("x = " + str(N - 1), font_size=19, color=st.BAND)
        site_hi.next_to(axes.c2p(N - 1, -ymax), DOWN, buff=0.10)

        counter = always_redraw(lambda: Text(
            "step " + str(int(round(step.get_value()))),
            font_size=26, color=st.INK).to_corner(DOWN + RIGHT, buff=0.40))

        # ---------------------------------------------------------- build up
        self.play(FadeIn(head, shift=DOWN * 0.15), run_time=0.8)
        self.play(Create(axes), FadeIn(ylab), run_time=0.9)
        self.play(FadeIn(legend), FadeIn(seam_lbl), FadeIn(site_lo),
                  FadeIn(site_hi), Create(seam_l), Create(seam_r),
                  run_time=0.8)
        self.add(ring, mob, counter)
        self.wait(0.6)

        # ------------------------------------------- phase 1: nothing differs
        note = Text("all 32 amplitudes agree, exactly",
                    font_size=23, color=st.BAND)
        note.next_to(legend, DOWN, buff=0.20)
        self.play(FadeIn(note), run_time=0.5)
        self.play(step.animate.set_value(kf - 1),
                  run_time=self.STEP_TIME * (kf - 1), rate_func=lambda t: t)
        self.wait(0.4)

        # ------------------------------------------- phase 2: the single flip
        self.play(step.animate.set_value(kf), run_time=self.STEP_TIME,
                  rate_func=lambda t: t)

        box = Rectangle(width=1.05, height=axes.y_length * 0.30,
                        stroke_color=st.EDGE, stroke_width=3.5)
        box.move_to(axes.c2p(xf, 0.0))
        self.play(Create(box),
                  Flash(axes.c2p(xf, 0), color=st.EDGE, line_length=0.2,
                        num_lines=14, flash_radius=0.5), run_time=0.8)

        hit = Text("one amplitude has crossed the seam", font_size=24,
                   color=st.EDGE)
        hit.next_to(legend, DOWN, buff=0.20)
        self.play(FadeOut(note), FadeIn(hit), run_time=0.5)

        nums = VGroup(
            Text("ring      " + st.signed(d["ring_value"]),
                 font_size=25, color=st.RING),
            Text("Möbius  " + st.signed(d["mobius_value"]),
                 font_size=25, color=st.MOBIUS),
            Text("1 of 32 numbers.  Same size, opposite sign.",
                 font_size=20, color=st.INK),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.13)
        nums.next_to(box, RIGHT, buff=0.7).shift(UP * 1.15)
        self.play(FadeIn(nums), run_time=0.7)
        self.wait(self.HOLD * 1.6)

        # ------------------------------- phase 3: one sign infects everything
        self.play(FadeOut(nums), FadeOut(box), run_time=0.5)
        spread = Text("that single sign now interferes outward",
                      font_size=24, color=st.MOBIUS)
        spread.next_to(legend, DOWN, buff=0.20)
        self.play(FadeOut(hit), FadeIn(spread), run_time=0.5)

        self.play(step.animate.set_value(d["steps"]),
                  run_time=self.STEP_TIME * (d["steps"] - kf),
                  rate_func=lambda t: t)

        done = Text("two different walks, from one flipped sign",
                    font_size=25, color=st.INK)
        done.next_to(legend, DOWN, buff=0.20)
        self.play(FadeOut(spread), FadeIn(done), run_time=0.6)
        self.wait(self.HOLD * 1.4)


class SeamFlipShort(SeamFlip):
    """Tighter cut for a README GIF."""
    STEP_TIME = 0.22
    HOLD = 0.6


if __name__ == "__main__":
    # Running `python seam_flip.py` directly would otherwise just define the
    # class and exit without rendering anything -- Manim renders through its own
    # CLI, not through the module. This forwards to it so the obvious command
    # works. For more control, call the CLI yourself:
    #     manim -pqh animations/seam_flip.py SeamFlip
    import subprocess
    import sys as _sys
    _sys.exit(subprocess.call([_sys.executable, "-m", "manim", "-pql",
                               __file__, "SeamFlip"]))
