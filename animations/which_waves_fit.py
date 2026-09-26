"""
Scene 4: which waves can live on the band -- and why k = 0 cannot.

    manim -pql animations/which_waves_fit.py WhichWavesFit
    manim -qm --format=gif animations/which_waves_fit.py WhichWavesFitShort
    python animations/which_waves_fit.py

This is the payoff step. Scenes 1 and 2 show that the seam multiplies an
amplitude by -1; this one shows what that costs you.

A wave has to meet itself after one lap round the loop. Carry cos(k x) from
x = 0 to x = N and the two gluings demand

    ring    :  psi(x + N) = +psi(x)   ->  cos(k N) = +1
    mobius  :  psi(x + N) = -psi(x)   ->  cos(k N) = -1

Sweeping k and watching where the end of the wave lands picks out

    ring    :  k = 2 pi m / N
    mobius  :  k = 2 pi (m + 1/2) / N

-- the half-spacing shift, derived from nothing but "the wave has to close up".

The last beat is the one that matters most. A flat wave (k = 0) is the same
everywhere, so returning negated would need c = -c, i.e. c = 0. On the Mobius
band the flat wave does not shift; it cannot exist. That is precisely the
momentum where the split-step gap closes, which is why the Mobius walk stays
gapped where the ring does not.

Uses Text (Pango) only -- no LaTeX required.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from manim import (Axes, Create, Dot, DashedLine, FadeIn, FadeOut, Flash, Line,
                   NumberLine, Scene, Text, VGroup, ValueTracker,
                   always_redraw, config, DOWN, LEFT, UP)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mstyle as st                                       # noqa: E402
from _data import closure_condition                        # noqa: E402

config.background_color = st.BG


class WhichWavesFit(Scene):
    SWEEP_TIME = 7.0
    HOLD = 1.0

    def construct(self):
        d = closure_condition(N=8, k_max=np.pi)
        N = d["N"]

        axes = Axes(x_range=[0, N, 1], y_range=[-1.45, 1.45, 1],
                    x_length=8.6, y_length=2.7,
                    axis_config=dict(stroke_color=st.BAND, stroke_width=2,
                                     include_ticks=False),
                    y_axis_config=dict(stroke_opacity=0.0),
                    tips=False).shift(UP * 1.05 + LEFT * 1.5)

        head = Text("A wave has to meet itself after one lap",
                    font_size=30, color=st.INK).to_edge(UP, buff=0.24)

        x0_lbl = Text("x = 0", font_size=18, color=st.BAND)
        x0_lbl.next_to(axes.c2p(0, -1.45), DOWN, buff=0.10)
        xN_lbl = Text("x = N", font_size=18, color=st.BAND)
        xN_lbl.next_to(axes.c2p(N, -1.45), DOWN, buff=0.10)

        k = ValueTracker(0.0)
        target = ValueTracker(1.0)          # +1 for the ring, -1 for Mobius

        wave = always_redraw(lambda: axes.plot(
            lambda x: np.cos(k.get_value() * x), x_range=[0, N, 0.02],
            stroke_color=(st.RING if target.get_value() > 0 else st.MOBIUS),
            stroke_width=4))

        start_dot = always_redraw(lambda: Dot(
            axes.c2p(0, 1.0), radius=0.075, color=st.ACCENT))
        end_dot = always_redraw(lambda: Dot(
            axes.c2p(N, np.cos(k.get_value() * N)), radius=0.075,
            color=st.ACCENT))

        tgt_line = always_redraw(lambda: DashedLine(
            axes.c2p(0, target.get_value()), axes.c2p(N + 0.35, target.get_value()),
            stroke_color=st.EDGE, stroke_width=2.5,
            dash_length=0.10).set_opacity(0.8))

        def gap_line():
            end = np.cos(k.get_value() * N)
            tgt = target.get_value()
            ok = abs(end - tgt) < 0.02
            return Line(axes.c2p(N, end), axes.c2p(N, tgt),
                        stroke_color=(st.BAND if ok else st.MOBIUS),
                        stroke_width=5)

        gap = always_redraw(gap_line)

        kline = NumberLine(x_range=[0, np.pi, np.pi / 4], length=9.2,
                           stroke_color=st.BAND, stroke_width=2,
                           include_ticks=True, include_numbers=False,
                           tick_size=0.07).shift(DOWN * 2.50)
        klab = Text("allowed wavenumbers k,  from 0 to π", font_size=20,
                    color=st.BAND).next_to(kline, DOWN, buff=0.78)
        self.klab = klab

        self.play(FadeIn(head, shift=DOWN * 0.15), run_time=0.8)
        self.play(Create(axes), FadeIn(x0_lbl), FadeIn(xN_lbl), run_time=0.9)
        self.add(wave, tgt_line, gap, start_dot, end_dot)
        self.play(Create(kline), FadeIn(klab), run_time=0.7)
        self.wait(0.5)

        found = VGroup()
        self.add(found)

        ring_marks = self._sweep(d, "ring", +1.0, st.RING, axes, k, target,
                                 kline, found,
                                 "ring:  the wave must come back as it left")
        mob_marks = self._sweep(d, "mobius", -1.0, st.MOBIUS, axes, k, target,
                                kline, found,
                                "Möbius:  it must come back upside down")

        self.kline_caption_slot = klab.get_center()
        self._compare(d, kline, ring_marks, mob_marks)
        self._flat_wave(d, axes, k, target, head)

    # ------------------------------------------------------------------
    def _sweep(self, d, topo, tgt, colour, axes, k, target, kline, found, caption):
        """Sweep k once and drop a marker at every wave that closes up."""
        allowed = d[topo + "_allowed"]
        cap = Text(caption, font_size=24, color=colour)
        cap.next_to(axes, UP, buff=0.12)

        target.set_value(tgt)
        k.set_value(0.0)
        self.play(FadeIn(cap), run_time=0.5)

        marks = always_redraw(lambda: VGroup(*[
            Dot(kline.n2p(a), radius=0.085, color=colour)
            for a in allowed if a <= k.get_value() + 1e-9]))
        self.add(marks)

        self.play(k.animate.set_value(d["k_max"]), run_time=self.SWEEP_TIME,
                  rate_func=lambda t: t)
        self.wait(0.5)

        # freeze the markers so the next sweep can reuse the tracker
        frozen = VGroup(*[Dot(kline.n2p(a), radius=0.085, color=colour)
                          for a in allowed])
        self.remove(marks)
        self.add(frozen)
        found.add(frozen)
        self.play(FadeOut(cap), run_time=0.4)
        return frozen

    # ------------------------------------------------------------------
    def _compare(self, d, kline, ring_marks, mob_marks):
        """Lift the two sets apart so the half-step offset is unmistakable."""
        self.play(ring_marks.animate.shift(UP * 0.32),
                  mob_marks.animate.shift(DOWN * 0.32), run_time=0.9)
        lbl_r = Text("ring", font_size=21, color=st.RING)
        lbl_r.next_to(kline.n2p(0), UP, buff=0.34).shift(LEFT * 0.55)
        lbl_m = Text("Möbius", font_size=21, color=st.MOBIUS)
        lbl_m.next_to(kline.n2p(0), DOWN, buff=0.34).shift(LEFT * 0.72)
        note = Text("every Möbius wave sits half a step off the ring's",
                    font_size=23, color=st.INK)
        note.next_to(kline, DOWN, buff=0.78)
        self.play(FadeIn(lbl_r), FadeIn(lbl_m),
                  FadeOut(self.klab), FadeIn(note), run_time=0.7)
        self.wait(self.HOLD * 1.5)
        self.play(FadeOut(note), run_time=0.4)
        self.compare_labels = VGroup(lbl_r, lbl_m)

    # ------------------------------------------------------------------
    def _flat_wave(self, d, axes, k, target, head):
        """The finale: the flat wave is allowed on the ring, impossible on Mobius."""
        new_head = Text("and the flattest wave of all?", font_size=30,
                        color=st.INK).to_edge(UP, buff=0.24)
        self.play(FadeOut(head), FadeIn(new_head), run_time=0.6)

        target.set_value(1.0)
        self.play(k.animate.set_value(0.0), run_time=1.0)
        ok = Text("k = 0 on the ring:  comes back as it left.  Allowed.",
                  font_size=24, color=st.RING)
        ok.next_to(axes, UP, buff=0.12)
        self.play(FadeIn(ok), run_time=0.6)
        self.wait(self.HOLD)

        self.play(FadeOut(ok), run_time=0.35)
        self.play(target.animate.set_value(-1.0), run_time=0.9)
        self.play(Flash(axes.c2p(d["N"], 1.0), color=st.MOBIUS,
                        line_length=0.22, num_lines=14, flash_radius=0.5),
                  run_time=0.7)

        bad = VGroup(
            Text("k = 0 on the Möbius band:  it would have to equal minus itself.",
                 font_size=21, color=st.MOBIUS),
            Text("Only zero does that.  The flat wave cannot exist here.",
                 font_size=21, color=st.MOBIUS),
        ).arrange(DOWN, buff=0.12)
        bad.next_to(axes, UP, buff=0.10)
        self.play(FadeIn(bad), run_time=0.7)
        self.wait(self.HOLD * 1.4)

        punch = Text("k = 0 is where the gap closes — so the Möbius walk keeps its gap.",
                     font_size=24, color=st.INK)
        punch.move_to(self.kline_caption_slot)
        self.play(FadeIn(punch), run_time=0.7)
        self.wait(self.HOLD * 1.6)


class WhichWavesFitShort(WhichWavesFit):
    """Tighter cut for a README GIF."""
    SWEEP_TIME = 4.0
    HOLD = 0.6


if __name__ == "__main__":
    # Running this file with plain Python would only define the class; Manim
    # renders through its own CLI. Forward to it so the obvious command works.
    #     manim -pqh animations/which_waves_fit.py WhichWavesFit
    import subprocess
    import sys as _sys
    _sys.exit(subprocess.call([_sys.executable, "-m", "manim", "-pql",
                               __file__, "WhichWavesFit"]))
