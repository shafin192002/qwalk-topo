# Animations

Manim scenes for the four things in `qwalk-topo` whose meaning lives in the
*motion*, and which therefore cannot be a static figure. Everything else the
package produces is already covered by the PNGs in `../figures/`.

Deliberately not animated: the Klein Z2, the noise study, the phase diagram and
the circuit backend. Those are reproductions or side results, their PNGs say
everything a film would, and each extra GIF costs README load time.

| Scene | File | What it shows |
|---|---|---|
| **Seam flip** | `seam_flip.py` | Ring and Möbius walks as *signed* amplitudes. All 32 components agree exactly until step 8, when precisely one flips sign; that lone sign then interferes outward. |
| **Frame transport** | `mobius_frame.py` | A frame carried once around the band returns inverted — why the seam twist is `−1` and not `+1`. |
| **Which waves fit** | `which_waves_fit.py` | The payoff: a wave must meet itself after one lap, so the Möbius gluing shifts every allowed `k` by half a step — and forbids `k = 0` outright. |
| **Winding number** | `winding_loop.py` | The chiral Bloch vector sweeping the Brillouin zone, with the accumulated angle read off live: a full lap for `ν = −1`, out-and-back for `ν = 0`. |

The first three scenes carry the package's own result, in order: *why the twist
is −1*, *what the twist does*, and *what it costs you*. The fourth is the bulk
invariant, which is a separate result — there is no Möbius in it.

## Requirements

```bash
pip install manim      # Manim Community (NOT manimgl - the two are incompatible)
```

That is the whole list. Verified against a clean virtualenv with Manim 0.21.0:
all four scenes render to mp4 and gif with **no ffmpeg and no LaTeX installed**.

- **ffmpeg is not needed.** Older guides tell you to install it separately.
  Manim has since moved to PyAV, which it bundles, and encodes video in-process.
- **LaTeX is not needed**, because every scene uses `Text` (Pango) rather than
  `MathTex`. That is a deliberate constraint: it keeps the barrier to
  re-rendering these near zero.

If you add a scene, keep to `Text`. The trap is that LaTeX creeps back in
through mobjects that render *through* `MathTex` without saying so --
`DecimalNumber` is the one that caught us, and it needs `mob_class=Text` passed
explicitly. `Axes(..., include_numbers=True)` is another. The symptom is an
unhelpful `FileNotFoundError [WinError 2]` from a `latex` call, not a clear
"LaTeX missing" message, so it is worth knowing in advance.

## Rendering

```bash
manim -pql animations/seam_flip.py SeamFlip            # 480p draft, auto-plays
python animations/seam_flip.py                         # identical, shorter to type
manim -pqh animations/seam_flip.py SeamFlip            # 1080p final
manim -qm --format=gif animations/seam_flip.py SeamFlipShort   # README GIF
```

`-p` previews when done, `-q` sets quality (`l`/`m`/`h`/`k` = 480p/720p/1080p/4K).
Draft at `-pql` while iterating; `-pqh` is many times slower and wastes time on
a layout you are still changing.

Rendered files land in `media/` by default, or wherever `--media_dir` points.
Neither `media/` nor `renders/` is committed: the scenes are the source of
truth, and a video is a build artefact you can regenerate in seconds.

Each scene has a `...Short` subclass with tightened timings, meant for the
looping GIFs that go in the top-level README. Keep those under ~500 KB each —
the README already loads 3.7 MB of PNGs, and GitHub gets sluggish past 5–6 MB.

## Structure

```
_mstyle.py   palette, copied from examples/_style.py so films and figures match
_data.py     all physics: imports qwalktopo, returns NumPy arrays
*.py         scenes - drawing only, no physics
```

**No physics belongs in a scene file.** The scenes import arrays from `_data.py`
and draw them, so the animations are driven by the same verified package as the
figures rather than by a re-implementation that could silently drift from it.

`_data.py` runs standalone as a self-check, without Manim installed:

```bash
python animations/_data.py
```

It prints the seam-divergence steps, confirms the transported frame returns
negated, and reports the winding number for both phases. If that output looks
wrong, the animation will be wrong too — check it before rendering anything.

## Why signed quantities, not densities

`seam_flip.py` plots the *signed* amplitude, and `which_waves_fit.py` turns on
whether a wave returns as `+psi` or `-psi`. That is not a stylistic choice.

The seam's whole effect is a factor `-1`, and `|psi|^2` cannot see a sign, since
`|-a|^2 = |+a|^2`. At the moment of crossing, the ring and Möbius *densities are
exactly equal*. The first version of `seam_flip.py` plotted density and was
therefore animating a quantity mathematically incapable of showing its own
subject — it ran, it looked plausible, and it showed nothing. If you add a scene
about the seam, plot something that can hold a sign.

## A note on the two divergence steps

`_data.py` reports the seam crossing twice, and the gap between them is real:

```
amplitudes differ at : step 8    <- the seam stamps the -1
density differs at   : step 9    <- when it becomes visible
```

The sign lands on the amplitude one step before it shows up in `|ψ|²`, because a
sign on a single component is not observable until it interferes. `seam_flip.py`
calls out both moments rather than blurring them into one.

## Performance

Don't compute physics inside `construct()`. An updater that calls
`split_step_walk` runs it 60×/second and render time explodes. All arrays here
are built once up front — the full set takes about 0.02 s, which is also why
nothing is cached to disk. If you add a scene needing a heavy sweep (a
phase-diagram grid, say), cache that one rather than adding a cache layer for
these.
