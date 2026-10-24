"""minecu_tg_opt -- the same DDA, written as competent code.

WHAT THIS PACKAGE IS FOR
------------------------
`minecu_tg/render.py` was written to a brief that said "write it plainly, do
not write anything hardware-adjacent", and it was executed as *write it
naively*. Per DDA step it did three float->int casts, three clips, and rebuilt
a 4-index gather from three float coordinates -- on all 32 steps. That is not
"plain", it is a worse algorithm than the one the reference implements, and the
26.2x performance ratio and the 47.2x instruction ratio derived from it
measured the port, not the compiler.

This package is the same computation written the way a competent,
performance-minded C programmer writes it, in the same language, with the same
vocabulary. The difference is entirely algorithmic. **The rule that separates
the two lanes is worth stating exactly, because it is the whole point:**

  * ALLOWED, and required: hoisting anything loop-invariant out of the step
    loop; maintaining the flat voxel address incrementally with precomputed
    strides instead of rebuilding it from coordinates; computing the per-axis
    bounds test once (as a countdown of steps-to-exit) instead of three casts
    and three clips per step; keeping the per-step live state as small as the
    algorithm allows; any other rewrite that removes redundancy.

  * FORBIDDEN: `Tensor.custom_kernel` or any escape hatch that emits CUDA; a
    tile size, `TC_SELECT`, `ALLOW_TF32`, `OPT_LEVEL` or any other codegen
    knob; hand-unrolling, hand-staging, hand-fusing; a `Context(...)`; reading
    the reference's SASS while writing and coding toward it; a
    device- or architecture-specific branch; per-thread scratch or occupancy
    tuning.

A rewrite a good C programmer would do unprompted is fair game. A rewrite that
only makes sense once you have read the SASS is not, and nothing in here needs
the SASS: every transformation is the textbook form of Amanatides-Woo and of
"do not recompute a loop invariant".

    render.py       the optimized render                     <- the subject
    ablate.py       the ladder V0..V4 that attributes the gap
    verify_opt.py   the correctness gate
    opcount.py      per-step operation count, and per-step SASS count
    capture_opt.py  the generated CUDA, captured with no GPU
    census_opt.py   kernel count vs max_steps and vs batch
    battery_opt.py  timing and memory on a real card
    kaggle/         the push recipe

Read `render.py`'s docstring before quoting anything about what this code is.
"""

from .render import camera_basis, render, render_variant, VARIANTS

__all__ = ["camera_basis", "render", "render_variant", "VARIANTS"]
