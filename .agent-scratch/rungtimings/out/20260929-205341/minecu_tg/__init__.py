"""minecu_tg -- the same voxel DDA, written as plain TinyGrad.

A hand-written CUDA voxel raymarcher exists at `minecu_ref/src/kernels.cu`
(3,245 lines, unmodified). This package is the same computation written the way
a person writes it with array operations and a loop, so the study can measure
where a code generator's output lands relative to hand-written code.

    render.py         the DDA render, as tensor ops          <- the subject
    blocks.py         block ids and the colour table, from the reference
    world.py          the worldgen kernels, as tensor ops
    cpu_reference.py  an independent float64 CPU transcription
    verify.py         the correctness gate
    codegen.py        kernel-count census and CUDA source capture
    bench.py          timing on a real card
    kaggle/           the push recipe

Read `render.py`'s docstring before quoting anything about what this code does;
the constraints that shaped it are part of the result.

NOTE ON THE NAMES. `render` is the *module*, and `render.render` is the
function. This package deliberately does not re-export the function under the
bare name `render`, because that would overwrite the module attribute and make
`minecu_tg.render.camera_basis` a confusing AttributeError later. Use

    from minecu_tg.render import render, camera_basis

or the `render_obs` alias below.
"""

from .render import camera_basis
from .render import render as render_obs
from .world import expand_world, generate_world, ground_of

__all__ = ["camera_basis", "render_obs", "generate_world", "expand_world", "ground_of"]
