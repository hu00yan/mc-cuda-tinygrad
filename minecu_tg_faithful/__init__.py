"""minecu_tg_faithful -- the reference's strategy in TinyGrad, crossings marked.

This arm answers a different question from the naive port (`minecu_tg/`) and
the competent port (`minecu_tg_opt/`). Those two compare a hand-written kernel
against what TinyGrad generates from *its own* ideas, which cannot separate
"the compiler failed to find something a cost model could find" from "the
optimisation lives below the abstraction".

This one reproduces the reference's actual strategy, including the parts below
the abstraction, and records every place it has to cross to get there. Where
the boundary turns out to be *is the result*.

`Tensor.custom_kernel` is not used. No `.cu` is written or compiled alongside.
Everything that runs is CUDA that TinyGrad's own renderer emitted.

    render.py       the port. `mode="clean"` and `mode="faithful"`.
    crossings.py    every crossing, with TinyGrad `file:line`, plus the gaps.
    verify.py       the correctness gate: both modes, against the reference's
                    own CPU model and an independent float64 transcription.
    census.py       launch and program counts, offline, for both modes.
    probe_knobs.py  which ContextVars reach this graph at all.
    sass.py         the SASS census, both sides, through one categoriser.
    kaggle/         the T4 push: both modes and the hand-written baseline,
                    one card, one run.

Read `render.py`'s module docstring and then `crossings.py` before quoting
anything about what this code is.
"""

from .render import camera_basis, render

__all__ = ["camera_basis", "render"]
