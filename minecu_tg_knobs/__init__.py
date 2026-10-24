"""minecu_tg_knobs -- arm B: the same DDA, plus the codegen knobs the framework
actually exposes.

THE QUESTION THIS ARM EXISTS TO ANSWER
--------------------------------------
The study classifies every gap between TinyGrad's generated code and the
hand-written CUDA as one of four things:

    (i)   cannot be expressed in the framework at all
    (ii)  expressible, but the scheduler or cost model does not find it
    (iii) not implemented
    (iv)  does not exist for this architecture

Arm A -- the plain port, every knob at its default -- **cannot separate (ii)
from (i) and (iv)**. If the plain version is slow, that is equally consistent
with "the compiler cannot express this" and with "the compiler can express it
but does not look for it," and those are opposite conclusions.

Arm B runs the same computation with the framework's own knobs set. If a knob
recovers most of the gap, the capability exists and the search is not choosing
it -- class (ii). If no knob helps, that is evidence for (i), (iii) or (iv),
and the note has to say which and why it can tell them apart.

WHAT IS AND IS NOT ALLOWED IN THIS LANE
----------------------------------------
FORBIDDEN, because using any of them would replace the experiment with the
reference's own kernel and measure nothing about code generation:

  * `Tensor.custom_kernel` -- it takes CUDA source and emits it
  * inline PTX
  * `Context(...)` to STEER THE SCHEDULER (a tile size, a group size, an
    explicit `Opt` list). Setting a named ContextVar that the code generator
    reads is a knob and is what this arm is about; handing the scheduler a
    schedule is arm C and is a different question.
  * hand-written `.cu` files compiled alongside

Every line of the DDA is ordinary TinyGrad, so the framework's codegen is what is
under test.

FILES
-----
    registry.py     the eight knobs, each with the tinygrad line that READS it,
                    its real Context key, and whether that read is reachable for
                    a kernel with no reduction axis. Three run-time gates.
    lean.py         arm B's port: the same masked-array DDA, carrying the index
                    arithmetic Amanatides-Woo specifies. NOT a kernel -- no
                    knob, no escape hatch, no hand-unrolling.
    verify.py       the lean port against an independent float64 transcription,
                    on the discrete step count as well as the image. A
                    correctness gate and not performance evidence.
    codegen.py      the offline census: real sm_75 CUDA per (port, knob
                    configuration), with a sha256 per kernel so "identical" is
                    a byte comparison. Runs on the NULL device -- no GPU.
    sass.sh         the captured CUDA -> real sm_75 SASS, offline, in the
                    archlab container.
    sass_census.py  the SASS table, through the repo's ONE shared categoriser
                    (`tools/archlab/inspect_ptx.py:CATEGORIES`).
    battery.py      the on-card timings. Refuses to run on anything but CUDA.
    kaggle/run.sh   one push, one run, both ports, same card, same session.

THE CONSTRAINT THAT SHAPED THE MOST
------------------------------------
`minecu_tg/render.py` -- arm A's port -- recomputes the flat voxel address from
three float coordinates on all 32 steps, and ledger **C-14** withdrew the
26.2x-vs-7198.9us comparison that used it as a measurement of the compiler. A
knob sweep on top of that port would measure the port's defect and the knob at
the same time, and attributing the result to the compiler would be the error
C-14 exists to stop. Hence `lean.py`, and hence the rule that **arm B's
absolute ratios are not comparable to arm A's** -- only the within-arm-B
comparisons are attributable, and they are exact.

At the time this arm ran, `notes/23-good-vs-naive.md` did not exist and
`minecu_tg_opt/verify_opt.py`'s log FAILED its own "V-lean" step-count checks
(390/1024 and 411/1024 pixels differing). So there was no verified good port to
inherit, and the note says so and states the consequence.

A NOTE ON WHAT IS NOT HERE
--------------------------
This directory also held an earlier, unvalidated attempt (`armb.py`,
`census.py`, `count_steps.py`, `knobs.py`, `sass_matrix.py`, `shape.py`,
`kaggle/make_notebook.py`, `kaggle/kernel/`, all dated 2026-09-28). None of it
ever produced a result: the four run logs under `out/minecu_tg_knobs/` from that
day are all pre-flight output or a failed kernel, and there was no note. It is
superseded by the modules listed above and has been moved out of this directory
rather than deleted, to `.agent-scratch/knobs/superseded/`, so that a reader who
finds those names in a log can see what they were and that they were replaced
for a reason rather than by accident.
"""

__all__ = ["lean", "registry", "codegen", "verify", "battery", "sass_census"]
