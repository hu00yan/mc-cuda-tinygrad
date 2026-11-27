# minecu/opt — TinyGrad algorithmic ladder (naive → lean)

The same voxel DDA as `minecu/tinygrad/render.py`, written the way a competent,
performance-minded C programmer writes it, in the same language, with the same
vocabulary. This directory exists because the first port was told "write it
plainly, do not write anything hardware-adjacent" and was executed as *write it
naively* — and the difference between those two readings is most of a
performance gap.

**The rule that separates the two lanes, stated exactly:**

| allowed, and required | forbidden |
|---|---|
| hoisting anything loop-invariant out of the step loop | `Tensor.custom_kernel` or any escape hatch that emits CUDA |
| maintaining the flat voxel address incrementally with precomputed strides | a tile size, `TC_SELECT`, `ALLOW_TF32`, `OPT_LEVEL`, any codegen knob |
| computing the per-axis bounds test once, as a steps-to-exit countdown | hand-unrolling, hand-staging, hand-fusing |
| keeping the per-step live state as small as the algorithm allows | a `Context(...)` steering the scheduler |
| any rewrite a good C programmer would do unprompted | reading the reference's SASS and coding toward it; device-specific branches; per-thread scratch |

Hoisting is algorithmic and therefore required. Hardware knowledge is
forbidden. Every transformation here is the textbook form of Amanatides–Woo and
of "do not recompute a loop invariant"; none of them needs the SASS.

## What it found

At batch 1024 on a T4, render against render, output byte-identical to the
hand-written kernel's own bytes on the same card in the same run: **27.2× becomes
9.0×**. Below the fusion crossover the optimized port is 14% *slower*, and that
is reported rather than hidden. Full argument and numbers:
`notes/23-good-vs-naive.md`.

## The ladder

`render.py` carries five rungs, each the rung above plus exactly one rewrite, so
`ablate.py` can price them separately. All five produce bit-identical
observations and step counts (`verify_opt.py`, 93 checks).

| rung | the rewrite |
|---|---|
| `naive` | `minecu/tinygrad/render.py`'s body, structurally verbatim |
| `int_cursor` | the cursor is an `int`, cast once instead of 32 times |
| `flat_gather` | one flat index into the gather, not four |
| `incremental` | the address is maintained with precomputed strides; the bounds test is computed once per ray |
| `lean` | no separate `found` flag — `hit < 0` already means "not hit yet" |

The camera basis is **imported from `minecu.tinygrad.render`, not copied**, so a diff
between the two ports shows the DDA and nothing else.

## Running it

```bash
# correctness, CPU, no GPU, ~11 min
.venv/bin/python minecu/opt/verify_opt.py

# the whole ladder: capture, sm_75 compile, every count
.venv/bin/python minecu/opt/ablate.py --batch 1024 --steps 4,8,16,32,64

# the T4 run: gates on verify_opt.py locally first, then the notebook
./minecu/opt/kaggle/run.sh

# a completed run's table, re-derived from its own JSON
.venv/bin/python minecu/opt/analyse_run.py out/minecu/opt/<stamp>
```

## Files

| what | where |
|---|---|
| the optimized port, and the ladder | `render.py` — read its docstring first |
| the correctness gate, CPU, no GPU | `verify_opt.py` |
| the ladder, priced, in one command | `ablate.py` |
| the generated CUDA, captured with no GPU | `capture_opt.py` |
| the per-step counts, both sides, one categoriser | `opcount.py` |
| the sm_75 compile, offline, staging inside the repo | `compile_sass.sh` |
| the kernel census against `max_steps` and batch | `census_opt.py` |
| timing and memory on a real card | `battery_opt.py` |
| a completed run's table, re-derived | `analyse_run.py` |
| the T4 notebook and its push recipe | `kaggle/` |

## Two things this package refuses to do

**No scratch outside the working tree.** `minecu/tinygrad/sass_tg.sh` stages through
`mktemp -d "${TMPDIR:-/tmp}"`; `compile_sass.sh` stages inside the repository
and every byte it produces is evidence the note can point at.

**No codegen knob, and the note says so.** Instruction and register counts are a
property of a compiler version as much as of a source. Both sides are counted at
CUDA 12.9 in the `archlab/nvcc:12.9` container, and the Kaggle notebook
explicitly declines to quote SASS because its image carries 12.8.
