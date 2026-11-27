# minecu/tinygrad — naive TinyGrad DDA port

This directory is the **generated side** of the study. The **hand-written side**
is `Infatoshi/mine.cu src/kernels.cu`: 3,245 lines of CUDA, vendored byte-identical to
`Infatoshi/mine.cu`, never modified.

The question this directory exists to answer is the one `notes/17-minecu-spec.md`
sets up: given the same computation, where does code-generated CUDA land
relative to hand-written CUDA, and **what is the size of the gap and what kind
of gap is it?**

## The one rule that makes the measurement valid

`render.py` is the "plain C++" equivalent — what a person writes when handed the
specification and asked to use array operations and a loop. It is not what a
person writes after reading the reference's SASS. So it does not:

- call `Tensor.custom_kernel`, or any escape hatch that emits CUDA
- set a tile size, `TC_SELECT`, `ALLOW_TF32`, a block size, or any other knob
  the code generator reads
- unroll the DDA loop by hand, fuse stages by hand, or stage data through
  global memory to shape the schedule
- pass a `Context(...)` to steer the scheduler
- special-case `max_steps` to imitate the reference's `#pragma unroll 4`

**The gap is the measurement.** Optimising this side would destroy the
experiment. Everything in `notes/19-tinygrad-vs-handwritten.md` is an
attribution of a measured difference, not an attempt to close one.

## Files

| file | what it is |
|---|---|
| `render.py` | **the subject.** The DDA render as tensor ops. Read its docstring first. |
| `world.py` | the three worldgen kernels, as tensor ops |
| `blocks.py` | block ids and the colour table, from the reference |
| `cpu_reference.py` | an independent **float64** transcription of the same algorithm |
| `verify.py` | the correctness gate. Runs on the CPU, no GPU. |
| `codegen.py` | kernel-count census and generated-source capture |
| `census_main.py` | census driver, any backend |
| `capture_cuda.py` | the generated CUDA for sm_75, on a host with no NVIDIA card |
| `analyse_codegen.py` | turns a captured schedule into a per-kernel table |
| `sass_matrix.py` | the SASS instruction matrix, one categoriser for both sides |
| `sass_tg.sh` | compiles the generated CUDA to sm_75 SASS in the archlab container |
| `bench.py` | timing and allocator accounting |
| `battery.py` | the whole measurement, in one process |
| `kaggle/refdrv.cu` | the hand-written side, driven: dumps its state, times itself per batch |
| `kaggle/make_notebook.py` | assembles the Kaggle notebook |
| `kaggle/run.sh` | verify, pre-flight, push, poll, download |

## Reproducing, without a GPU

```bash
# 1. correctness. This is the gate; it exits non-zero if the render is wrong.
.venv/bin/python minecu/tinygrad/verify.py

# 2. the schedule. Kernel launches and distinct programs, per max_steps.
DEV=CPU .venv/bin/python minecu/tinygrad/census_main.py

# 3. the generated CUDA for sm_75, captured with the CUDA renderer attached to
#    the NULL device (same code generator, no hardware).
.venv/bin/python minecu/tinygrad/capture_cuda.py --arch sm_75 --outdir out/tg-sm75 \
    --steps 1,2,4,8,16,32,64

# 4. per-kernel table from that capture
.venv/bin/python minecu/tinygrad/analyse_codegen.py out/tg-sm75

# 5. the generated CUDA compiled to real sm_75 SASS, registers and spills.
#    Needs the archlab container: bash tools/archlab/build_image.sh
./minecu/tinygrad/sass_tg.sh out/tg-sm75/max_steps_0032 sm_75 out/tg-sass

# 6. the same categoriser over the reference's SASS, for the side-by-side
.venv/bin/python minecu/tinygrad/sass_matrix.py \
    --sass bench_cuda/build/kernels_sm_75.sass \
    --res  bench_cuda/build/resources_sm_75.txt --select render_kernel
```

## With a GPU

```bash
./minecu/tinygrad/kaggle/run.sh              # verify, pre-flight, push, poll, download
./minecu/tinygrad/kaggle/run.sh --dry-run   # everything but the network
./minecu/tinygrad/kaggle/run.sh --build-only
```

One Kaggle GPU run. The notebook carries every source as base64, sha256-checks
each one before building, compiles the vendored `kernels.cu` for sm_75, runs
both sides on the same card in the same session, compares the two observations
on the state the reference itself produced, and compiles the generated CUDA for
sm_75 to get real registers and spills.

## Two things worth knowing before reading a number here

**`PARALLEL=0` is mandatory for the codegen capture, not an optimisation.**
`tinygrad/engine/realize.py:273` compiles in a multiprocessing worker pool
whenever more than one kernel is outstanding, and only takes the in-process
branch when exactly one is. `compile_cached` runs wherever the work runs, so an
in-process hook sees only the `len(todo) == 1` rounds. That is why
`kernels/build_notebooks.py` captured 1 kernel out of 12,082 for KDA
(`notes/13-schedule-vs-decay.md` §4). `codegen.py` sets it and says so.

**Every timing is printed next to the DDA step count of its own run.** A
throughput number without the workload behind it is not a measurement.
`bench_cuda/buried_agent_fix.md` is four weeks of that lesson; `battery.py`
carries the same gate (`check_configuration` before the run,
`check_workload` on its results) and refuses to report rather than caveating.

## The pre-flight gates, and the four bugs they caught

The first four pushes of `kaggle/run.sh` all died before measuring anything,
each in under two minutes of card time, and **every one of them was a bug the
local gates did not cover at the time**. They are listed because the pattern is
the useful part, not the individual fixes.

| push | died on | gate that now catches it |
|---|---|---|
| 1 | `multiple definition of launch_render_uint8_prebasis` — the notebook passed **both** `kernels.cu` and `refdrv.cu` to nvcc, and `refdrv.cu` already `#include`s `kernels.cu`. A **link** error, not a compile error, so the pre-flight that only checked "refdrv compiles" passed while the notebook failed. | the nvcc gate runs the **exact** command, for both binaries, and links them |
| 2 | `ModuleNotFoundError: No module named 'minecu.tinygrad'` — payloads were written to `/kaggle/working/<basename>`, so the package lost its directory. This died **after** the reference side had already run. | stage 1 asserts the travel names and that `minecu/tinygrad/__init__.py` landed inside the package |
| 3 | `ModuleNotFoundError: No module named 'tinygrad'` — the new import check was placed **before** `pip install tinygrad`. | stage 2 runs write-then-import in the notebook's **order** |
| 4 | `ModuleNotFoundError: No module named 'inspect_ptx'` — `sass_matrix.py` resolves `tools/archlab` relative to itself, and that directory does not exist on the card. This would have died at the SASS stage, **after** the timings. | stage 1 asserts `tools/archlab/inspect_ptx.py` is shipped, and stage 2 imports it |

Two things generalise, and they are the reason the gates are shaped this way:

- **A pre-flight that tests a different command than the notebook runs is worse
  than no pre-flight**, because it looks like a gate. Push 1 is the case: the
  gate compiled a file, the notebook linked two.
- **A gate placed before the thing it gates is a race.** Push 3 is the case: the
  check was correct and the ordering was wrong, so it failed on a dependency
  that had not been installed yet.

The two gates that already existed and worked are the ones this repo learned the
hard way, and they are not negotiable: `verify.py` on the CPU before the push,
and the nvcc build with `-arch=sm_75` before the push.
