# axis/ — TinyGrad's capability envelope on one T4

The hand-written kernels are **points on a curve**, not the definition of it.
Everything else in this study compares a hand-written kernel against a TinyGrad
port and reports a ratio; that makes the port the variable and every number a
statement about the port. This directory inverts it: it measures the framework's
best case, and produces one ladder that the study's existing subjects can be
located on.

The write-up is [`notes/33-tinygrad-axis.md`](../notes/33-tinygrad-axis.md).
This file is the map.

---

## The shape of it

```
rungs.py         the ladder. Seven operator classes, the TinyGrad arm of each, the
                hand-written arm's launch spec, and the naive-habit control.
refkernels.cu    the hand-written arm. extern "C" __global__ kernels, compiled by
                the CARD's nvcc to a device-only cubin for sm_75.
simref.py        a host-side EMULATOR of refkernels.cu -- phase-structured, with
                real shared-memory barriers, so a wrong index is caught before
                a push instead of after a queue.
harness.py       the measurement. Loads that cubin into TinyGrad's own CUDA
                context, on TinyGrad's own buffers, timed with TinyGrad's own
                cuEvent path. Both arms, one process, one card, one run.
verify.py        the CPU gate. BOTH arms against an independent float64 oracle,
                every scan entry point on its own shape, an AST check that the
                TinyGrad arm is pure Tensor, and a mutation selftest that
                re-injects the historical defects and requires the oracles to
                reject them.
classify.py      the four-way classification frame, read from the framework's
                source. Names the discriminator for each rung BEFORE the timing.
analysis.py      turns a card report into the axis table, the controls, the
                 designed contrasts, and the falloff correlates.
sass_census.py   per-function SASS instruction census for the reference cubin.
kaggle/          assemble, push, poll, fetch.
```

## Running it

```bash
# 1. the gate. CPU, no network, no GPU, ~15 s. BOTH arms, plus the mutation
#    selftest that proves the gate can fail.
.venv/bin/python axis/verify.py

# 2. the classification frame. Reads TinyGrad's source; prints file:line.
.venv/bin/python axis/classify.py

# 3. the reference's instruction classes, without a GPU. Uses the repo's
#    existing archlab/nvcc:12.9 container; nvcc is x86-64 Linux and this host
#    is arm64 macOS.
docker run --rm --platform linux/amd64 -v axis:/w archlab/nvcc:12.9 \
  /usr/local/cuda/bin/nvcc -arch=sm_75 -O3 --use_fast_math -std=c++17 \
  -diag-suppress 177 -cubin -o /w/r.cubin /w/refkernels.cu
docker run --rm --platform linux/amd64 -v axis:/w archlab/nvcc:12.9 \
  /usr/local/cuda/bin/cuobjdump -sass /w/r.cubin > /w/ref.sass
.venv/bin/python axis/sass_census.py axis/ref.sass

# 4. the measurement, on a T4.
axis/kaggle/run.sh              # gate, compile-check, assemble, push, poll, fetch
axis/kaggle/retry.sh            # same, but polls until a Kaggle GPU slot frees

# 5. the tables, on the host, from the JSON the card produced.
.venv/bin/python axis/analysis.py out/axis/<stamp>/axis_report.json
```

## The rules this holds itself to

These are the study's, restated where the code enforces them.

1. **Re-derive, never transcribe.** Every number is printed by a script in this
   directory. `analysis.py` copies cells out of a card-produced report and says
   which column is arithmetic on them.

2. **A passing gate is not performance evidence.** `verify.py` runs before every
   push and caught defects in this harness's own code, all of them right-shaped
   and wrong. It is necessary and nowhere near sufficient.

3. **Both arms are gated, and the gate has been seen to fail.** Two of the
   costliest defects were in `refkernels.cu` -- the hand-written reference --
   which for most of this study had no host gate at all. `simref.py` emulates
   it on the CPU, and `verify.py` re-injects those defects and requires the
   oracles to reject them, signature included. A gate that has never been
   observed to fail is not known to discriminate, and a gate that only runs the
   entry point one shape happens to select is not a gate over the file.

4. **The TinyGrad arm is pure Tensor, and that is checked by AST, not by
   comment.** No `Tensor.custom_kernel`, no pasted CUDA, no inline PTX, no
   compiled-alongside `.cu` on the framework's side. The hand-written reference
   is supposed to be CUDA — that is what a reference is — and it is launched by
   the harness, never by the framework.

5. **"Best TinyGrad" means the habit is fixed everywhere.** `notes/26` and
   `notes/31` isolated one habit — reducing to a scalar inside a loop where a
   wide contraction belongs — worth ~60x on its own. Every rung's idiomatic arm
   avoids it, and the naive spellings are timed **in the same run** as separate
   labelled controls, so the price is a measurement rather than an assumption.

6. **Wall clock, never instruction counts.** `notes/06` C-20 and C-22 are two
   corrections about censuses presented as measurements. The SASS census here is
   used only to decide which of the four classes are *open* — a question about
   the binary — and never as a cost.

7. **A measured floor, not a datasheet.** Every bandwidth figure is a percentage
   of what the card did in the same run (`axis_fill`/`axis_copy`), which is
   `minecu_ref`'s discipline.

8. **A classifier names its falsifier first.** `classify.py` prints which of the
   four classes are open per rung and which experiment separates the rest,
   before any ratio is read. Choosing the discriminator after seeing the result
   is the rationalisation `notes/06` C-16 records.

## A note on the measurement failure that cost two rounds

`kaggle kernels push` prints `Kernel push error: Maximum batch GPU session count
of 2 reached` and **exits 0** when the account's free-tier GPU sessions are
full. An early `run.sh` ignored that, polled for a status, saw the *previous*
run's `COMPLETE`, and downloaded its output — byte-identical files, a
successful-looking report, and a build predating every fix. Two rounds of this
harness were "measured" against stale code before it was noticed.

`run.sh` now tests the exit status and greps for the message, and refuses to
poll. The general form is the study's own: **a check that can pass by returning
the previous result is not a check.**
