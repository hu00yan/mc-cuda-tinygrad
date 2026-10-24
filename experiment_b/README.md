# experiment_b — comparison (b), and it needs no GPU

The study this directory belongs to compared compiler-generated GPU code against
hand-written GPU kernels for Kimi Delta Attention, and recorded itself as
blocked because no usable GPU was available. **That is the study's central
mistake, and this directory is the correction.**

There are two different comparisons hiding in the word "compare", and the study
kept collapsing them:

| | what is held fixed | what the gap means | status here |
|---|---|---|---|
| **(a)** | the algorithm, both sides | codegen and scheduling quality | **invalid on every card this study can reach** — `b5` |
| **(b)** | nothing; the human chose a better algorithm | can the compiler reach the human's *construction* at all | **measured, device-free** — `b1`…`b4` |

(b) is not a speed race. It asks two questions whose answers are properties of
the code rather than of a clock: **how many bytes get materialised**, and **which
instruction classes get emitted**. Both are answerable by reading a schedule and
a renderer. Neither needs a card, and `run_all.py` refuses to run if one is
visible so that "device-free" stays a property of the evidence rather than a
claim about it.

## Run it

```sh
cd <repo>
.venv/bin/python experiment_b/run_all.py --fast    # ~4 min
.venv/bin/python experiment_b/run_all.py           # full grid, ~15 min
```

No flags, no GPU, no arguments. `run_all.py` checks the artifact pins first and
stops if one has drifted, because a stale number printed confidently is worse
than a crash.

## What is where

| file | settles |
|---|---|
| `common.py` | shapes, artifact pins, the status vocabulary, the device-free census harness, the instruction-class vocabulary |
| `b1_bytes.py` | the byte accounting, read off the one emitted CUDA kernel in the repo and re-derived (index arithmetic evaluated over the whole grid, not regexed) |
| `b2_schedule.py` | the schedule census on the real CUDA backend, with **no device**; is there a 67,108,864 B kernel argument? (no) |
| `b3_kernel_count.py` | the kernel count as a function of (C, NT), fitted to an exact closed form |
| `b4_codegen.py` | the instruction-class matrix, both sides; also `--verify-spill` |
| `b5_arch_gate.py` | why (a) is invalid here, and why the compiler's ceiling is a table you can read |
| `b6_next_measurement.py` | the one measurement that would give (b) a cost, its exact command, and the decision each outcome forces |
| `repin.py` | deliberately re-pin an artifact, after re-deriving what depended on it |

## Rules this directory enforces

1. **Re-derive, never transcribe.** Every figure in `notes/15-experiment-b.md`
   is printed by a script here. If it cannot be regenerated it does not belong
   in the note.
2. **Label the regime.** `MEASURED` (an instrument ran here), `RE-DERIVED`
   (arithmetic from shapes or source), `MODELLED` (arithmetic on a stated
   premise), `INFERRED` (needs hardware this host lacks), `ARTIFACT` (read out
   of a pinned file). The same words in the code and the prose.
3. **Fail loudly on a changed artifact.** `common.require` raises
   `ArtifactDrift` rather than print a number derived from a file that moved.
4. **Refuse to run on a GPU** unless told to. See `common.require_no_gpu` for
   why that is a correctness condition and not caution.

## Device-free: how, and what it does not buy

Two tinygrad affordances, both asserted in code rather than cited:

* `Tensor.schedule_linear()` → `memory_plan_rewrite` is a pure UOp rewrite
  driven by **device strings**. It allocates nothing and launches nothing.
* `Context(DEV="CUDA")` makes `Device.DEFAULT` read the literal string `"CUDA"`,
  short-circuiting device discovery, so `CUDADevice.__init__` — which calls
  `cuInit(0)` — is never reached.

So the census asks the real CUDA scheduler a real question on hardware that
cannot run the answer. It does **not** render, compile, or launch, so it says
nothing about `CUDARenderer` output; `b4` covers rendering, also without a card,
via `DEV=NULL:CUDA:<arch>`.

What none of this buys, stated plainly so it is not overclaimed:

* **no wall clock.** Nothing here is a timing.
* **no DRAM traffic.** A kernel signature bounds what a kernel *may* touch. The
  measurement of what it *did* touch is `dram__bytes.sum`, and `b6` names it,
  costs it, and pre-registers what each outcome forces.
* **no SASS.** The instruction-class matrix compares post-ptxas PTX on one side
  against pre-ptxas CUDA C on the other. It answers *which families of
  instruction each construction can reach at all* — a well-posed question — and
  not *who issues more instructions*, which is not, across that gap.
