# mc-cuda-tinygrad

Two questions, one repo:

1. **Can TinyGrad replace netherite's CUDA layer?** netherite
   (`Infatoshi/netherite`) is a from-scratch C/CUDA Minecraft 1.11.2 that
   bit-matches the real Java game and runs thousands of batched RL worlds on one
   GPU. Its game logic is a portable C core (`blaze/core/*.h`, `MC_HD`
   `__host__ __device__`); the CUDA layer on top is a parallel decomposition of
   the same code.
2. **Where does TinyGrad's automatic optimisation stop on complex operators?**
   KDA (Kimi Delta Attention) is the specimen.

**Both answers are yes, with one caveat each.** TinyGrad expresses the
semantic camera and KDA completely, including their sequential parts. It
cannot deliver bit-exactness against netherite (three separate backend
properties, none of them a knob), and on KDA it materialises 67 MB of
intermediates that a hand-written kernel never builds.

| report | verdict |
|---|---|
| [`notes/01-obs-camera.md`](notes/01-obs-camera.md) | 100% block-id agreement with the C reference on 41,472 pixels. 4 orders of magnitude slower, and the reason is not the kernel. |
| [`notes/02-obs-camera-residuals.md`](notes/02-obs-camera-residuals.md) | Why bit-exactness is unreachable: FMA contraction with no knob, and **float32 division that is one ULP wrong in 25.9%–27.0% of general `a/b` cases** — `derivation/divide.py:mechanism_sentence`, which is the sentence to cite. |
| [`notes/03-kda.md`](notes/03-kda.md) | KDA ports completely (6.6e-06). 12,090 kernels; the bottleneck is two `[C,C,K]` intermediates, not the sequential loops. |
| [`notes/DECISIONS.md`](notes/DECISIONS.md) | 17 decisions, 11 bugs. The bugs are the interesting part. |

## Layout

```
netherite/            sparse clone of Infatoshi/netherite (blaze/ + docs/ only)
derivation/           executable claims; a number in a document is cited here
  divide.py             TinyGrad's float32 `/`: mechanism, rate, divisor law
  divide_backends.py    the same, per backend, from the backends on this host
probes/
  bench.py            timing helper (rebuilds the graph per timed call)
  gather_floor.py     does a TinyGrad gather have an O(source) floor?
  fp32_correctness.py is TinyGrad's float32 arithmetic correctly rounded?
  real_gather.py      the custom_kernel plugin attempt -- and why it fails
  oc/                 probe 1: the semantic camera
    ref_oc.c            C reference, built from netherite's own obs_camera.h
    tiny_oc.py          the TinyGrad port
    np_oc.py            independent NumPy third opinion, bit-exact with C
    run_oc.py           correctness + step depth
    oc_report.py        every number in notes/01
    dbg_oc.py trace_oc.py trace2_oc.py   divergence finders
kda/                  probe 2: KDA
  SPEC.md NOTES.md    math, compiler-hardness ranking (research)
  ref.py test_ref.py  float64 reference, 17 tests, chunked == recurrence
  cost.py             analytic FLOP/byte model
  tiny_kda.py         the TinyGrad port
  test_tiny.py        6 tests against the float64 reference
  perf_kda.py         kernel count, GPU/wall split, intermediate sizes
```

## Reproducing

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python tinygrad numpy tabulate

cd probes/oc && make check      # camera: correctness against the C reference
cd ../../kda   && ../../.venv/bin/python test_ref.py && ../../.venv/bin/python test_tiny.py

cd .. && .venv/bin/python probes/fp32_correctness.py
       .venv/bin/python probes/gather_floor.py
```

## Conventions

- Every claim in `notes/` has a command that reproduces it.
- **A number is derived, never transcribed.** Every rate in these documents is
  printed by a module in `derivation/`, and the citation names the claim that
  prints it. When a probe's own arithmetic was wrong, its figure was propagated
  ~58 times before anyone noticed; the fix is structural — the figure now lives
  in exactly one executable place, and a document that cannot derive a number
  deletes it rather than hedging it.
- **GPU time and wall time are reported separately.** They differ by 90x on the
  camera and 90x on KDA, and conflating them gives the wrong answer in both
  directions. For both operators the majority of wall time is Python building
  the graph, which a real training loop does not pay per step.
- A third implementation whenever two disagree. Of the 11 bugs, 4 were
  *shared* transcription errors or wrong-orientation mistakes that produced
  finite plausible output; a two-way diff would have misattributed all of them.
- `Tensor.realize()` on an already-realized graph is a silent no-op, so
  `probes/bench.py` rebuilds the graph on every timed call. Getting that wrong
  reads as ~3 ns.

## What we would do next

1. **Write the two upstreamable TinyGrad patches.** (a) a correctly-rounded
   divide — the Tensor API lowers `/` to `a · fl32(1/b)`, above the scheduler and
   above backend selection, so every backend that does not itself render a real
   divide returns a double-rounded result and there is **no user-facing knob**
   (`derivation/divide.py:mechanism_no_knob`); (b) a graph-replay primitive,
   which is 99% of the wall time on both operators.
2. **The exp-folding rewrite for KDA.** A rewrite pass that applies the decay's
   `exp` inside the matmul that consumes it, so `[B,NT,HV,C,C,K]` is never
   built. Without it the port cannot run at the target shape at all
   (8.2 GB per intermediate).
3. **Probe the second netherite kernel class.** The camera was
   gather-sequential-per-lane. `blaze/env/blaze_cuda.cu`'s
   `k_recenter_fill` is the opposite: a warp-cooperative bulk copy that
   netherite says was its own dominant cost. If TinyGrad is good at that, the
   answer generalises; if it is also bad there, the problem is deeper than
   sequencing.
