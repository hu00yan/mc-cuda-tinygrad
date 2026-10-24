# KDA in TinyGrad — findings

arXiv **2510.26692** (Kimi Linear, Moonshot AI). The hand-optimized
implementation the paper cites is `fla/ops/kda` in flash-linear-attention.

Reference: `kda/ref.py` (float64 NumPy, 17/17 tests, chunked form agrees with
the per-token recurrence to **5.15e-15**). Port: `kda/tiny_kda.py`
(float32 TinyGrad, 6/6 tests, **6.6e-06** against the float64 recurrence).

---

## 1. It works

`tiny_kda.py` reproduces KDA, including both sequential parts, with a static
TinyGrad graph:

- the **UT transform** — a C×C unit lower-triangular inverse by forward
  substitution, unrolled C times
- the **inter-chunk scan** — NT steps carrying a `[B,HV,K,V]` state, unrolled

| test | worst relerr |
|---|---|
| matches the per-token recurrence (9 shapes) | 6.64e-06 |
| chunk-size invariance (C = 1…32) | 4.95e-07 |
| nonzero incoming state | 8.72e-07 |
| degenerate gates (g=0, β=0, β=1, hard gate) | 1.88e-06 |
| causality (no future leakage) | 0.00e+00 |

The residual is float32 rounding, not algorithm: the same port in float64
would close it, and `ref.py` proves the algebra independently.

**So the answer to "can TinyGrad express KDA" is yes, completely.** That was
never really in doubt. The interesting question is what it costs.

---

## 2. What it costs

`B=2, T=512, H=HV=4, K=V=64, C=64` on Metal, with `q`/`k` L2-normalised (which
the real KDA layer always does — see the correction in §6):

```
kernels per call : 12,090
GPU time         : 274.3 ms
  r_ (reduce/matmul)  7,984 kernels  200.0 ms   72.9%
  E_ (elementwise)    4,106 kernels   74.2 ms   27.1%

wall: numpy float64  146.4 ms   tinygrad float32  10,604.7 ms   72.4x
```

And where the 10.6 s of wall actually goes, from `kda/replay_kda.py`:

```
DECOMPOSITION of the warm call
    1.78 s   Python constructing UOps
    9.01 s   TinyGrad's SCHEDULER (rewrite passes)
    0.11-0.27 s  GPU execution
```

**This corrects what I wrote in the first version of this note.** I reported
"99% of wall time is Python building the graph" and called it a harness
artifact. The objection — that a graph is compiled once and then replayed, so a
training loop never pays this again — is **right about the conclusion and I
stated the cause wrong.** It is not graph construction (16%); it is the
**scheduler**. TinyGrad does cache compiled kernels: the first call costs
49.5 s including compilation, and steady-state calls do not recompile. But
`realize()` re-runs every rewrite pass over the ~12,000-UOp graph every time,
because the API has no separate "plan" and "execute" step.

So the accurate statement is: **the compiled code is fine; the planning is
redone per call.** A replay path that cached the *scheduled program list*, not
just the compiled kernels, would bring the per-call cost to the ~110-270 ms of
GPU time. That is a bigger and more tractable change than a graph-replay API
that merely avoids rebuilding Python objects, which is what I originally
proposed.

### 2b. 12,090 kernels, and the prediction was 72

I predicted ~72 kernels from the camera probe's finding (one extra kernel per
sequential step) plus the step count C + NT = 72. The actual number is
**168× that**. Two reasons, and the second is the interesting one:

1. **The UT transform is O(C²) in graph ops, not O(C).** Each column's
   substitution is a sequential sum over `r` previous entries, so the whole
   thing is C(C-1)/2 = **2,016 elementary steps** at C=64, not 64. TinyGrad
   unrolls all of them into the graph.

2. **The `[C,C,K]` intermediates dominate the time, not the sequencing.** From
   `kda/perf_kda.py`:

   | tensor | elements | bf16 |
   |---|---|---|
   | `Dl` (decayed gram) | 16,777,216 | 33.6 MB |
   | `Dq_all` (causal gram) | 16,777,216 | 33.6 MB |
   | everything else | ≤ 262,144 each | ≤ 0.5 MB |

   Two `[B,NT,HV,C,C,K]` tensors, **64× larger than anything else in the
   operator**, and they are pure intermediates. 76.8% of GPU time is in
   reduce kernels, which is the contraction over `K` that consumes them.

**This is the finding.** The hand-optimized kernel never builds these. It folds
the `exp` of the decay into a matmul operand load with a per-16-sub-chunk
reference scalar, so `C²K` never exists at any point. That is not a fusion the
compiler does — it is a data layout choice that has to be made before the
compiler runs.

---

## 3. Where the rank order actually landed

The pre-registered prediction (in `kda/NOTES.md`, written from source-reading
before any TinyGrad code) ranked these as hardest. Checking each against what
the port measured:

| predicted | what happened |
|---|---|
| 1. Inter-chunk scan: no scan primitive, every loop boundary materializes | **Correct, and worse than predicted.** 99% of wall is graph build. But note the *state* `S` is only 0.1 MB — the spill that matters is `v_new` (0.5 MB × NT), not the state. |
| 2. UT transform: sequential, no batched BLAS, kernel's blocked form is hand-scheduled | **Correct.** 2,016 unrolled steps. TinyGrad has `qr` and `svd` but no triangular solve, so unrolling is the only faithful option. |
| 3. The `K/Γ` division trap | **Not tested.** I did not write the divided form, so this is still a prediction. It is cheap to falsify. |
| 4. Decay dataflow: `cumsum → broadcast-subtract → exp → contract` | **Correct, and it is the actual bottleneck** — 33.6 MB × 2 of intermediates, 76.8% of GPU time. This should have been ranked #1. |
| 5. GVA head grouping | Correct but untested in the port (all my cases have HV = H except two, which do pass). |
| 6. Memory-bound, no intensity win available | Consistent with what we see. |
| 7. Two regimes (prefill vs decode) | Out of scope for the port. |

**The one that actually dominated was #4, ranked fourth.** The reason is worth
stating plainly: it is the only item on the list that is about *not
materializing something*, rather than about expressing a control flow. TinyGrad
handles control flow badly and recoverably (unroll it, pay scheduler time). It
cannot recover from materialising 67 MB of intermediates, because that is a
decision the graph has already made by the time the scheduler runs.

That is a sharper statement of the general lesson than "sequential code is hard
for a graph compiler":

> **A static-shape graph compiler can absorb arbitrary control flow at the
> cost of planning time. It cannot absorb a bad intermediate, because the
> intermediate is part of the graph's semantics.**

---

## 4. The fix, and it is a TinyGrad change

The single highest-value change: **fold the decay's `exp` into the matmul that
consumes it.** Concretely, replace

```
Z[p,j] = sum_d exp(gc[p,d] - gc[j,d]) * k[p,d] * k[j,d]     # needs [C,C,K]
```

with a form where the `exp` is applied to a `[C,C]` or `[C/16,C/16]` operand
inside the contraction, so the `C²K` tensor is never built. That is a rewrite
pass over the matmul, and it is exactly the kind of thing TinyGrad does not do
today: there is no exp-folding into a matmul operand anywhere in its scheduler,
and it would need a data-dependent shift (the decay is per-channel, so the
"reference scalar" trick only works after a 16-sub-chunk reduction).

Second, and independent of the above: **separate planning from execution.**
TinyGrad already caches compiled *kernels*; it does not cache the *schedule*.
A `Tensor.plan()` that returns a re-dispatchable program list, so that
subsequent calls re-launch without re-running the rewrite passes, would take
this port from 10.6 s to ~0.27 s per call. That is a 40x win on any RL
workload and it is a strictly more tractable change than the rewrite pass,
because the scheduling already happens and is already deterministic.

---

## 6. Corrections to the first version of this note

Two things I got wrong, both found by re-measuring after being challenged.

**6a. "99% of wall time is Python."** No: 16% is graph construction and 85% is
TinyGrad's scheduler re-running its rewrite passes on every `realize()`. The
kernels are compiled once and cached; the *plan* is not. See §2. The
practical consequence is the same (a replay path is the fix) but the fix is
different from what I described, and it is a bigger win than I claimed.

**6b. The perf numbers were measured on a graph producing NaN.** The benchmark
used `test_ref.rnd(..., l2n=False)`, i.e. un-normalised Gaussian `q` and `k`.
With `K=64`, `kᵀk ≈ 64`, so the delta-rule matrix `(I − βkkᵀ)` has eigenvalues
near `1 − 64β` and the state **diverges**: `|S|` reaches `1.9e140` in float64
and overflows to `inf` in float32, which then gives `NaN`.

This is a real property of the operator worth recording, not just a benchmark
bug: **KDA is only well-conditioned when `q` and `k` are L2-normalised**, which
is exactly what the KDA layer does (`l2norm`, `SPEC.md` §4). Any port that
accepts raw inputs will silently produce garbage outside that regime, and in
float32 it will be `NaN` rather than a large number. With `l2n=True`:

| input | `\|S\|` max (float64) | TinyGrad `o` | relerr vs recurrence |
|---|---|---|---|
| `l2n=False` | 1.9e+140 | **NaN** | — |
| `l2n=True` | 8.8e-01 | finite | **6.3e-07** |

The kernel count (12,090) and the graph shape are unaffected by the input, so
the structural findings hold; only the absolute GPU time moved when the input
was fixed.

---

## 7. Honest limits

- **Metal on an M1 Pro, not CUDA on an H100.** Every ratio involving the
  hand-optimized kernel here is a *reading of its source*, not a measurement.
  There is no NVIDIA GPU and no Triton on this machine.
- **No torch, so no fp16/bf16 path.** The whole port is float32. The real
  kernels are bf16 with an fp32 triangular solve, and the arithmetic intensity
  argument in `NOTES.md` §6 is about bf16 bytes.
- **Forward only.** No backward pass, so the ranking for training is a
  prediction.
- **The division trap (§3 item 3) is untested.**
- `Dl` and `Dq_all` at 33.6 MB each are for *this* shape. At the shape
  `NOTES.md` analyses (B=4, T=4096, H=32, K=128) they are **8.2 GB each**,
  which does not fit in this machine's 32 GB at all. So the port as written
  cannot run at the target shape; the rewrite in §4 is not an optimisation
  there, it is a precondition.

## Files

| file | role |
|---|---|
| `kda/ref.py` | float64 reference, 3 independent code paths (subagent) |
| `kda/test_ref.py` | 17 tests, chunked == per-token recurrence (subagent) |
| `kda/SPEC.md`, `kda/NOTES.md`, `kda/cost.py` | math, compiler-hardness ranking, analytic cost (subagent) |
| `kda/tiny_kda.py` | the TinyGrad port |
| `kda/test_tiny.py` | 6 tests against the float64 reference |
| `kda/perf_kda.py` | kernel count, GPU/wall split, intermediate sizes |
