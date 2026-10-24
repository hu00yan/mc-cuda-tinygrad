# Probe 1 — the semantic camera in TinyGrad

**Target:** `netherite/blaze/core/obs_camera.h`, ported to TinyGrad.
**Question it answers:** is netherite's CUDA layer expressible in TinyGrad, and
where does TinyGrad fail to optimise it?

Everything below is measured on this machine (M1 Pro, 32 GB, Metal) unless a
row says otherwise. Reproduce with `probes/oc/oc_report.py`.

---

## 0. Why this kernel first

`blaze/cuda/obs_camera.cu` is the cleanest specimen of "the CUDA layer" in the
whole repo:

- one thread per `(pose, pixel)`, DDA march state in thread-local registers
- one kernel launch for the whole frame
- the C reference and the CUDA kernel compile the *same* `MC_HD` header and are
  gated on bit-for-bit equality, so we have ground truth on this Mac
- published throughput: **~396 M rays/s** on their box

If TinyGrad can express this, it can express the rest. If it can't, we learn
exactly which part of the abstraction is missing.

---

## 1. It works. Nearly bit-exactly.

Three independent implementations, same region tensors, same 6 poses, same eye
`(8.5, 80, 8.5)`, region `113x104x113` (1,328,176 cells) at `(-48, 28, -48)`:

| impl | what it is | id mismatches | depth | edge |
|---|---|---|---|---|
| `ref_oc` | C, netherite's own `obs_camera.h`, scalar, 1 thread | — (truth) | — | — |
| `np_oc` | NumPy, vectorised over pixels | **0** / 41,472 | **0** | **0** |
| `tiny_oc` | TinyGrad, 96-step DDA unroll over a `[B, 2304]` ray tensor | **0** / 41,472 | 9 | 3 |

`np_oc` being bit-exact is load-bearing: it means the transcription of
`obs_camera.h` is correct, so the 12 residual TinyGrad pixels are a TinyGrad
property and not a shared misreading.

****These 12 pixels are root-caused in `02-obs-camera-residuals.md`. Short version:
the cause is not FMA (though FMA contraction is real and is a separate blocker),
it is that MC's `sin` is a 65,536-entry LUT indexed by
`mc_f2i(value * 10430.378f)`. One bucket is 9.587e-05 wide, and the yaw->radians
conversion lands exactly on a bucket boundary, so a 1-ULP float32 difference
picks a different bucket. The DDA still reaches the same block; only the
quantised depth moves.**

Two additional TinyGrad-side facts, both measured, both relevant to any
bit-exactness target:

- **FMA contraction is real and unconfigurable.** `a*b+c` differs from the
  unfused result in 27.14% of 2 M random triples and matches an exact FMA
  emulation 100% of the time. `SPEC.md` rule 4 requires the opposite.
  `product.realize()` is a complete barrier (0/500 k vs unfused) but costs
  kernels, and inserting it everywhere did not move these 12 pixels.
- **TinyGrad's float32 division is not correctly rounded** — it returns
  `a · fl32(1/b)` instead of `fl32(a/b)`, which is 1 ULP wrong in **25.9%–27.0%**
  of general `a/b` cases on a divisor-independent draw, where numpy is exact. The
  rewrite is in the **Tensor API**, above the scheduler and above backend
  selection, so it is not a Metal bug, not a codegen bug, and has no knob
  (`derivation/divide.py:mechanism_sentence`). `+`, `-`, `*` are exact over
  18,874,368 op instances. This is a standalone bug worth reporting upstream, and
  it hits this kernel hard because every DDA step size is `1.0f/dx`.

**Finding 1. Bit-exactness against netherite is not reachable today, and it is
not one missing flag.** It needs three things at once: no FMA contraction, a
correctly-rounded divide, and the same constant folding as clang. Two of the
three are TinyGrad-side, and none of them is a knob. The honest target for a
port is therefore *semantic* equivalence, which is what this achieves: 100% of
block ids, 99.97% of quantised depth.

---

## 2. The unroll is tight, and 4.8x wasted

The DDA has a data-dependent trip count, so it becomes a fixed unroll. The bound
is analytic: cell crossings to travel distance `L` with a unit direction are at
most `L*(|dx|+|dy|+|dz|)+3`, and `|d|₂ = 1` caps that sum at `√3`. At
`OC_FAR = 48` that is **86.1**, so 96 is safe. Measured max over all rays: **84**.
The bound is tight, and `--check-steps` asserts it.

But rays march a mean of only **19.8** steps. So:

> **a ray that hits at step 3 still pays 93 more iterations of full-width work.**

4.8x of the unrolled work is provably dead. This is the first thing a
TinyGrad-side rewrite should attack, and it is not a kernel-quality problem —
it is that the tensor is `[B, 2304]` and stays that width for all 96 steps
regardless of how many rays are still alive.

---

## 3. Where the time actually goes

```
THROUGHPUT   device=METAL
  np_oc    1 thread, vectorised  :     5.25 ms/frame      0.44 M rays/s
  tiny_oc  B=32, GPU only        :    39.79 ms/frame      1.85 M rays/s   (1365 kernels, 101 reduces)
  tiny_oc  B=32, wall (rebuild)  :   952.45 ms/frame      0.08 M rays/s
  ref_oc    1 thread, scalar      :      5.67 M rays/s   (this Mac, CPU)
  netherite CUDA (their box)      :    396.00 M rays/s   (published)
```

Wall time is **flat in B**: B=1 costs 923 ms, B=32 costs 998 ms. 32x the work,
8% more time. So wall is Python UOp construction, and the two costs are:

- **~912 ms Python**, building ~4,000 UOps per frame. ~230 µs per op.
- **~40 ms GPU**, spread over **1,365 kernels**.

**Finding 2. TinyGrad's fusion is excellent; its sequential-scan story does not
exist.** A 128-deep chain of elementwise ops is *one* kernel. But a
gather-chained-against-itself DDA becomes **1,365 kernels — 14 per unrolled
step.** Isolated, it is exactly `K + 5` kernels for `K` steps, i.e. **one extra
kernel launch per indexed load.** netherite's CUDA version is **one kernel for
the entire frame**, because each thread holds its march state in registers.

That is the structural gap: TinyGrad has no thread-local state, so a
per-ray sequential loop must be expressed as tensor state plus one memory
round-trip per step.

### What each gather costs

Measured on GPU, gathering a fixed 64 elements and growing the source
(`../probes/gather_floor.py`):

| source | src MB | GPU µs | reduce kernels |
|---|---|---|---|
| 65,536 | 0.3 | 29.0 | 1 |
| 262,144 | 1.0 | 30.6 | 1 |
| 1,048,576 | 4.2 | 26.9 | 1 |
| 1,328,176 | 5.3 | 44.8 | 1 |
| 4,194,304 | 16.8 | 27.6 | 1 |
| 16,777,216 | 67.1 | **669.9** | 2 |

Cost is flat in the number of elements requested, and only becomes source-bound
once the source leaves cache. Gathering *many* elements from a small source
scales with the output instead (16,384 → 1,048,576 outputs: 15.0 µs → 384.7 µs,
no reduce kernel at all).

**Finding 3. A large-source gather costs one extra kernel, plus — once the
source exceeds cache — a pass over the whole source, no matter how few elements
you actually want.** The DDA region is 1.33 M cells = 5.3 MB, and it does 96
gathers per frame, which matches the measured 101 reduce kernels and their
37.6% of GPU time. netherite's CUDA kernel reads only the cells it visits.

`Tensor.gather` is written as a one-hot matmul (`mixin/op.py:1044`), and
`_getitem` only takes a real linear-index path for consecutive *multi-dim*
tensor indices — and even then re-enters `_getitem` with a single tensor index,
landing back in the expensive form. The backend capability is one layer down:
`Ops.INDEX` + `Ops.LOAD` renders as a native address computation
(`renderer/cstyle.py:50`). So the fix is to expose it, not to build it.

---

## 4. What to do about it, in order of leverage

1. **Bucket rays by step count and shrink the tensor.** With mean 19.8 and max
   84, splitting the unroll into e.g. 4 buckets of 24 and compacting between
   them removes ~60% of the dead work. Needs a stream-compaction primitive;
   TinyGrad has the cumsum, but the gather-based scatter is the hard part.
2. **An empty-space-skipping rewrite.** Build a solidity mip of the region
   (a `2x2x2` reduce — one fused kernel, TinyGrad is good at this) and let rays
   skip runs of air in O(log) steps instead of O(cells). This attacks the root
   cause: mean 19.8 steps to cross at most 48 blocks is mostly air.
3. **Expose `Ops.INDEX` for 1-D fancy indices** (Finding 3). The backend already
   renders a real indexed load; the Tensor API just never asks for one. This is
   the best gain-to-patch-size ratio in the probe and it is a TinyGrad PR, not a
   workaround.
4. **A `scan`/`while` primitive.** The 912 ms of Python is the other half of
   the wall time and is entirely an artefact of rebuilding a 4,000-UOp graph
   every tick. A `Tensor.while_loop`, or just a documented "build once, assign
   inputs, re-realize" pattern, removes it.
5. **Correctly-rounded float32 division.** The Tensor API lowers `/` to
   `a · fl32(1/b)` above the scheduler and above backend selection, so every
   backend that does not repair it is 1 ULP wrong on a divisor-dependent fraction
   of general divides (`derivation/divide.py:mechanism_sentence`;
   `../probes/fp32_correctness.py` measures it on a device). Independent of this
   kernel, affects every reciprocal, and is a standalone bug report.

Items 3, 4 and 5 are TinyGrad patches and are worth writing up as concrete
diffs; 3 and 5 are upstreamable today. Items 1 and 2 are port-level and stay on
our side.

### The plugin route, and why it was not needed

`Tensor.custom_kernel` accepts a callable and gets the output tensor as its
first argument (`custom_kernel(*srcs)` where `srcs = (out, *args)`), so a
drop-in replacement for the gather is a few lines:

```python
def getitem(self, indices):
    if isinstance(indices, Tensor) and indices.dtype == dtypes.int32 and self.ndim == 1:
        out = Tensor.empty(indices.shape, dtype=self.dtype, device=self.device)
        return Tensor.custom_kernel(out, self, indices, fxn=lambda o, a, b: o.store(a[b]))[0]
    return orig_getitem(self, indices)
```

This was built and then abandoned, and the reason is the interesting part.
`ops_python.py` and `cstyle.py` both handle `Ops.INDEX` and will happily emit a
real indexed load, but reaching it from the Tensor API fails at
`codegen/late/coalesce.py:156` with `attempting multiple stores`. Setting
`DMC=1` to disable memory coalescing gets further and then emits
`device int *` pointer arithmetic, which MSL rejects. So the blocker is not
that the capability is missing — it is that **the index tensor has to be
staged into HBM before it can be used as an address, and the coalescer assumes
every store to a given buffer has a distinct affine address.** A runtime
permutation violates that assumption by construction.

`probes/real_gather.py` keeps the working version of this idea for reference.

**The hybrid you suggested — split the work, run part of it on the CPU — is
also worth trying, and the measurement above says where it would pay.** The
gather's cost is flat in the number of elements requested and only becomes
source-bound past cache size. At B=32 the region buffer is 170 MB, far past
cache, so a CPU-side gather would have to ship 170 MB across PCIe per step:
96 steps x 170 MB is hopeless. But the *ray setup* (LUT lookups, ray
construction) has no large source at all, and the *edge* pass touches only the
[B, 2304] outputs. Those are the parts a CPU offload could actually take.

---

## 5. Honest limits of this probe

- All TinyGrad numbers are **Metal on an M1 Pro**, which is not the hardware
  netherite targets. The 396 M rays/s reference is their NVIDIA box. The
  *ratios* here are about TinyGrad's compiler decisions, which are
  hardware-independent; the absolute gap is not.
- We only ran Metal. The CUDA path may contract differently, and the indexed
  load cliff is a TinyGrad-level property so it should reproduce.
- The camera is one kernel out of ~85 in `blaze/cuda/`. The other load classes
  (neighbourhood CA for light and fluid, global scatter/gather for the
  physics-window recenter, the per-env serial tick) are still unmeasured, and
  the recenter in particular is the one netherite itself says was its
  dominant cost.

## Files

| file | role |
|---|---|
| `ref_oc.c` | C reference harness, compiled from netherite's `obs_camera.h` |
| `tiny_oc.py` | the port |
| `np_oc.py` | independent NumPy third opinion, bit-exact with C |
| `run_oc.py` | correctness + step-depth driver |
| `oc_report.py` | all numbers in this document |
| `dbg_oc.py`, `trace_oc.py`, `trace2_oc.py` | divergence finders (kept: they found 3 of the 4 bugs) |
| `../gather_scale.py` | the indexed-load scaling table |
| `../bench.py` | timing helper (rebuilds the graph per call — see its docstring) |
