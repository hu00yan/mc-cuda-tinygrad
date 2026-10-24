---
kind: reference
---

# 26 — what makes a TinyGrad port good, in this repo

**Why this file exists.** This project writes every computation twice, once by
hand and once in TinyGrad, and reports the gap. The gap is only worth reading if
the TinyGrad side is a *good* implementation. Two ports have been found doing
quadratic scalar-update work where a matmul belongs, and the difference between
"correct TinyGrad" and "good TinyGrad" is the difference between a valid study
and a worthless one. This is the briefing for the agents rewriting them.

**The rule it exists to support.** A port may not use hardware hints. What it
must do is choose the right *algorithm and the right operator for each step* —
which is a statement about the computation, not about the card. `notes/19`
withdrew the earlier version of that rule as having conflated "not tuned" with
"naive" (ledger **C-14**); this file is the corrected half.

## What is measured here, and how

No NVIDIA GPU on this host (`nvidia-smi` absent; `Device.DEFAULT` is `METAL`).
Every number below is one of three kinds, and the label is on the claim:

| tag | meaning |
|---|---|
| `MEASURED` | executed on this host, on real tinygrad 0.14.0. Every row names the instrument and the shape it was run at. |
| `READ-FROM-SOURCE` | read out of a file in `.venv/.../tinygrad/`, quoted with `file:line`. |
| `UNVERIFIED` | could not be checked here. Carried as unknown, not as true. |

Two instruments, both re-derived here rather than copied:

- **Schedule census** — hooks `tinygrad.schedule.memory_plan_rewrite`, the last
  point at which the `LINEAR` is still one UOp per kernel with logical buffers
  (`schedule/memory.py:20`, called from `schedule/__init__.py:209`). Counts
  kernels, REDUCE kernels, distinct buffers, max buffer bytes, and ALU-lane ops.
  Never calls `realize()`. Runs under `Context(DEV="CUDA")`, so it is the real
  CUDA scheduler on a machine with no CUDA device. Same argument as
  `kda/schedule_census_cuda.py:20-27`, which I re-derived rather than assumed.
- **Codegen capture** — `matmul/tg_backend.py`, unmodified. Intercepts the
  source string at the point nvcc would be called. Real tinygrad pipeline
  (`Tensor.schedule_linear` → `lower_and_compile` → `to_program`), stock
  renderer, stock scheduler.

One trap to know before running either: `lower_and_compile` compiles in a
**multiprocessing worker pool** when more than one kernel is outstanding
(`engine/realize.py:273`), so an in-process capture hook sees only the
stragglers. `PARALLEL=0` makes the pool `None` and everything runs in-process.
Without it, a 12-step graph silently captures **0** kernels instead of 10. This
is the mechanism behind the old "captured 1 kernel" result
(`kda/schedule_census_cuda.py:37-41`).

---

## 1. Batched matmul: one `Tensor.matmul`, always

**Rule 1. Never loop over the batch. Put the batch in the leading axis and call
`Tensor.matmul` once.**

`[nb, C, C] @ [nb, C, C]` is a single kernel. The batch axis becomes the grid
axis; there is no per-block op, and nothing to loop over.

`MEASURED` (codegen capture, sm_80, fp16):

| shape | kernels | grid | threads/blk | tensor core |
|---|---|---|---|---|
| `[8, 64, 64]` | 1 | 8 | 64 | yes |
| `[512, 64, 64]` | 1 | 512 | 64 | yes |
| `[4096, 4, 64, 64]` | 1 | 16 384 | 64 | yes |

The mechanism: `Tensor.dot` (`mixin/op.py:367-392`) reshapes both operands and
returns `(x*w).sum(-1)`. A leading batch axis is just more indices on that one
`MUL` + `REDUCE`. It is not special-cased, and it does not need to be.

**The python-loop form is catastrophic and looks fine.** `Tensor.stack(*[a[i] @ b[i]
for i in range(nb)])` is also ONE kernel, so a kernel count will not catch it.
The damage is inside that kernel:

`MEASURED` (census, fp32, `DEV=CUDA`), `alu_lane` = static ALU uops × trip count:

| nb | batched | loop | ratio | ratio ÷ nb |
|---|---|---|---|---|
| 8 | 441 | 392 490 | 890× | 111 |
| 64 | 3 969 | 31 759 938 | 8 002× | 125 |
| 512 | 32 193 | 2 089 261 314 | 64 898× | 127 |

**The penalty is linear in the batch size and already ~890× at nb=8** — it does
not need a large batch to hurt. Both columns are also linear in `nb`, with a
per-element constant of ~62 ALU-lane ops (batched) and ~7 970 (loop), so the
ratio is a shape-independent constant of about 127, not a growing one. Quote the
ratio, not `nb²`: the "quadratic" description is wrong for this measurement, and
`notes/06-corrections-ledger.md` **C-03** is a record of exactly that kind of
error surviving in this project.

**Mechanism**, `READ-FROM-SOURCE`: the loop form's kernel is named
`r_16_8_4_8_16_4_64_64_64_…` — **63 separate reduce axes of extent 64**, against
the batched form's single one (`r_64_2_8_16_4_4_16_4`). `Tensor.stack` becomes
`Ops.STACK`, and `expand_horizontal_reduce` (`codegen/__init__.py:222-225`)
turns a reduce with `arg[1]` horizontal axes into a serial chain of `alu` nodes
— one per stacked element. So `nb` independent GEMMs are compiled as one
GEMM-shaped kernel that re-derives the whole reduction per stacked element.

**Do not "fix" this with `.assign()`.** Writing `out[i].assign(a[i] @ b[i])` in a
loop is worse again: it is `nb` separate kernels, not one fused one
(`MEASURED`: 8 assignments → 8 kernels, 27 963 B, vs 1 kernel and 4 399 B for
`stack`, vs 1 kernel and 3 536 B for the batched form).

---

## 2. Materialisation and the host round trip

**Rule 2. A device-to-host sync happens in exactly one place: `_buffer()`.
Avoid `numpy()`, `tolist()`, `data()`, `item()`, and anything whose result is a
Python number.**

`READ-FROM-SOURCE`. `Tensor._buffer()` (`tensor.py:466-473`) is the sole choke
point, and it calls `realize()`. Measured by patching `Tensor.realize` and
calling each API on a live tensor:

| call | `realize()` |
|---|---|
| `x + 1`, `x.sum()`, `x @ x` | 0 |
| `x.shape`, `x.numel()`, `len(x)` | 0 |
| `x.numpy()`, `x.tolist()`, `x.data()`, `x.item()` | **1** |
| `Tensor(np_array)` from numpy | 0 |
| `x.assign(y)` | 0 |
| `.contiguous()` on a permuted view | 0 |

`numpy()`, `tolist()` and `data()` all route to `_buffer()` (`tensor.py:532`,
`:516`, `:489`); `item()` is `self.data()[...]` (`mixin/op.py:32`).

**The trigger nobody expects: a data-dependent shape.** Reading a value to
compute a shape is a host read, by construction. `MEASURED`: `n = int(s.item())`
followed by `Tensor.zeros(n)` is 1 `realize()`, and everything after the `.item()`
re-plans from scratch.

**Correcting the KDA port's own comment.** `kda/tiny_kda.py:166-168` says a
`np.tril(Btri.numpy())` round trip "was the only place this port left the
device". That is now false as written. Grepping the file: the only `.numpy()`
calls are the two return values at `kda/tiny_kda.py:252`. The tril was already
rewritten on-device (`:169-170`). The port leaves the device **twice, both at the
end** — which is the correct place for it. Treat the comment as stale, not the
code.

**Also free, contrary to expectation:** `.contiguous()`. `MEASURED` (census): a
transposed view is 0 kernels on its own, and `(x*2).T.contiguous()` is **1**
kernel, same as `x*2` with no contiguous at all. Movement ops are absorbed into
the consumer's indexing.

---

## 3. Chains of many small ops

This is the section the two ports under repair get wrong, in opposite
directions. There are **two** fusion barriers, they are different, and they
produce completely different numbers.

### 3a. Elementwise chains fuse without limit

`MEASURED` (census, `DEV=CUDA`): a chain of 4 dependent elementwise ops per
step, 8 steps, at N from 64 to 16 777 216 elements — **1 kernel at every size**.
Sweeping the chain length from 1 to 256 steps at N = 1 048 576 — **1 kernel at
every length**, on CUDA and on METAL alike.

There is no chain length at which TinyGrad gives up on elementwise fusion. A
claim that "4 032 small ops means 4 032 kernels" is false for elementwise ops.

### 3b. A reduce in the chain is a barrier — but only if the reduce reads the carried state

`MEASURED` (census, `DEV=CUDA`). The KDA forward-substitution body
`b = b - e_r * (A[:,:,:,r,:r] * b[:,:,:,:r]).sum(-1)`, with `r` a Python int:

| C | steps | kernels | of which REDUCE |
|---|---|---|---|
| 8 | 7 | 12 | 6 |
| 16 | 15 | 24 | 14 |
| 32 | 31 | 48 | 30 |
| 64 | 63 | 96 | 62 |

**The same chain with the reduce replaced by pure elementwise: 3 kernels, at
every C from 4 to 64.** One kernel per step versus one kernel total, same
arithmetic, same shapes.

Reading the captured CUDA confirms the mechanism: the reduce's output is
materialised as its own buffer and the consumer reads it back from global
memory. For C=4 the four kernels are `E_4`, `r_2`, `r_3`, `E_4` — a reduce
kernel of **487 bytes** writing a single float, then the elementwise kernel that
needs it.

Two refinements, both `MEASURED`:

- If the reduce reads a **constant** rather than the carried state, the count
  drops but does not reach 1: at C=64, 66 kernels instead of 96. Reading the
  carried state is what makes it a true barrier.
- Replacing a contraction in the chain position with a **matmul whose output has
  width** does not split the chain at all. At `[1024,64] @ [64,256]`, chains of
  4, 8 and 16 dependent steps are **1 kernel each**; the same chain with
  `(A*b).sum(-1)` in that position is 22 (§4). This is a different shape from the
  KDA slot, and it is stated as such: the KDA slot's contraction is rank-1, and
  §4 shows a rank-1 matmul there is byte-identical to the reduce.

### 3c. A gather is a second, different barrier — and it is about the index, not the gather

Isolating the cause first, `MEASURED` (census) at 65 536 elements, 16 steps,
CUDA backend:

| gather index | kernels |
|---|---|
| invariant (same index every step) | **2** |
| advanced by a Python constant | 17 |
| advanced by a tensor | 17 |

**A gather whose index does not change fuses. A gather whose index changes every
step does not, and costs ~1 kernel per step.** In a DDA the index is derived
from the position, and the position is updated by the previous step, so the
index is a function of the carried state — it changes by construction. That is
the barrier, and it is not about the gather being "expensive".

Applied to the real thing. `MEASURED` by running the repo's own
`minecu_tg/census_main.py` on `minecu_tg/render.py`, CPU backend, `PARALLEL=0`.
Against `max_steps`, at batch 8:

| `max_steps` | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| launches | 8 | 12 | 16 | 24 | 47 |

Against batch, at `max_steps=32`:

| batch | 1 | 8 | 64 | 256 | 512 |
|---|---|---|---|---|---|
| pixels | 1 024 | 8 192 | 65 536 | 262 144 | 524 288 |
| launches | 9 | 48 | 48 | 48 | 48 |

**Linear in the step count with slope 1, and flat in the pixel count** over a
64× range. (The 47 and the 48 are the same `max_steps=32` graph measured in runs
with different `--steps` lists; `codegen.py:157-161` resets counters per row but
the compiled-program cache is shared across rows, so the two runs differ by one
cache hit. Quote the shape of the curve, not the last digit.) Batch ≥ 1024 was
already measured at 9 by `notes/19:299-306` from `out/minecu_tg/census-cpu-b*`;
I did not re-run those, they are expensive to schedule.

**So `notes/19`'s crossover is real and the barrier is the gather, not the pixel
count.** Two things follow, and the second is the one that matters for the
rewrite:

1. There is no fusion-size threshold to raise. The count is flat in size over a
   64× range of pixel counts, and a heuristic keyed on size cannot move a
   function that does not depend on size. `notes/19:829` proposes bisecting the
   batch axis to "find the constant in `schedule/` that it corresponds to". My
   measurement says that constant is not a function of the batch.
2. **A structural copy of the DDA body will not reproduce the port's numbers.**
   I wrote one, got 217 kernels at 16 384 pixels and 185 at 1 048 576 — a drift
   `notes/19` does not report — and it disagreed with the real port on both
   count and slope. Do not price a rewrite against a hand-copy of the loop. Run
   `census_main.py` on the actual `render.py`.

**Unverified:** whether anything at the port level removes the one-kernel-per-step
gather barrier. `probes/real_gather.py:1-8` is a related negative result and
`codegen/late/coalesce.py:156` is where a tensor-indexed load is asserted to
fail. Not settled here.

---

## 4. Reductions to a scalar, and when the answer is a matmul

**Rule 3. A contraction whose output is rank-1 is a reduce, and `Tensor.matmul`
will not change that. Only make it a matmul when the output has real width.**

The brief's premise — "the idiomatic alternative when that number is really a
matmul result" — is half right, and the half that is wrong matters a lot.

`READ-FROM-SOURCE` + `MEASURED`: `Tensor.matmul` is `dot`, and `dot` is
`(x*w).sum(-1)` (`mixin/op.py:390-392`). So a rank-1 contraction and a matmul
are the same UOp graph. I captured the generated CUDA for both forms of the same
computation, `[1024,64] ∘ [1024,64]`:

```
reduce  : r_64_8_4_4_8   md5 d789dcb37b2f   2112 B
matvec  : r_64_8_4_4_8   md5 d789dcb37b2f   2112 B
IDENTICAL SOURCE: True
```

Byte-identical. Writing `(A*b).sum(-1)` as the rank-1 matmul
`A.reshape(N,1,C) @ b.reshape(N,C,1)` changes nothing whatsoever — same kernel,
same schedule, same instruction count. Measured chain counts confirm it: 22
kernels either way, at 16 steps.

**When a matmul really is different: the output has width.** `MEASURED` at
`[1024,64]`, 16 steps in a chain:

| form | kernels |
|---|---|
| `(A * b).sum(-1)` | 22 |
| rank-1 matmul `A.reshape(N,1,C) @ b.reshape(N,C,1)` | 22 |
| `A @ W` with W `[64,256]` | **1** |

So the actionable rule is about **output width, not about the operator name**.
`A[:,:,:,r,:r] * b[:,:,:,:r]` summed to a scalar per `(batch, chunk, head)` is
correctly written as a reduce, and rewriting it as a rank-1 matmul is a no-op.
If the whole column could be batched into one `[..., C, C] @ [..., C, V]`, that
is a different computation with a wide output, and that one is a matmul. §10
records that the KDA UT transform is not obviously batchable that way, and that
whether a pure-matmul reformulation exists is **not settled here**.

**What a reduce to a scalar costs when it must be one.** `MEASURED`, a single
trailing reduce at varying output size: 1 kernel for 4 096 outputs and below;
**2 kernels at 65 536 outputs and above**. `SPLIT_REDUCEOP` (default 1,
`helpers.py:240`) splits a reduce whose `prod(input)/prod(output)` exceeds
`REDUCEOP_SPLIT_THRESHOLD`, default 32 768 (`schedule/rangeify.py:73`). This is
a two-kernel split, not an explosion, and it is not something a port controls.

---

## 5. What TinyGrad does not do

### 5a. It does not unroll. There is no loop, and "the trip count is known" changes nothing.

`READ-FROM-SOURCE`. The full op set of tinygrad 0.14.0 contains **no `WHILE`,
no `ASSIGN`, no loop or iterate op**. The only range op is `Ops.RANGE`, which is a
kernel-internal axis, not a control-flow construct. `tiny_kda.py:26-27` says
"unroll C=64 iterations (fine, trip count is known)" and `:30-31` says the
inter-chunk scan is "again a known trip count (T/C), so unrollable". **Both are
wrong about what unrolling means here.** Unrolling in Python is the *only*
option, and it produces one graph op per iteration — the compiler never collapses
them, because there is nothing to collapse them with.

There is a kernel-internal `UNROLL` opt, and it is not this:

- `READ-FROM-SOURCE`, `codegen/opt/postrange.py:155`: `check(amt <= 32, "don't
  unroll more than 32")`. Capped at 32.
- `heuristic.py:141-153` applies it **only to a reduce axis whose length is ≤ 32**
  (or splits by 4), and only when the upcast size is small.
- `MEASURED` (codegen): a reduce of length 16, 32, 64, 128 and 256 all emit the
  same shape of code and the same `'for('` count of 0 — because the axis is
  vectorised, not unrolled, and the K loop is a runtime `for`.

So at C=64 the framework will not unroll a 64-long axis even if you want it to.
The implication for the ports: **unrolling is not a decision the port makes and
gets credit or blame for.** It is what Python does when you write `for r in
range(C)`. What the port *does* control is the operator chosen at each step, and
that is where §1, §3 and §4 apply. Any cost model that prices "the port
hand-unrolled the loop" is pricing something the port had no alternative to.

### 5b. No triangular solve, no general `inverse`

`READ-FROM-SOURCE`. The entire matrix-factorisation surface of `Tensor` is
`qr` (`mixin/op.py:1803`) and `svd` (`mixin/op.py:1821`). There is no `inverse`,
no `solve`, no triangular solve, no `cholesky`, no `pinv`, no `lstsq`.

Both are themselves unrolled Python loops (`qr`: `for i in range(min(m, n))` at
`mixin/op.py:1808`), and the cost is not subtle. `MEASURED` (census,
`DEV=CUDA`), `Tensor.qr` on `[2, C, C]` against the C-step forward substitution
on `[1,1,1,C,C]` (§3b), same instrument:

| C | `qr` kernels | `qr` ALU-lane | fsub kernels | fsub ALU-lane | kernel ratio | lane ratio |
|---|---|---|---|---|---|---|
| 8 | 47 | 18 294 | 12 | 169 | 3.9× | 108× |
| 16 | 95 | 169 726 | 24 | 721 | 4.0× | 235× |
| 32 | 191 | 1 461 774 | 48 | 2 977 | 4.0× | 491× |
| 64 | 383 | 12 131 374 | 96 | 12 097 | 4.0× | 1 003× |

**`qr` is a flat 4× the kernels at every C, and its ALU-lane penalty grows with
C** (108× at C=8, 1 003× at C=64) because the Householder step is `O(C)` work
inside a loop that runs `C` times while the substitution's is `O(r)` inside the
same `C`. At the KDA shape it is 12 131 374 against 12 097. If a rewrite reaches
for `qr` or `svd` to express the KDA UT transform it has exchanged a C-step
forward substitution for a C-step Householder sequence, and it has changed the
numerics as well. `tiny_kda.py:155-157` states the absence of these correctly;
this table is what reaching for one would cost.

### 5c. No early exit, no per-ray divergence, no keeping a dependent chain in registers

`READ-FROM-SOURCE`. `Ops.IF` exists but is emitted only for gated stores
(`codegen/__init__.py:402-408`, `pm_linearize_cleanups`), and it is not a
control-flow construct a port can target. There is no way to say "this thread
stops here". Every step runs for every element.

**Consequence for the two ports.** A sequential recurrence with a carried state
of `[B,HV,K,V]` has nowhere to live. Each step reads the state from a global
buffer and writes it back. That is a property of the graph, not of the schedule,
and no amount of careful port-level rewriting removes it — it is why
`kda/materialisation_verdict.md` and `notes/19` are describing different things
when they disagree about kernel counts.

---

## 6. Sharp edges in this version, that cost time or change numbers silently

The two the brief names are real and confirmed. These are the others, each
either `READ-FROM-SOURCE` or `MEASURED` here.

### 6a. `Tensor.sum()` accumulates in float32, not in the input type or float64

`READ-FROM-SOURCE`, `dtype.py:216` and `mixin/reduce.py:44`:
`self.cast(sum_acc_dtype(self.dtype) if dtype is None else ...)`, and
`sum_acc_dtype` returns `least_upper_dtype(dt, getenv("SUM_DTYPE", "float32"))`.

`MEASURED` on this host: `sum_acc_dtype(half) = float`, `sum_acc_dtype(bfloat16)
= float`, `sum_acc_dtype(float) = float`.

A clean discriminator, 2²⁴ float32 values that are all 1.0 except a leading 1e8
— the case where fp32 accumulation visibly loses the small terms:

| backend | `sum()` | `sum(dtype=float64)` | equal? |
|---|---|---|---|
| CPU | 116 776 960.0 | 116 777 215.0 | **no**, off by 255 |
| PYTHON | 116 776 960.0 | 116 777 215.0 | **no**, off by 255 |

**This one is usually a blessing and occasionally a trap.** For a KDA port that
accumulates in fp32 it costs nothing — it is the reason the port is not losing
precision it did not know it was losing. For a port that expects a reduction to
accumulate in the input type it does not. Concrete, at 2²⁰ fp16 values of 0.01:
TinyGrad returns **10 488.0**, the float32 answer, on both CPU and PYTHON. A
genuine sequential fp16 accumulation of the same values returns **32.0** — fp16
stalls because 0.01 + 0.01 is not representable and the running sum stops
growing, which is a ~330× different answer and neither is `inf`. Pass `dtype=`
explicitly wherever the reference's accumulation dtype is known.

### 6b. fp32 matmul never reaches a tensor core, at any shape

`READ-FROM-SOURCE`, `codegen/opt/postrange.py:226`:
`if self.ren.target.device in ("CUDA", "NV") and tc.dtype_in == dtypes.float and
not ALLOW_TF32: continue`. `ALLOW_TF32` defaults to 0 (`helpers.py:281`,
`ALLOW_TF32 = ContextVar("ALLOW_TF32", 0)`; read out at runtime as
`ALLOW_TF32.value == 0`).

`MEASURED` (codegen, `__WMMA_` call sites in the generated source):

| dtype | shape | `__WMMA` calls |
|---|---|---|
| fp32 | 64³, 1024×64×64, 4096³ | 0, 0, 0 |
| fp16 | 256³ | 17 |
| bf16 | 256³ | 17 |

The fp32 row is the same finding `matmul/RESULTS.md` reports as the sm_75…sm_100a
FFMA column. **If a port is fp32, every matmul in it is an FFMA kernel and
comparing it against an fp16 hand-written kernel compares instruction classes,
not implementations.** A fp32 port is a legitimate thing to measure; it has to be
labelled as one.

### 6c. The tensor-core threshold in fp16 is N ≥ 8 and M ≥ 64

`MEASURED` (codegen, `__WMMA` count, K=64): N=2 → 0, N=4 → 0, **N=8 → 5**,
N=16 → 9, N≥32 → 17. Holding N=8 and varying M: M=1 → 0, M=8 → 0, **M=64 → 5**,
M=1024 → 5. Holding N=8 and varying K: K=2 → 0, K=4 → 0, **K=8 → 5**, K≥8 → 9.

So a `[64, 64] @ [64, 64]` fp16 matmul is **not** a tensor-core kernel. This is
the shape KDA's `w = M @ (Gamma * k)` and `u = M @ v` actually have
(`tiny_kda.py:198-199`: `[...,C,K] @ [...,C,K]`, C=K=V=64). A port that reaches
for a matmul at C=64 gets FFMA, and gets it *because the shape is too small*, not
because of anything about how it was written. Anyone measuring "matmul is faster
than the scalar loop" at C=64 is measuring that threshold.

### 6d. `/` is `MUL(a, RECIPROCAL(b))`, on every backend, with no flag

Already in the ledger (**C-05**, **C-10**, **C-11**). Re-confirmed here because
a port rewriting division will hit it: `mixin/elementwise.py:251` is in the
**Tensor API**, above the scheduler and above backend selection, so it fires
before the renderer and before any backend choice.

`MEASURED` on this host, 200 000 random pairs, three backends:

| backend | `a/b` wrong vs float64 | `a*fl32(1/b)` wrong | bit-identical |
|---|---|---|---|
| CPU | 200 000 | 200 000 | 200 000/200 000 |
| PYTHON | 200 000 | 200 000 | 200 000/200 000 |
| METAL | 200 000 | 200 000 | 200 000/200 000 |

(Every value in [0.5, 2) is misrounded at least once by a double rounding — the
rate is not the point. The point is that all three backends are bit-identical to
`a * fl32(1/b)`, so it is one rewrite in the API, not a per-backend defect, and
**writing `/` in a port silently gives double-rounded results on every backend
including CPU.**)

### 6e. `bf16` matmul's default output round-trip is lossy in a way fp16 is not

`READ-FROM-SOURCE`, `mixin/op.py:392`:
`return (x*w).sum(-1, dtype=dtype).cast(least_upper_dtype(x.dtype, w.dtype) if
dtype is None else to_dtype(dtype))`. With `dtype=None` the accumulator is fp32
(`sum_acc_dtype`) and the **output is cast back to the input dtype**.

`MEASURED` (K=256, random, vs float64 exact): bf16 `@` with `dtype=None` and
with `dtype=dtypes.float32` give **identical** results, max relative error
**3.07e-3** — which is bf16's epsilon, so the whole error is the output cast, not
the accumulation. fp16 gives 1.79e-4, also output-cast.

The trap is not the error, it is that `dtype=dtypes.float32` **does not fix it**
— it only changes the accumulator, and the output is still cast to bf16. A port
that wants an fp32 result out of a bf16 matmul must `.float()` the result
explicitly. On the CUDA codegen path the TC helper is
`__WMMA_8_16_16___bf16_float` returning `float4`, so the accumulator is genuinely
fp32 there; on the CPU backend measured here the same loss appears, so this is
not a CUDA-only property.

### 6f. `Tensor.realize()` is a silent no-op on an already-realised graph

`READ-FROM-SOURCE`, `tensor.py:422`:
`to_realize = [x for x in (self,)+lst if not x.uop.is_virtual and not
x.uop.has_buffer_identity()]`, then `if len(to_realize): run_linear(...)`.
Realising twice, or realising a graph whose buffers are already realised, emits
**no kernel and raises nothing**. Use `Tensor.schedule_linear()` when you want
the linear program unconditionally. This bit me during this investigation: two
probes of the same graph returned 10 kernels and 0, and the only difference was
the entry path.

---

## 7. The knobs, and why this study does not use them

Enumerated from `to_program_config` (`codegen/__init__.py:493-494`), which is the
authoritative list of context vars that change generated code. "Changes output"
below is `MEASURED` by capturing the generated CUDA under
`Context(**{...})` and md5-ing it, for a 256×256×128 fp32 matmul and the fp16
version.

| knob | default | changes output? | what it actually does |
|---|---|---|---|
| `NOOPT` | 0 | **yes** | Skips `hand_coded_optimizations` entirely (`postrange.py:346`). fp32 256×256×128: 3 509 B → **726 B**, kernel `r_8_2_8_16_4_4_64_4` → `r_256_128_256`. The untuned kernel is a correct but **unvectorised 1-accumulator** GEMM: one scalar load per operand, one `MULACC`, no `float4`, no tiling, no `LOCAL`. The tuned one is 4.8× the source. This is the knob that most changes what "the compiler" means, which is why it is the one most worth refusing. |
| `ALLOW_TF32` | 0 | **yes (fp32 only)** | Un-gates the fp32 tensor core (`postrange.py:226`). 3 509 B → **10 529 B**, and a `__WMMA` kernel instead of an FFMA one. No effect on fp16 (measured SAME). |
| `NOLOCALS` | 0 | **yes (fp32)** | Forbids shared memory (`heuristic.py:163`). fp32 3 509 B → 3 443 B. fp16 is **byte-identical** (its tensor-core kernel already uses no shared memory). |
| `IMAGE` | 0 | **yes** | Switches the renderer to image/tensor semantics. **3 kernels instead of 1** for a plain matmul (fp32: `E_32_4_8_16_4` 672 B + `E_4_16_8_16_4` 669 B + the matmul at 3 512 B). |
| `TC_SELECT` | -1 | **yes (fp16)** | Picks one entry from the arch's tensor-core table (`tc.py:94-98`, `get_cuda`). fp16 12 248 B → **4 513 B**; the selected shape has no `__WMMA`. |
| `TC_OPT` | 0 | no (measured SAME) | Gate on how many reduce axes a TC candidate may have (`heuristic.py:28`). Same at 0 and 2 for these shapes. |
| `TC` / `USE_TC` | 1 | **yes (fp16)** | 0 disables tensor cores entirely. fp16 12 248 B → 4 513 B, same as `TC_SELECT=1`. |
| `BEAM` | 0 | yes, by construction | Runs beam search over opt sequences (`postrange.py:341-344`). Reachable via `compile_linear(linear, beam=…)` or `Context(BEAM=…)` (`engine/realize.py:307`). Not exercised here — it needs a device to time candidates. |
| `TRANSCENDENTAL` | 1 | no (SAME here) | `>= 2` picks a different `exp`/`sin` expansion (`codegen/__init__.py:372`). |
| `DISABLE_FAST_IDIV` | 1 | no (SAME here) | Selects the fast integer-divide sequence (`codegen/__init__.py:371`). |
| `MAX_KERNEL_BUFFERS` | 0 | no (SAME here) | Overrides `DEVICE_MAX_BUFS` (`rangeify.py:363`). Only 3 devices have an entry (`rangeify.py:354`: `{"METAL": 31, "WEBGPU": 8, "CPU": 31}`) — **CUDA has none, so this whole rewrite never runs on CUDA.** |
| `SPLIT_REDUCEOP` | 1 | no (SAME here) | Gate on the two-kernel reduce split (`rangeify.py:73`). |

**`REDUCEOP_SPLIT_THRESHOLD` (32768) and `REDUCEOP_SPLIT_SIZE` (2²²) are plain
`getenv()` calls, not ContextVars** (`schedule/rangeify.py:73,86`) — they can only
be set as environment variables, not through `Context`.

**Why the study uses none of them.** The comparison is *hand-written kernel vs
what a compiler produces from the same computation*. A knob is a hardware hint
supplied by someone who has read the reference's SASS: `TC_SELECT` names a
specific `mma.sync` shape, `ALLOW_TF32` grants a numeric concession fp32 does not
otherwise get, `NOLOCALS` overrides the renderer's own decision about shared
memory, `NOOPT` is a way to submit a deliberately worse kernel. Using any of them
answers "what does TinyGrad emit when told the answer" and stops answering "what
does TinyGrad emit when given the computation". The gap is the measurement; a
knob replaces the gap with a number about the knob.

**The distinction `notes/19` got wrong, restated.** The knobs above are
hardware knowledge and are excluded. §1, §3, §4 and §6 are algorithm and operator
choice, and are **not** excluded — they are what "a good port" means. `minecu_tg`
recomputing an Amanatides–Woo address from three floats on all 32 steps
(`render.py:235-261`) is not a tuning decision; it is a worse implementation of
the same traversal, and it is fair game to fix. Ledger **C-14** is the correction
that makes this distinction explicit.

---

## 8. Things that look like they should help and provably do not

1. **Replacing a rank-1 reduce with a matmul.** Byte-identical generated CUDA
   (§4). `Tensor.dot` *is* `(x*w).sum(-1)` (`mixin/op.py:392`).
2. **Looping over a batch instead of batching the leading axis.** One kernel
   either way, so a kernel count does not catch it; 890× the ALU lane ops at
   nb=8 and a shape-independent ~127× at every nb (§1). This is the single most
   expensive mistake available in a port.
3. **Hoisting a `.contiguous()`.** Free. It fuses into the consumer (§2).
4. **Worrying about elementwise chain length.** 256 dependent elementwise steps
   at 1 M elements is one kernel (§3a).
5. **Raising a fusion-size threshold to fix the DDA.** The real port's kernel
   count is flat at 48 from 8 192 to 524 288 pixels (§3c). A size knob cannot
   move a function that does not depend on size. The discriminator is the gather
   index.
6. **`dtype=dtypes.float32` on a bf16 matmul to "get fp32 precision".** It
   changes the accumulator, not the output, which is still cast to bf16 (§6e).
   Measured identical results.
7. **Assuming fp32 is a fair default.** fp32 matmul gets no tensor core at all
   (§6b), and at C=64 even fp16 does not (§6c).
8. **Counting kernels to check fusion.** A python-loop batched matmul is one
   kernel and is ~127× the work (§1); a reduce-to-scalar and a rank-1 matmul are
   one kernel each and are the *same* kernel, byte for byte (§4). Kernel count is
   a necessary check, never a sufficient one.

---

## 9. Check your port against this list

Seven questions. Each is answerable by running a probe, and a negative answer is
a defect in the port, not a property of TinyGrad.

| # | question | §  |
|---|---|---|
| 1 | Is every batch a leading axis, with exactly one `Tensor.matmul` per contraction — and zero `for` loops over a batch? | 1 |
| 2 | Does the port call `numpy()`, `tolist()`, `data()`, `item()`, or use a value to compute a shape, anywhere but the final return? | 2 |
| 3 | How many kernels does the port produce, and how many *reduce* kernels? A reduce per sequential step is the flag. | 3b |
| 4 | For every contraction whose output is rank-1, is a reduce the right operator — i.e. is there a wide-output batching available that was not taken? | 4 |
| 5 | Does the port rely on unrolling being "legitimate because the trip count is known"? Nothing in this framework unrolls. | 5a |
| 6 | Is the port fp32? Then every matmul in it is FFMA and the comparison is against a different instruction class. | 6b |
| 7 | Is every accumulation dtype explicit — `sum(dtype=…)`, `matmul(dtype=…)` — wherever the reference's is known? | 6a, 6e |

## 10. Open, and named

- **The one-kernel-per-step gather barrier is characterised, not explained.** I
  know the trigger (the index changes), the count (linear in steps, slope 1),
  and that it is flat in tensor size. I do not know what, if anything, removes
  it from the port side. `probes/real_gather.py:1-8` is a related negative result
  and `codegen/late/coalesce.py:156` is where a tensor-indexed load is asserted
  to fail. `UNVERIFIED`.
- **Whether the batch ≥ 1024 collapse to 9 kernels is the same mechanism or a
  different one.** `notes/19:299-306` measured it; I re-verified only the
  flat-at-48 region and stopped, because each larger batch takes minutes of
  scheduling. If a rewrite targets the DDA, that crossover is the thing to
  understand, and it is not explained by §3c. `UNVERIFIED`.
- **Whether a wide-output reformulation of the KDA UT transform is possible at
  all.** §4 shows a wide contraction would give 1 kernel instead of 96, and §5b
  rules out `qr`/`svd`/triangular-solve as the route. A blocked or doubling
  formulation in pure matmuls — forward-substitution doubling, `C/2` independent
  half-solves composed with a `[C/2, C/2]` correction — is the remaining
  candidate, and its FLOP count, its kernel count, and whether it is numerically
  the same operator are all **not settled here.** This is the most promising
  unexploited finding in the file and the obvious first thing for the KDA
  rewrite to price.
- **fp16 vs fp32 as the port dtype.** §6b and §6c together mean the two are not
  comparable against the same hand-written baseline: an fp32 port has no tensor
  core at any shape, and even an fp16 port does not get one at C=64. Which side
  to move is a study-design question this note does not answer.
