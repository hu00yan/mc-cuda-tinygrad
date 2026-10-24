# Probe 1 addendum — what the residual 12 pixels actually are

Follow-up to `01-obs-camera.md`. The first write-up blamed the 12 residual
pixels on FMA contraction. That was **half right**, and the other half is worse.
Both were measured; this file replaces the FMA-only explanation.

## Summary

| claim | verdict | evidence |
|---|---|---|
| TinyGrad float32 `+ - *` is correctly rounded | **true** | 0 errors in **18,874,368** op instances, 3 magnitude regimes (`probes/fp32_correctness.py`; the earlier "18.9 M" is the rounded form) |
| TinyGrad float32 `/` is correctly rounded | **FALSE on every backend that does not repair it** | wrong in **25.9%–27.0%** of general `a/b` cases on a divisor-independent draw, 1 ULP always (`derivation/divide.py:mechanism_sentence`). CPU is exact. Root cause: the **Tensor API** lowers `/` to `a*(1/b)` before any backend is chosen. **Not** a Metal defect, and **not** architecture-specific — a T4 reproduces it (`notes/06` C-04 retracted) |
| the reciprocal is *approximate* | **FALSE** | every failing backend is bit-identical to `a * fl32(1/b)` — a **correctly rounded** reciprocal. A reciprocal 1 ULP worse predicts **~86%**, not ~27% (`derivation/divide.py:reciprocal_sensitivity`) |
| `a/a == 1.0` exactly | **FALSE on the backends that reach the rewrite** | 30,748/200,000 return `0.99999994`; CPU is exact. There is no validity condition under which `a/a != 1` — and note the failing row is a **degenerate** test, not the rate for division (see "Correction 0") |
| TinyGrad contracts `a*b+c` into FMA | **true**, and unknobbed | 27.14% of 2 M triples differ from unfused, 0.00% from exact FMA; no user-facing flag either way |
| `product.realize()` is a complete FMA barrier | **true** | 0/500 k differ from unfused after the barrier |
| putting FMA barriers in the port fixes the 12 pixels | **FALSE** | identical result with the barrier on and off, bit for bit |
| the 12 pixels come from the sin LUT quantiser | **true** | exact-arithmetic trace below |

## The real mechanism: MC's sin LUT is a quantiser, and 1 ULP moves a bucket

`mc_math.h` is not `sinf`. It is a 65,536-entry LUT indexed by

```c
sin_table[mc_f2i(value * 10430.378f) & 65535]
```

One bucket is `2*pi/65536 = 9.587e-05` wide. So an input error of one ULP can
change `sin` by ~1e-4. For yaw = 180 degrees the conversion lands *exactly* on a
bucket boundary:

```
x  = 3.1415927            (float32)   -- identical in numpy and TinyGrad
c  = 10430.378            (float32)
exact product             = 32767.9995899240020662546157837
correctly rounded float32 = 32768.0            bits 00000047   <- numpy, and C
TinyGrad on Metal         = 32767.998          bits ffffff46   <- 1 ULP low
floor()                   = 32768   vs   32767
LUT index                 = 32768   vs   32767
```

Measured directly (`tinygrad` vs `numpy`, same input, same op):

```
numpy fyaw        : 3.1415927      bits db0f4940
tinygrad fyaw     : 3.1415927410125732  bits db0f4940     <- identical
numpy   * 10430.378 -> 32768.0      bits 00000047
tinygrad* 10430.378 -> 32767.998    bits ffffff46     <- 1 ULP low
LUT index: numpy 32768, tinygrad 32767
```

One bucket of LUT error is 9.587e-05 in `sin`/`cos`, which is ~1e-4 in the ray
direction. The DDA reaches the same block anyway, but `t` shifts enough to flip
`(int)(t*4)` on 9 pixels and the edge test on 3. **That is the whole residual.**

**This reframes the porting problem.** netherite does not merely need
`-ffp-contract=off`; it needs the *entire* float32 expression to be bit-stable
across three implementations, because a quantiser in the middle amplifies a
last-bit difference into a visible one. That is why their `SPEC.md` has a
determinism section and why their CPU==CUDA gate is described as hard-won.

## A standalone TinyGrad bug, worth reporting upstream

`probes/fp32_correctness.py`, 2,097,152 samples per cell, reference = the same
operation in float64 rounded once to float32 (exact, since a float32 product
needs 48 bits of float64). The `/` cell is now drawn with an **independent
divisor**; see the correction immediately below for what it used to measure:

| op | TinyGrad wrong | numpy wrong |
|---|---|---|
| `+` | 0 (0.000%) | 0 |
| `-` | 0 (0.000%) | 0 |
| `*` | 0 (0.000%) | 0 |
| `/` | 556,107 / 567,877 / 563,788 over the three regimes (26.5% / 27.1% / 26.9%) | 0 |

`+ - *` are exact over **18,874,368** op instances (3 ops × 3 regimes ×
2,097,152), not the 6,291,456 this table's row count suggests — 6,291,456 is one
op's worth of samples and the sentence naming all three quoted it.

### Correction 0: the `/` row measured `a/a`, not `a/b` — and that number was everywhere

The version of `probes/fp32_correctness.py` that produced this note's original
table read

```python
b = regimes[name] if op in "*/" else a     # regimes[name] IS a
```

so `regimes[name]` and `a` are the same object and the `/` row computed
**`a/a`**. The band this note published, **12.4% – 15.4%**, is therefore a
property of *self*-division and was never a rate for division. It was copied
into ~58 places across 19 files.

`a/a` is degenerate in a provable way, not harmlessly so: its exact quotient is
`1.0`, which is representable, so the second rounding starts pinned at a
**corner** of a rounding cell whose two edges are unequal, and it can only fail
on the lower quarter of the reciprocal's error range. Measured against an
independent divisor, general division is wrong **1.77x–1.86x** more often on the
same `a`.

The corrected rates, the divisor laws they depend on, and the interval they
carry are derived in `derivation/divide.py` (`published_rate_table`,
`sampling_interval`). The one-line statement of the whole finding is
`derivation/divide.py:mechanism_sentence`, and a document should cite that
rather than restate a rate.

### Correction: it is TinyGrad's `div()`, not the renderer and not any backend

This file went through **three** wrong attributions, and all three are worth
naming because each is a plausible thing to write:

1. "TinyGrad's Metal backend does not produce correctly rounded float32
   division" — blamed a backend.
2. "tinygrad's `cstyle` renderer emits `a*(1/b)`" — blamed the right *kind* of
   component, in the wrong layer. `ClangRenderer` is also a cstyle renderer, and
   it is the one that is **correct**.
3. "the backend contracts `1/x` into an approximate reciprocal" — blamed a
   mechanism that **does not exist on any backend measured**. The reciprocal is
   correctly rounded; the loss is a second rounding.

The real location is `mixin/elementwise.py:251`, in the Tensor API. And a fourth
attribution, "it is architecture-specific", is refuted by a T4 measurement
(Correction 2 below). An audit of the installed 0.14.0 source found the first
three; a measurement on real hardware was needed for the fourth.

All line numbers below are read out of
`.venv/lib/python3.12/site-packages/tinygrad/` at the version this study ran
against (tinygrad 0.14.0).

**1. The rewrite is in the Tensor API layer, above backend selection.**
`/` goes `__truediv__` → `div()`, and `div()` ends with:

```python
# tinygrad/mixin/elementwise.py:251   (in ElementwiseMixin.div)
d = a.alu(Ops.MUL, b.reciprocal())
```

`reciprocal()` is `self.alu(Ops.RECIPROCAL)` (`mixin/elementwise.py:465`). So
by the time a device is chosen, the graph already contains `MUL(a,
RECIPROCAL(b))` and **there is no `Ops.FDIV` node in the IR to render**. This is
above the renderer, above the scheduler, above backend selection. It is a
semantic decision made once, for every backend.

That is why the per-backend split is a *repair* story and not a *design* story.

**2. One backend repairs it, and the repair is a real rule.** `Ops.FDIV` exists
in the IR, but only one renderer in the tree can lower it:

```python
# tinygrad/renderer/cstyle.py:271-274   (in ClangRenderer, i.e. the CPU path)
code_for_op = {**({k:v for k,v in CStyleLanguage.code_for_op.items()
                   if k not in [Ops.EXP2, Ops.SIN, Ops.LOG2, Ops.TRUNC, Ops.RECIPROCAL]}),
               ...
               Ops.FDIV: lambda a,b,dtype: f"({a}/{b})"}
```

Line 271 **deletes `RECIPROCAL`** from the inherited map; line 274 adds `FDIV`.
And there is a decomposition rule, gated on exactly that difference:

```python
# tinygrad/codegen/decomp/op.py:133-136
# some backends emit FDIV for RECIP, in that case: a*(1/b) -> a/b
if Ops.FDIV in ops:
  pat += [(UPat.var("x").reciprocal(), lambda x: UOp.const(1.0).alu(Ops.FDIV, x))]
  pat += [(UPat.var("a") * UPat(Ops.FDIV, dtypes.floats, src=(UPat.const(1), UPat.var("b"))), lambda a,b: a.alu(Ops.FDIV, b))]
```

`ops` is the backend renderer's op set. `FDIV` is in `ClangRenderer`'s map and
in **no other renderer in the tree** (verified: `ClangRenderer` True, `PYTHON` /
`CL` / `METAL` / `CUDA` / `PTX` all False). So the rule fires on CPU and is inert
everywhere else. On CPU, `MUL(a, RECIP(1,b))` → `FDIV(1,b)` → `FDIV(a,b)` →
a real `/`, and the CPU backend is correctly rounded.

The generated sources, captured per backend in this session, show both ends of
it:

```
CPU    *((float4*)((data0_64+alu0))) = (float4){(val0[0]/val1[0]), ...};
CL     ... = (float4)((val0.x*(1/val1.x)), ...);
METAL  ... = float4((val0.x*(1/val1.x)), ...);
```

**3. Correction to the audit: `renderer/cstyle.py:274` is not dead code.** The audit
reported that the `Ops.FDIV` handler at `renderer/cstyle.py:274` is unreachable, because
"the rewrite has already happened before the renderer sees it." The first half
is right and the conclusion does not follow. Line 274 lives in `ClangRenderer`
(class begins at `renderer/cstyle.py:256`), not in the generic `CStyleLanguage`, and it
is reached — via the decomposition rule at `codegen/decomp/op.py:133-136`, which manufactures
exactly the `FDIV` node the Tensor API failed to emit. It is the load-bearing
last line of the CPU repair, and it is the direct reason the CPU column measures
0.00% wrong. The audit's *headline* (the rewrite is at the Tensor API layer) is
confirmed; its characterisation of line 274 is not. Recorded in
`notes/06-corrections-ledger.md` **C-06** and **C-12** — and **C-12 exists because
the same false claim was still sitting, uncorrected, in `bench/FAIR-BASELINE.md`
when C-06 had already settled it.** A correction that lives in one file does not
travel to the other files holding the claim; that is the same propagation problem
as the rate, one layer down.

**4. What the affected renderers actually emit.** Once `RECIPROCAL` reaches a
backend, three different mechanisms handle it — and the grouping by *result*
(PYTHON with CL and METAL) is **not** a grouping by mechanism:

| backend | renderer | has `FDIV`? | has `RECIPROCAL`? | what happens to `RECIPROCAL` | site |
|---|---|---|---|---|---|
| CPU | `ClangRenderer` | **yes** | no (deleted) | never reached — the `op.py` decomposition removed it | `renderer/cstyle.py:271-274` |
| PYTHON | `PythonRenderer` | no | yes | **not a code emitter at all.** Evaluates the node as Python `1/x` | `uop/ops.py:1289` |
| CL | `OpenCLRenderer` | no | yes | emits `(1/x)` in OpenCL C | `renderer/cstyle.py:137` (inherited) |
| METAL | `MetalRenderer` | no | yes | emits `(1/x)` in MSL | `renderer/cstyle.py:137` (inherited) |
| CUDA | `CUDARenderer` | no | yes | emits `(1/x)` in CUDA C for fp32 | `renderer/cstyle.py:425` |
| CUDA (fallback) | `PTXRenderer` | no | yes | emits `rcp.approx.f32` in PTX | `renderer/ptx.py:19` |

Two corrections to the previous version of this table, both found by checking
the class hierarchy rather than assuming it:

- **`PythonRenderer` is not a cstyle renderer.** It derives from `Renderer`
  directly, not from `CStyleLanguage`, and its `RECIPROCAL` comes from the
  interpreter table at `uop/ops.py:1289`, not from `renderer/cstyle.py:137`.
- **PYTHON's 27.13% is NOT a different cause from CL's and METAL's.** An earlier
  version of this note said CL and METAL were *approximated* — that the backend
  "contracts `1/x` into an approximate reciprocal" — and that they therefore had
  a different mechanism from PYTHON. **That is false, and it is retracted.** All
  three are bit-identical to `a * fl32(1/b)`, which is a *correctly rounded*
  reciprocal followed by one correctly rounded float32 multiply: the same
  **double rounding**, three renderings of it. The load-bearing evidence is not
  the rate but the bit-identity, and the corroborating evidence is the
  sensitivity model: a reciprocal one ULP *worse* than correctly rounded predicts
  ~86% wrong against the ~27% every one of them measures
  (`derivation/divide.py:reciprocal_sensitivity`). So the reciprocal's accuracy
  is not incidental — it is exactly what identifies it as correctly rounded, and
  an earlier claim that the rate is "insensitive to reciprocal accuracy" is
  withdrawn.

So: one mechanism, three renderings of it, one shared 1-ULP symptom, and **none
of them an approximation**. "The cstyle path" and "an approximate reciprocal" name
neither.

The generated MSL is the CL/METAL case — TinyGrad's division rendered as the
reciprocal primitive the backend has, in the backend's own source language:

```metal
... = float4((val0.x*(1/val1.x)), (val0.y*(1/val1.y)), ...);
```

That expression is the whole mechanism. There is no second transformation to
find: the rewrite is TinyGrad's, unconditional, at `mixin/elementwise.py:251`, and
the backend then evaluates `(1/b)` to its own accuracy, which on every backend
measured here is the correctly rounded one.

**5. It is not universal.** Same inputs, per backend, against correctly rounded
(re-measured this session, `probes/divide_precision.py` generator, N=200,000).
Every row is a property of a **backend and a host**; 27.13% is one divisor law at
one `n`, not a rate for `/` (`derivation/divide.py:published_rate_table`):

| backend | `a/b` wrong | equals `a*fl32(1/b)` bit for bit? | error class | source / mechanism |
|---|---|---|---|---|
| `CPU` | **0.00%** (0) | no — correctly rounded | none | emits `val0[0]/val1[0]` |
| `PYTHON` | 27.13% (54,270) | **yes** | **double-rounded** | interprets `RECIPROCAL` as Python `1/x` |
| `CL` | 27.13% (54,270) | **yes** | **double-rounded** | emits `1/val1.x` |
| `METAL` | 27.13% (54,270) | **yes** | **double-rounded** | emits `1/val1.x` |
| `CUDA` | 27.13% (54,270) `[RECORD]` | **yes** `[RECORD]` | **double-rounded** | emits `(1/x)` at `renderer/cstyle.py:425` |

The `CUDA` row is read from
`out/kaggle/20260928-014248-gpu/logs/divide_precision.log` on a T4 and is not
host-independent: the same tinygrad `CL` backend gives 27.13% on this Mac and
29.76% on the T4, and only the first is bit-identical to `a*fl32(1/b)`
(`notes/12` §5). `derivation/divide_backends.py` holds the per-backend, per-host
split; never quote a row of this table as a property of TinyGrad.

**6. `-fno-fast-math` does not prevent it — and could not.** `runtime/ops_metal.py:93`
passes it and the rewrite still happens. The reason is not that the flag governs
a different layer: the rewrite is **already in the emitted source** before the
compiler sees it, so there is nothing for a compiler flag to act on. This is the
same fact as point 7, reached from the other side.

**7. There is no knob — and the previous version of this note said the opposite
thing.** This note used to read: "`Ops.RECIPROCAL` exists and `ptx.py` renders it
as `rcp.approx.f32`, so tinygrad does have a first-class *explicitly
approximate* reciprocal. `/` simply never goes through it." That is backwards.
**`/` always goes through `Ops.RECIPROCAL`** — that is the defect. `rcp.approx.f32`
in `renderer/ptx.py:19` is the honest spelling of an approximation, and on the CUDA path
it is *not* what fires, because CUDA prefers `CUDARenderer` over `PTXRenderer`
(`runtime/ops_cuda.py:121`, renderer order `[CUDARenderer, PTXRenderer,
NVCCRenderer]`) and `CUDARenderer` renders `RECIPROCAL` as `(1/x)` in CUDA C,
delegating the decision to NVRTC. A grep of the tree finds no fast-math or
exact-division switch anywhere: the string `use_fast_math` appears **zero** times
in the installed package, `prec-div` appears zero times, and the only
`fp_fast_math` hits are struct fields in `runtime/autogen/mesa.py` and
`renderer/nir.py` (Mesa/NIR plumbing, unrelated). `Ops.FDIV` is not dead code
either: `ClangRenderer` keeps it (`renderer/cstyle.py:271,274`) and the CPU
backend's decomposition manufactures it (`codegen/decomp/op.py:133-136`), which
is why CPU is correctly rounded.

### Correction 2 — a falsifiable prediction: NVIDIA should *not* reproduce this

**Status: REFUTED (ledger P-1). Nothing in this section was measured on an
NVIDIA GPU by this host; the T4 run in `out/kaggle/20260928-014248-gpu` is.**

It is kept, with its refutation attached, because the reasoning error it contains
is the one this repo has now made twice.

**The prediction failed, and the step that failed is step 4.** `out/kaggle/20260928-014248-gpu/logs/divide_precision.log`
reports CUDA at **27.13% (54,270) wrong, bit-identical to `a*fl32(1/b)`** — the
`~27%` branch, not the `~0%` branch. Step 4 assumed that without
`--use_fast_math` NVRTC would emit `div.rn.f32`; it does not help, because the
rewrite is already in the emitted source as `a * (1.0f/b)`, and no compiler flag
reaches a decision the source has already made. **The error is not in the
division at all — it is introduced by tinygrad's own Tensor-level `a*(1/b)`
rewrite.** Steps 1, 2, 3 and 5 survive; step 4 does not, and step 6 follows from
step 4, so it does not either.

**One already-open question this prediction inherits.** If the division
mechanism is the same shape as the *measured* PYTHON case — a correctly-rounded
reciprocal, then a float32 multiply — then the error is **double rounding**, not
approximation, and the expected NVIDIA error rate is not zero but
*data-dependent*. PYTHON is the control: it is a correctly-rounded reciprocal
followed by a float32 multiply, and it measures **27.13%**. That is the single
most useful thing on this page for calibrating P-1, and it cuts against the
prediction. The two competing expectations for NVIDIA were therefore:

- **~0% wrong** → a correctly-rounded divide, and the "no fast-math" reading holds.
- **~27% wrong, bit-identical to `a*fl32(1/b)`** → double rounding, i.e. the same
  error class as PYTHON. This is *not* a refutation of the mechanism; it is a
  refinement of it, and it would mean the study's claim should be "TinyGrad's
  Tensor-level rewrite costs you a second rounding" — with no approximation
  anywhere.

**Which of those two branches it took, and what it settles.** It took the second.
So the claim this note is now entitled to is the one in
`derivation/divide.py:mechanism_sentence`: the Tensor API lowers `/` to
`a · fl32(1/b)` above the scheduler and above backend selection, every backend
that does not itself render a real divide returns a correctly rounded reciprocal
followed by one correctly rounded float32 multiply, and the result is 1 ULP
wrong on a rate that is a function of the divisor law.

**The chain of reasoning, with its confidence marked.**

| step | claim | status |
|---|---|---|
| 1 | CUDA compiles via NVRTC: `runtime/support/compiler_cuda.py:48` sets `compile_options = [f'--gpu-architecture={arch}']` | **VERIFIED** (read from source) |
| 2 | no `--use_fast_math` is ever added; the only other options are `-I` include paths and `--minimal` on NVRTC ≥ 12.4 (`runtime/support/compiler_cuda.py:53`) | **VERIFIED** (read from source; `use_fast_math` appears zero times in the package) |
| 3 | CUDA renders `RECIPROCAL` as `(1/{x})` for fp32 (`renderer/cstyle.py:425`), so NVRTC receives `a * (1.0f/b)` | **VERIFIED** (read from source, and the string template invoked directly) |
| 4 | without `--use_fast_math`, NVRTC emits `div.rn.f32` — a correctly-rounded divide — rather than `div.approx.f32` | **REFUTED** — a flag cannot undo a rewrite that is already in the source |
| 5 | therefore on NVIDIA the result is `RN(RN(1/b) * a)` — **double-rounded, not approximated** | **CONFIRMED** by the T4 log: 54,270 wrong, bit-identical |
| 6 | so the 27.13% error rate should **NOT** reproduce on NVIDIA | **REFUTED** — it reproduced |

**What each outcome would have meant.**

- **Does not reproduce.** Would have made the defect *not* a general property of
  TinyGrad's codegen; it would have narrowed to **a Metal/CL property** — those
  two frontends turning a syntactically correct `1.0f/x` into an approximate
  reciprocal. **This branch did not happen, and the "approximate reciprocal"
  reading it depended on is separately wrong**: CL, METAL, PYTHON and CUDA are
  all bit-identical to `a * fl32(1/b)`, and a reciprocal 1 ULP worse than
  correctly rounded would predict ~86% wrong, not 27%.
- **Reproduces at ~27% as double rounding, matching PYTHON's error class.**
  **This is the branch that happened.** The mechanism is refined, not refuted.
  NVIDIA is grouped with PYTHON, METAL and CL: correctly-rounded reciprocal,
  then a second rounding. The correct framing is: **the Tensor-level rewrite
  costs one extra rounding on every backend that does not repair it, and there
  is no approximation and no knob anywhere in the chain.** The upstream report
  gets *stronger*, because the rewrite is now shown to be lossy on every backend
  including the ones that looked fine.
- **Reproduces at ~27% but NOT bit-identical to `a*fl32(1/b)`.** Steps 4–5 would
  both be wrong and something other than a correctly rounded reciprocal would be
  at work. The candidates, in order: NVRTC contracting `1.0f/b` into
  `rcp.approx.f32` on its own; or `CUDARenderer` not being the renderer in play
  (`PTXRenderer`, which emits `rcp.approx.f32` directly at `renderer/ptx.py:19`,
  is the fallback and would produce exactly this); or the error coming from
  somewhere other than the reciprocal. **This did not happen** — CUDA is
  bit-identical, and the sensitivity model rules an approximate reciprocal out at
  ~86% predicted against ~27% measured.
- **Reproduces at an intermediate rate.** Would point at double-rounding
  (`RN(RN(1/b)·a)` differs from `RN(a/b)` for a measurable fraction of inputs,
  and double-rounding errors are data-dependent in a way a pure approximation is
  not), and the defect would need restating as
  "double rounding" rather than "approximation".

**The probe that would test it, and the opportunity that was already missed.**
`probes/divide_precision.py` already runs unchanged on any backend — it
enumerates `Device._devices` and prints the per-backend table — so on an NVIDIA
box it needs no modification. **It has never been run there.** A T4 (sm_75) run
*did* happen on 2026-09-27 at 23:19 UTC and
`out/kaggle/gpu-v1/SUMMARY.md` records `divide_precision | skip | 0.0 | payload
has no probes/divide_precision.py`, with all 15 battery items skipped and
`payload sha256: n/a`. The cause is traceable in
`out/kaggle/20260927-231908-gpu/dataset-version.log`: the payload dataset
version was still being created (`Dataset version is being created`) when the
kernel was pushed, so the kernel ran against no payload at all. The probe was
reachable, the hardware was available, and the run was wasted on an empty tar.
Per the same log, 30.00 h of GPU quota remained at that moment.

### "Isn't this just an approximation?" -- answered

That is the right question, and the answer is: **yes, and that is exactly the
defect.** IEEE 754 specifies float32 division, so two implementations cannot
both be right. This is not "each approximation has its own validity
conditions". The sharpest demonstration needs no statistics at all:

```
a / a  must be exactly 1.0 for every finite nonzero a
   CPU   :     0 / 200,000 wrong
   METAL : 30,748 / 200,000 wrong   (all returning 0.99999994 = 1 - 2^-24)
```

There is no validity condition under which `a/a != 1`. And `1 - 2^-24` is **not**
the signature of a reciprocal approximation — an earlier version of this note said
so and it is retracted, twice over:

- `0x3f7fffff` is the **only** wrong value `a/a` can produce, and its neighbour
  above, `1 + 2^-23 = 0x3f800001`, is **never** produced, in any regime, on any
  device measured (`derivation/divide.py:corner_predicate`). An approximation has
  a two-sided error and would produce both. A one-sided, exactly-predicted
  wrong value is the *opposite* of an approximation signature.
- the predicate is exact, not statistical: `a/a` is wrong **iff**
  `delta(a) < -2^-25`, with 0–9 disagreements in 12.6 M cases and every one an
  exact round-half-to-even tie. An approximation has no such closed form.

So this is **not an approximation and not "a legitimate approximation"** — the
reciprocal is *correctly rounded* and the loss is a **second rounding**:
`fl32(a · fl32(1/b))` against `fl32(a/b)`. What the "approximation" framing got
right is only that 1 ULP is a normal price; what it got wrong is the cause, and
the cause is what decides the fix.

The defect is that the second rounding is **silent and unknobbed**. The caller
cannot request a correctly rounded divide, cannot detect the loss, and cannot
route around it: computing `1/|dx|` as `|dx| * (1/|dx|)` was measured and differs
from the exact reciprocal in **100% of cases** (it still contains one division
plus a second rounding), and a Newton-Raphson or `rsqrt` refinement returns a
different value rather than the correctly rounded one. **There is no
division-free exact reciprocal in float32.**

netherite's DDA takes every step size as `1.0f/dx` and feeds the result into a
65,536-entry LUT with 9.587e-05 buckets. That is the worst possible
combination, and it is why the port lands 12/41,472 pixels off.

**The fix is upstream and smaller than this note previously claimed.** The
"`cstyle` emits `a/b`" framing implied a change to every renderer. It is not
needed. TinyGrad already contains a correct, general repair — the `FDIV`
decomposition at `codegen/decomp/op.py:133-136` — and it already works, on CPU.
The fix is to make that repair reachable: add `Ops.FDIV` to the op sets of the
renderers whose target language has a `/` (Metal, OpenCL and CUDA all do), and
the decomposition will emit a real divide on those backends too.
Alternatively, keep the reciprocal but make the choice *exposed*: a
`Tensor.div_exact`, or a context flag that routes `/` through `Ops.FDIV` when the
caller needs IEEE semantics. Either way the caller needs a knob, and today there
is none.

## FMA: real, controllable, and not the cause here

FMA contraction is real and is a genuine blocker for bit-exactness:

```
                    vs unfused        vs exact FMA
plain                 135,993             0
product.realize()         0           135,993
operands.contiguous() 135,993             0
```

(500,000 random float32 triples.) `product.realize()` is a complete and exact
barrier because it forces the product through HBM; copying the *operands* is
not a barrier at all.

But inserting barriers at every contraction site in the port — the ray-setup
products, the `sqrt` argument, the edge term, and the per-step `tmx + tdx`
increments — left the result **bit-identical with the barrier on and off**.
So for these 12 pixels FMA is not the mechanism, and the barrier machinery in
`tiny_oc.py` currently costs extra kernels for no correctness gain. It is left
in behind `tiny_oc.BARRIER = [True]` with this note rather than deleted, because
it is the right tool if the LUT quantiser problem is ever solved.

## The gather: corrected

The first write-up said the gather lowers to a one-hot reduction. **That was
wrong.** `probes/gather_floor.py` gathers a fixed 64 elements and grows the
source:

| source | src MB | GPU µs | reduce kernels |
|---|---|---|---|
| 65,536 | 0.3 | 29.0 | 1 |
| 262,144 | 1.0 | 30.6 | 1 |
| 1,048,576 | 4.2 | 26.9 | 1 |
| 1,328,176 | 5.3 | 44.8 | 1 |
| 4,194,304 | 16.8 | 27.6 | 1 |
| 16,777,216 | 67.1 | **669.9** | 2 |

A one-hot of [64, 1,328,176] would need 268 MB of traffic, about 2.7 ms at
M1 Pro bandwidth; the measured 44.8 µs rules that out. And gathering **many**
elements from a small source scales with the output, not the source
(16,384 → 1,048,576 outputs costs 15.0 µs → 384.7 µs, no reduce kernel at all).

What is actually happening: above roughly 256 K source elements TinyGrad emits a
second, reduce-shaped kernel. Its cost is flat in the number of elements you
want and only becomes source-bound once the source leaves cache (67.1 MB →
670 µs, 100 GB/s). So the correct statement is:

> **A large-source gather costs one extra kernel plus, once the source exceeds
> cache, a pass over the source — independent of how few elements you want.**

In the DDA that is 96 extra kernels per frame, which matches the measured 101
reduces, and 37.6% of GPU time.

## Status

- Port: **100% block-id agreement** with the C reference on 41,472 pixels;
  9 depth quanta and 3 edge bits off, root-caused to MC's LUT quantiser.
- Bit-exactness: **not achieved**, and now we know it needs (a) no FMA,
  (b) correctly-rounded division, (c) the same constant folding as clang.
  Two of those three are TinyGrad-side; all three are absent knobs.
  **Qualification, new:** (b) is only demonstrated to be missing on
  PYTHON/CL/METAL. On CUDA it is an open prediction, not a result — see
  "Correction 2 (new)" above. Do not quote this study as showing a
  backend-independent division defect until that prediction is run.
- **Open and cheap:** run `probes/divide_precision.py` on an NVIDIA backend. It
  needs no code change and it is the single measurement that decides whether
  this finding is a Metal/CL property or a TinyGrad property. It was already
  scheduled in the Kaggle battery and skipped on an empty payload
  (`out/kaggle/gpu-v1/SUMMARY.md`); 30 h of GPU quota was available.
- Open: the one 1-ULP multiply. `+ - *` measured exact over 18.9 M ops, so this
  specific `x * 10430.378` should also be exact. Next step is to read the
  generated `.metal` for that one expression and see whether the compiler
  rewrote it into a division.
