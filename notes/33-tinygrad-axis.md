---
kind: reference
---

# 33 — TinyGrad's capability envelope on one T4: an axis, and where it falls off

**The question.** Everything else in this study compares a hand-written kernel
against a TinyGrad port and reports a ratio. That framing makes the hand-written
kernel the reference and the port the variable, so each number is a statement
about our port — and twice now the port has turned out to be the reason for the
gap (`notes/06` §5b). This note does the other thing: **it measures TinyGrad's
best case and produces one curve that the hand-written kernels can be located
on.** The DDA raymarcher in `minecu_ref/` and the KDA operator against
flash-linear-attention become *points on the axis*, not the definition of it.

**Status of the numbers.** Every figure is tagged. `[MEASURED]` means executed
on a T4 by a script in `axis/`; `[READ-FROM-SOURCE]` means read out of
TinyGrad's source with a `file:line`; `[MODELLED]` means arithmetic on known
shapes; `[PREDICTED]` is never used as a result. **No number in this note is
transcribed from any other note**, per the rule in `notes/06` §5 and C-22.

**Status of the numbers.** Every figure is tagged. `[MEASURED]` means executed
on a T4 by a script in `axis/`; `[READ-FROM-SOURCE]` means read out of
TinyGrad's source with a `file:line`; `[MODELLED]` means arithmetic on known
shapes; `[PREDICTED]` is never used as a result. **No number in this note is
transcribed from any other note**, per the rule in `notes/06` §5 and C-22.
**All seven rungs are measured, `7/7 OK`**, in one run on one T4 with one
instrument (§7).

**The result, in three sentences.** TinyGrad is **level** with hand-written
CUDA on pure streaming bandwidth (`ew` 0.99×) and on a cache-missing gather
(`gather` 0.96×), so the framework's memory path and its addressing are not
where the gap is. It is **1.8–3.0× behind** where the operator needs a
contraction, and **22–36× behind on one operator, `scan`** — a prefix sum, measured
as a band rather than a point because that is what it is.
That last number is not a capability boundary and it is not a scheduling
problem: `Tensor.cumsum` computes a prefix sum by **re-summing a window of
width `min(N, 256)` per output** rather than carrying a running value, with
`SPLIT = 256` a hand-placed literal and the framework's own comment *"TODO:
someday the optimizer will find this on its own"* beside it
(`mixin/op.py:764-766`, `[READ-FROM-SOURCE]`). The re-reads hit cache, so the
cost shows up as **arithmetic, not bandwidth** — which is why `scan` has both
the lowest `%floor` and the highest ratio on the ladder. The framework **ships
good and bad lowerings of the same class in the same library**, and which one
you get is per-operator, not per-class.

**The correlate this note set out to test is refuted by the data that completed
it.** The ladder was built around *sequential depth*; the measurement says the
rung with **more** sequential depth and **more** contraction (`trisolve`, 2.97×)
came back **9× closer** to the hand-written kernel than the simpler one (`scan`,
26.0×, a representative value from a measured 22–36× band). All three candidate
structural correlates come out weak or negative
(§7). What orders the ladder instead is the quality of each operator's lowering,
and that is only visible by reading the generated program — which is why
`axis/lowering.py` exists and why it costs no GPU quota. §10 is the argument, and
§9's width sweep is the one-run falsifier — **which ran, and refuted the note's own
stated mechanism twice while confirming the class**: the window is `min(N, 256)` at
*every* N and saturates at 256, not a threshold at 512.

> **⚠ `notes/37` then falsified the surviving mechanism too, and this note's
> headline problem statement above is wrong in three ways.** It reads as though
> re-summing a window is the *injury*. It is the **mitigation**: at a fixed shape
> the framework's one-stage path measures **329.17 ms** against the two-stage
> path's **56.52 ms**, so the decomposition is 5.8× *faster* than what it replaces.
> The `SPLIT` literal **is** causal and correctly located — `2·SPLIT+1` predicts
> 129/257/513/1025/2049 and the measured switch points are exactly those — but
> the cost is **two nested windows**, per row `N·SPLIT + (N/SPLIT)²`, and this note
> counted one. Tuning the one literal recovers **4.5×**; **7.5× survives it**, and
> 4.5× is a lower bound. The cliff restates from 25.96× to **34×** against the
> reference. `notes/37` also found that **this note's hand-written scan reference
> has no correct kernel for 257 ≤ N ≤ 1023**, which is exactly the band where the
> decomposition switches — so a rung whose reference is missing there cannot price
> the mechanism it is being used to price.

**And the controls say the ladder's own rule does not hold on one rung.** The
habit `notes/26` tells ports never to write — a scalar reduction inside a loop —
costs **2.26×** against TinyGrad's idiomatic `red` `[MEASURED]`, and the same
habit is **20× *faster*** than TinyGrad's idiomatic `scan` `[MODELLED]`. So on
`scan` the framework's own lowering is worse than the spelling this study spent
two notes telling people not to write. "Best TinyGrad" is a per-operator
measurement, not a style, and §9 has the per-rung answer.

**What is NOT claimed, stated before the table.** Both GEMM references sit near
**20% of their own instruction-class peak** (§7), so those two ratios are
upper bounds on the real gap and are reported as weakened rather than as
findings. `trisolve` is measured against a **hand-written blocked inverse**,
not against flash-linear-attention, whose reference never compiled on sm_75
(§8). There are **no hardware counters** — Kaggle exposes neither `ncu` nor
`nsys` — so no conclusion here rests on occupancy, stalls or issue pressure
(§9).

```
.venv/bin/python axis/verify.py                 # the CPU gate: BOTH arms, no GPU, no network
.venv/bin/python axis/lowering.py scan         # what the framework EMITS, captured on the host
.venv/bin/python axis/classify.py               # the four-way frame, read from source
.venv/bin/python axis/sass_census.py _scratch/ref.sass   # the reference's instruction classes
.venv/bin/python axis/analysis.py out/axis/<stamp>/axis_report.json \\
                  out/axis/<stamp>/axis_controls.json   # the axis table and the controls
axis/kaggle/run.sh                              # push and measure on a T4
```

**`verify.py` is a gate over both arms, and it has been observed to fail.**
Step 3 emulates `refkernels.cu` on the host; step 4 checks every `axis_scan_*`
entry point separately (a shape-dependent symbol selection means one bench shape
exercises one of six); step 5 re-injects the two historical reference defects and
requires the oracles to reject them, signature included. §5 is the mechanism.

**`lowering.py` is what turned the largest gap from a measurement into a
diagnosis**, at zero GPU quota: it wraps `compile_cached`, realizes the rung, and
reports the kernel count and the trip counts of the generated program. On `scan`
it shows **3 kernels whose first one re-sums a 256-wide window per output**, and
`cumsum_lowering_facts()` reads the `SPLIT = 256` constant and the framework's own
TODO out of the source next to it. §7 and §10 are built on that.

---

## 1. What "the same run on the same card" has to mean here

The task requires each rung's two points to be measured in one run on one card.
That is a stronger requirement than it first looks, because the two arms of a
comparison can otherwise differ in more than the thing being compared. Three
specific ways this harness makes them differ, and closes each:

| what could differ | what this harness does instead |
|---|---|
| **the card, the clock, the driver state** | one process. The hand-written kernels are compiled by the *card's own nvcc* to a device-only cubin, loaded into **TinyGrad's own CUDA context** (`CUDADevice.context`) with `cuModuleLoadData`, and launched on the same stream. A subprocess, a second context, or a second card would each break the requirement. |
| **the data** | both arms read the *same device buffers*. Inputs are generated once with fixed seeds, uploaded once, and the reference is handed raw addresses into TinyGrad's own allocations (`harness.py:dev_ptr`). A disagreement about the input cannot hide inside a performance number. |
| **the instrument** | the same `cuEvent` path for both arms, in the same function (`harness.py:timed`). The ratio is a ratio of two readings of one clock, not of two measurement pipelines. |
| **the computation itself** | every rung is gated **before** it is timed: the reference and the TinyGrad arm are each compared against a float64 oracle and against each other. A rung whose arms disagree is reported `INVALID` and **no ratio is printed for it**. |

That last row is the one this repository has needed four times, and it is the
row that got extended last. §5 lists **twelve** defects, every one of the shape
*"right shape, wrong answer"*, and it is worth being precise about where they
were: **two of the twelve were in the hand-written reference**, whose only
host-side checks were a clean compile and a `cuobjdump` census — neither of which
can see a wrong index, and both of which are byte-identical for a correct kernel
and a broken one. `axis/simref.py` now emulates the reference on the CPU so both
arms are gated before a push, and `verify.py` re-injects those defects and
requires the oracles to reject them. The other three that are not "right shape,
wrong answer" are **instrumentation** defects — a normalisation, a selector, and
a hardcoded argument order — and they are the class that the oracle catches only
because it is looking at the answer rather than at the code.

**Why a cubin and not a linked binary.** A cubin has no host code and no
allocator. It can only touch memory this process already owns, which is what
makes "both arms, one allocation" true rather than aspirational. It also means
the hand-written side cannot quietly be using `cudaMalloc` and a different pool.

### The measurement that had to be fixed first, and it is the same error as C-03

**The first T4 numbers this harness produced were wrong, and wrong in the
direction that flatters the framework.** `ew` at `n=4096` read **3.81 ms** for
TinyGrad against **0.025 ms** for the reference — a 155× ratio on an operator
that moves 64 KB. Essentially all of the 3.81 ms was **Python**: the TinyGrad arm
was timed as `build().realize()` plus a synchronize, which constructs the graph
in Python on every call, while a cubin launch is one C call with no
counterpart. The two numbers were not comparable, and the error is
`notes/06` C-03's — dividing a quantity that includes host time by one that does
not — arriving in a new place.

The fix is structural, not a subtraction. `harness.py:tg_device_time` builds the
graph **once**, takes its linear program, compiles and links it exactly as
`Tensor.realize` would, and replays the compiled calls with
`run_linear(..., jit=True)` — which skips both `compile_linear` and
`link_linear` (`realize.py:317`) — inside **the same cuEvent bracket** used for
the reference. The graph is not rebuilt inside the timed region, so the
framework's codegen has already happened and what remains is the sequence of
kernel launches, which is what the cubin side also is.

Verified `[MEASURED]`: the replayed result is **bit-identical** to a normal
`realize` for every rung tested (`np.array_equal` true, max difference 0.0, on
`ew`, `red`, `gather`).

So each rung reports **two** numbers and the note uses the right one:

- `tg_device` — device time, cuEvent, Python excluded. **The ratio's numerator.**
- `tg` — end-to-end wall clock including the Python graph build. **Context
  only, never a denominator**, because the cubin side has no such term.

The difference between them is this framework's other cost, and it is a real
cost. It is simply not the cost the axis is about.

---

## 2. The ladder

Seven rungs, ordered by **what they demand of a compiler**, not by size. The
order is the argument; §7 reads the falloff along it.

| rung | demands of a compiler | why it is on the ladder |
|---|---|---|
| `ew` | **nothing** — no reduction, no gather, no reuse | the control. If the framework cannot reach the memory roofline here, no gap above it is interpretable. |
| `red` | one reduction over a K-long axis, no reuse | the rung that prices the **C-18 habit** (§3). |
| `gather` | one random address into a table larger than L2 | isolates *addressing* from arithmetic: the operator is one multiply. |
| `gemm32` | reuse; **no fp32 tensor core exists on sm_75** | the rung where a *missing instruction* is a live hypothesis. |
| `gemm16` | reuse; **tensor core present**, and the framework has a table to choose from | the rung where "a cost model could find it" is a real question rather than a rhetorical one. |
| `scan` | a carried dependent chain, **no contraction** | isolates *sequencing* with nothing else in the operator. |
| `trisolve` | a carried chain **with a dense contraction at every step** | the other end; one of this study's two subjects. |

`scan` and `trisolve` are the pair that earns the ladder its keep: they differ by
**exactly one thing** — a dense product inside the carried chain. The difference
between them is therefore attributable, and neither is confounded by the
gather or the reuse that make `gemm32`/`gemm16` harder to read.

**The pair ran, and it is the ladder's most informative result — because it came
out backwards.** `trisolve`, the rung with the chain *and* the contraction at
every step, is **2.97×**. `scan`, the bare chain with nothing else in it, is
**25.96×**. The deeper, more arithmetically dense rung is **9× closer** to the
hand-written kernel than the simpler one. §7 has the correlate table that this
kills and §10 has what replaces it. It is recorded here, at the point where the
ladder's central bet is stated, because the bet is what the measurement refuted
and a reader should meet the bet before the refutation.

---

## 3. Establishing "best", which is the whole difficulty

`notes/06` §5b: when a study about compilers reports a large gap, the first
hypothesis is not that the compiler is bad, it is that the comparison is
measuring the author. Two comparisons in this study were invalidated exactly
that way, and `notes/26`/`notes/31` isolated the dominant habit — **reducing to
a scalar inside a loop where a wide contraction belongs**, worth ~60x on its own,
hidden by kernel count and exposed by ALU-lane op count.

So "best TinyGrad" here is a **rule, applied everywhere, not just where the
habit was noticed**:

| rung | the idiomatic arm (the rung's TinyGrad point) | the habit spelling (a control, never the result) |
|---|---|---|
| `red` | `(x*w).sum(-1)` — one wide reduction | `for k: acc += (x[:,k]*w[:,k]).sum(-1)` — K−1 dependent rank-1 reductions |
| `scan` | `x.cumsum(-1)` — one op | a Python `for` over the axis, one add per step |
| `gemm32`/`gemm16` | one `matmul` | never a hand-unrolled K loop |
| `trisolve` | the block algebra, each step a `[1,BC]×[BC,BC]` contraction | the unrolled scalar forward substitution (`notes/06` C-15: 4 032 of them, and it passed 6/6) |

Both spellings of `red` and `scan` are **timed on the same card in the same
run** as separate, labelled measurements (§9). That is what was supposed to make
"the idiomatic arm is the good arm" a measurement rather than an assumption —
which is the single most load-bearing claim in this study, and which **the
measurement then refuted on one of the two rungs**.

> **The rule held on `red` and broke on `scan`.** `red`'s habit costs 2.26×
> `[MEASURED]`, so the idiomatic arm is right there. `scan`'s habit costs
> **20× in the other direction** `[MODELLED]`: the framework's idiomatic
> `cumsum` is 22.09 ms and the unrolled Python spelling is 1.114 ms on the same
> data at the same shape. **Applying this study's own rule to TinyGrad's own
> `cumsum` would have made the rung 20× worse.**

So "the habit is fixed everywhere" is the right rule for a **port** and the wrong
rule for a **framework's own lowering**, and the distinction is not a subtlety
— it is 20×. The rule is really: *the habit is expensive relative to a good
alternative*, and the whole job is establishing what the good alternative is for
each operator, per shape. That is why the controls are in the same run and why
they are rung-specific: `red` says use the idiomatic arm, `gemm32` says also
turn on the knob, and `scan` says **do not use `cumsum` at this shape**. §7 has
the mechanism for the last one, in the framework's own source.

The constraint that the TinyGrad side uses no `custom_kernel`, no pasted CUDA
and no compiled-alongside `.cu` is **enforced by an AST check**, not by a
comment: `verify.py:check_purity` parses `rungs.py` and fails on any reference
to `custom_kernel`, the driver API, `ctypes`, or a foreign framework.

---

## 4. The four-way classification, and the falsifier for each rung

The task asks for each rung to fall into (i) cannot express, (ii) expressible
and a cost model could find it, (iii) not implemented, (iv) does not exist for
this architecture — **and for the evidence that would have distinguished the
other three.** A classification with no falsifier is not a classification.

`axis/classify.py` does **not** assign a class from a ratio. It reads
TinyGrad's own source to decide which classes are even *available*, and records
`file:line` for each claim. `READ-FROM-SOURCE`, all re-derived by
`.venv/bin/python axis/classify.py`:

| framework fact | finding | consequence |
|---|---|---|
| triangular solve / matrix inverse | **absent** — no op, no UOp, no pass in `uop/ops.py`, `mixin/op.py`, `codegen/decomp/op.py`, `schedule/rangeify.py` | the `trisolve` rung's cost is a statement about *expression*, not a missing library call. |
| loop construct | **absent** — no `WHILE`/`ASSIGN` op | a recurrence must be unrolled in the source. |
| `UNROLL` | `codegen/__init__.py:45,71` — applies to `AxisType.UNROLL`/`UPCAST` under a `RANGE`, and `notes/06` C-16 records the cap at 32 | a recurrence longer than 32 cannot be unrolled even in principle. |
| `cumsum` | **exists** — `mixin/op.py:755` `_cumalu`, `:761` `_split_cumalu` | class (i) is **closed for `scan` by inspection**: the framework can write a scan down. What is open is whether its lowering emits one kernel or N. |

### The per-rung frame

| rung | open classes | closed, and why | the experiment that separates the rest |
|---|---|---|---|
| `ew` | ii, iii | i: elementwise is the native form. iv: it is a load, a multiply, an add, a store. | **the memory floor**, measured on this card in this run (`axis_fill`/`axis_copy`). At the floor, no gap and the rung is the control. |
| `red` | ii, iii | i: `sum(-1)` is first-class. iv: a reduction is FADD at any width. | **the (`ew`,`red`) pair** — same traffic class, one reduction added. Plus the `tg_naive` control, which prices the C-18 habit in the same run. |
| `gemm32` | **iv**, ii, iii | i: `matmul` exists. | **SASS instruction-class census of both arms** + the `ALLOW_TF32` knob. If both emit FFMA and no HMMA, class (iv) is settled by the ISA, not by the compiler. This is a property of the binary and needs no profiler. |
| `gemm16` | ii, iii | i: fp16 `matmul` exists. **iv excluded by prior, independent measurement**: `axis/sass_census.py` finds **80 HMMA** in the reference (below), so the hardware is present whatever the timings say. | **the knob sweep** — `TC`, `TC_SELECT`, `TC_OPT`, `BEAM`. If *any* setting in the framework's own search space reaches the reference, the capability is implemented and only the default schedule is bad: **(ii)**. If the whole space is flat and far away, the tensor-core schedule is not implemented: **(iii)**. This is the only rung where (ii) and (iii) are cleanly separable, which is why it is on the ladder. |
| `gather` | ii, iii | i: indexing by an index tensor is core syntax. iv: a gather is LDG. | **table size vs measured L2** (a validity gate — if it fits, the rung is silently a different operator), then the (`ew`,`gather`) pair. |
| `scan` | ii, iii | **i closed by inspection** (`cumsum` exists). iv: FADD and a shared-memory prefix. | ~~**the kernel count of the framework's own `cumsum`**, on the device, against the reference's one kernel.~~ **RUN — and the prediction was wrong in an informative way: the count is 3 against 1, so it is *not* a launch-count problem.** The discriminator had to be the *strategy*, found by reading the generated program (§7), and the named experiment that got there is the width sweep across `mixin/op.py`'s `SPLIT*2 = 512` threshold (§9). Recorded because a named experiment that fails is worth more than one that is quietly not run. |
| `trisolve` | ii, iii | **i partially closed, precisely**: the *solve* has no representation, but the *computation* is expressible and `verify.py` checks it against `np.linalg.inv`. iv: FADD/FFMA on 16×16 blocks. | **the (`scan`,`trisolve`) pair** — a chain, then the same chain with a contraction. **RUN, and it refuted the correlate the ladder was built around**: the deeper, more contraction-heavy rung came back **9× closer**, not further. |

**The `gemm16` row is the one that earns the whole classification exercise.** It
is the only rung where the hardware is *known* to be present from a measurement
taken before the timing, and the only one where the framework's own knobs span
enough of the space to say "a cost model could have found this" or could not.

**Two of the five rows above name an experiment that has now been run, and they
did not both come out the way the frame predicted.** That is the frame working,
not failing: naming the discriminator *before* the timing is what makes a
surprise legible. `trisolve`'s pair was supposed to decompose the falloff into
"sequencing" and "sequencing with contraction", and instead it inverted the
ordering; `scan`'s named discriminator was the kernel count, and the count turned
out to be unremarkable. The two surprises are the substance of §7 and §10.

---

## 5. The gate, and the defects it caught

`axis/verify.py` checks every rung's TinyGrad arm against an **independent
float64 transcription** (`np.linalg.inv` for the triangular inverse, deliberately
the slowest possible method, sharing no code path with either arm). It runs on
this host, on CPU, with no network, before any push.

`[MEASURED]` — all 7 rungs pass. The gate, and the on-card oracle, caught
real defects while this was being built. They are listed because the *pattern*
is the finding, not the individual bugs:

| # | defect | why nothing else caught it |
|---|---|---|
| 1 | the row-select mask was on the last axis, so it selected a **column** | the result was upper- rather than lower-triangular: same shape, same dtype. `notes/06` C-20 records a transposed-diagonal bug that survived a shape check the same way. |
| 2 | the identity term was dropped from the substitution | every row past the diagonal became 0; still lower-triangular. |
| 3 | the block-Schur sum started at `k = j+1` instead of `k = j` | dropped the leading term; the error grew with block distance, so it was **invisible at NC=2 and appeared at NC=4**. **The identical bug was in the CUDA reference at the same time** — which is the useful part: the defect is in the recurrence *as usually written down*, not in either implementation. |
| 4 | a `tril` mask was applied to the off-diagonal **blocks** | a block below the diagonal is not triangular *as a block*; its row 0 is strictly lower in the full matrix and is generally nonzero. Masking deleted it. |
| 5 | the scan's phase 3 re-read a register array that phase 1 had **overwritten** | see below — and this one is the sharpest lesson in the note. |
| 6 | four warps staged their fragments into one shared buffer | see below. |
| 7 | the TinyGrad arm was handed **numpy** during timing, putting an H2D copy inside the graph | see below. |
| 8 | `refkernels.cu`'s scan carried `run` as a **loop-local** | see below — a compile, a SASS census, a shape check **and a correct Python transcription** all passed. |

**The numbers inside the defect tables below are the numbers the runs that
*found* those defects produced, and they are not the ladder.** Keeping them is
deliberate — a defect record with the measurement that caught it is evidence, and
re-deriving it against a later build would be a different measurement. The
current ladder is §7, from one run.
| 9 | `refkernels.cu`'s triangular solve tested `c > i` where the row-wise recursion needs `c < i` | the kernel produced an upper-triangular `X` and so **never wrote a strictly-lower entry at all**; the signature was `ref = 0.0` exactly where a value was wanted. |
| 10 | the width sweep's normalisation divided by `N` while the sweep held `ROWS × N` **constant**, so a "nanoseconds per element" column fell as `1/N` by construction and a 3.6× "discontinuity" appeared at the threshold | the column's units depended on a quantity the experiment was holding fixed, so it moved in the direction the prediction pointed. A **normalisation is a claim about what is fixed**, and this one lied exactly where it was being looked at. |
| 11 | `_ref_args` hardcoded **one** argument order for a rung that has **two** kernels: `axis_scan_v{PT}` is `(x, y, R, N)` and `axis_scan_scalar` is `(y, x, R, N)` — same arity, same types, opposite order | the harness's own **type-pattern check passed throughout**, because `(float*, float*, int, int)` and `(float*, float*, int, int)` are the same pattern either way. A width check that passes is precisely what makes an order bug invisible. Found by the width sweep, which returned INVALID for every `N ≤ 256` and OK for every vectorised point. |
| 12 | a "fix" for defect 10 made the selector return the **fastest-looking** entry point, `axis_scan_v1`, for every `N` that was a multiple of 1024 | with one block per row and **no grid-stride loop**, a block covers exactly `1024 × PT` elements, so at N=4096 the PT=1 kernel writes 1024 of 4096 elements and reports **0.047 ms against the correct kernel's 0.219 ms — faster because it was doing a quarter of the work**. A kernel correct for a smaller span is not a faster kernel for a larger one; it is a wrong one. |

Defects 1–5 were caught by gates. Defects 6–7 were caught by the on-card oracle
comparison, and they are the two that matter most, because they are the two an
instrument of this kind exists to catch. Defects 8–9 were also caught on the
card, and they are the reason §5 now has a second half: **they were in the
reference, and the reference had no gate.** Defect 10 was caught by the falsifier
**disagreeing with the thing it was testing** — the sweep printed the reference's
own smoothness beside the framework's specifically so that a step in one and not
the other would be visible, and it was.

**Twelve defects, and the count is not the point. The point is that not one of them
was found by a measurement** — they were found by gates, by an oracle, or by
reading a generated program. Every gate in this file is a *correctness* gate, and
every one of the ten is a defect that a correct answer would have passed.

### Defect 5, in full, because it is the one to remember

`axis_scan` gives each thread a **contiguous run** of `per_thread` float4s and
scans it serially. Phase 1 stored the running inclusive prefix in a register
array `v`; phase 3 re-read `v` to write the output. But `v` is **one float4
reused by every iteration** — so phase 3 saw only the *last* iteration's
prefixes, and every earlier element of the thread's run was written with the
wrong offset. Relative error **0.297** at `N=4096`.

The part that matters is how it survived. **The smoke pass did not catch it,
because the smoke shape is `N=64` and `N=64` takes a different code path** — the
scalar fallback, which was correct. The same kernel was right at the shape the
smoke test used and wrong at the shape the ladder used, *in the same run, on the
same card, through the same instrument.*

So: **a mechanism check that exercises a different path than the measurement
does not check the measurement.** The smoke pass's job is to prove the pipeline
runs, and it did; it was never a correctness check at the bench shape, and
treating it as one would have been exactly the "a passing gate is not
performance evidence" mistake in a new form — here, *a passing gate is not
correctness evidence either*. Phase 3 now re-reads the raw values, which are
still live, and the corrected logic was checked against `np.cumsum` by
transcribing the kernel's index arithmetic into Python before it was pushed.

The generalisation, and the reason §5 is in this note rather than in a log: **on
this operator, "the shape is right" carried no information at all.** Every one
of these would have produced a confident ratio against the hand-written side.

The corresponding statement about the *hand-written* side is that it too is
written in the form a competent practitioner uses, not the easiest form — no
address recomputed from coordinates where an incremental advance exists (C-14),
no rank-1 chain where a contraction belongs (C-18), and the fp32 GEMM is a real
128×128 tiled kernel with 8×8 register blocking rather than the deliberately
worse one-accumulator kernel that TinyGrad's `NOOPT` knob emits. Using `NOOPT`'s
output as the reference would be a strawman, and §6 shows what it cost to get
the reference right.

**A second class of defect, found only on the card, and worth its own
paragraph.** Three of the first four T4 runs died with **CUDA error 716,
misaligned address**, on kernels whose arithmetic was correct and which
compiled clean. All three are the same mistake: *a vector type needs the base
and every stride to be aligned, and only the strides were checked.*

| where | what | why the local gate missed it |
|---|---|---|
| `axis_gemm_f16` shared tiles | written with 16-byte `float4` stores, but `__shared__ __half As[72][24]` has natural alignment **2 bytes**. The column padding made the offsets multiples of 16; the **base** still decided, and it did not. | `nvcc` compiles it without complaint and `cuobjdump` reports no illegal instruction. A compile is not an alignment proof. |
| `axis_gemm_f16` epilogue | wrote **four halves** with a `float4`, which is a **16-byte** store. Four halves is 8 bytes, so the store was twice the data and misaligned for every `c` that was not a multiple of 8. It is now `uint4` — eight halves, 16 bytes — built from four `half2` pairs, with a `N % 8` guard. | **a type mismatch that is also an alignment bug, in one.** The accumulator is `float` and the output is `__half`, and the narrowing happened at the store; the store width was written as if the accumulator were still 4 bytes wide. |
| `axis_scan` scalar fallback | with `N=64` and 256 threads it read 192 elements past each row, because the `threadIdx.x < N` guard was missing. | the **bench** shape is `N=4096`, larger than the block, so the guard never fires there. A kernel that is only correct at the shape it was measured at. |

The first of these poisoned the CUDA context, so the three rungs **after**
`gemm16` in the ladder also reported 716 — four rungs failing from one kernel's
base alignment, and the error message pointing at none of them. That is worth
stating plainly because it is a property of GPU programming that the CPU gate
cannot see and that no SASS census detects: **a `cuobjdump` census of a
misaligned kernel is identical to a census of a correct one.** The only thing
that found it was running it, and the only thing that made running it cheap was
the `--smoke` pass, which exists precisely so that a mechanism failure is found
in a minute rather than after a queue.

Fixes: `__align__(16)` on the shared arrays, a guard rather than an assertion on
the store, and a bounds check on the fallback. The general rule, and the reason
the fallback existed at all: **a kernel that only works at the shape it was
measured at is a measurement of that shape.**

**Two more defects, and the second is the most important in this note.** Both
were found by the same mechanism — the oracle comparison — and neither would
have been found by any instrument this repository had previously used.

| # | defect | how it presented | why it is instructive |
|---|---|---|---|
| 5 | the fp16 GEMM's epilogue staged all **four warps' fragments into one shared buffer**. The warps are at four different 32×32 quadrants of the 64×64 tile, so each warp's `store_matrix_sync` overwrote the others and three quarters of the output was another quadrant's numbers. | relative error **0.956** — a large fraction wrong, which says "wrong" and not "which part" | A *shared* staging buffer is only correct when all threads in the block share the same coordinates. `__syncwarp()` was the right primitive here and `__syncthreads()` would not have helped. |
| 6 | **the TinyGrad arm was being handed numpy inputs during the timing loop**, so the `Tensor(...)` each builder constructed put a **host-to-device copy inside the graph** — three copies of 67 MB for `ew` at n=16.7 M. | `ew` read **44.1 ms** against a **1.08 ms** reference, a 40× "gap" on an operator that moves 268 MB | This is the worst one, because it is a **measurement bug that flatters the framework in the direction the study wants to find**. It would have been reported as "TinyGrad is 40× slow on elementwise". The reference at 1.08 ms for 268 MB is **96% of the measured copy floor** (239.4 GB/s), so the reference was right and the *framework's* number was the artefact. |

Defect 6 is worth isolating as a rule, because it is the same shape as the
study's founding error: **the thing being measured changed and nothing about the
measurement said so.** Both arms were timed with the same instrument, on the
same buffers, in the same run — and the ratio was still nonsense, because one
arm's graph contained work the other arm's did not.

The fix is one line of discipline (`as_tensor`: wrap numpy, pass a Tensor
through) and one structural change — the builders are handed the **device**
tensors while the oracle keeps the **numpy**, so all three views of the same
bytes stay available and the graph contains kernels and only kernels.

**The general form, and it is the note's central methodological claim: a ratio
is only as meaningful as the difference in what the two arms are asked to do,
and that difference is invisible to the instrument unless something independent
— an oracle, a floor, a shape the kernel was not written for — is looking at
the answer.**

### The half of the comparison that had no gate

Everything above gates the **TinyGrad** arm. For most of this study the
**reference** arm had no host gate at all: `refkernels.cu` was checked by a
clean compile, a `cuobjdump` census, and then run on the card. Two of the twelve
defects were in it, and they are defects 8 and 9.

| # | what the reference did | on-card signature | what had already failed to catch it |
|---|---|---|---|
| 8 | `SCAN_BODY` declared `run` **inside** the per-`float4` loop, so every `float4` after the first was written as `offset + its own four elements` and the running total of the previous ones was dropped | exact at thread-relative positions **0–3**, wrong at **4–15**; **56.2%** of elements off; `ref 69.041` where `want 85.875` | a **clean sm_75 compile**; a `cuobjdump` census (identical for a correct kernel and a wrong one); a shape check; and — the sharp one — **a correct Python transcription of the kernel's index arithmetic, checked against `np.cumsum`, passing at 2.2e-07** |
| 9 | the diagonal-block substitution tested `c > i`, i.e. swept the recursion **column-major** | `ref = 0.0` where `want = 0.5505707`, at `[4135, 10, 8]`; 27.0% of elements bad | a clean compile; a SASS census; a shape check; the float64 oracle on the *TinyGrad* side, which was correct throughout |

**Defect 8 is the important one, and the reason is the *transcription*, not the
kernel.** A Python transcription of that kernel is a five-line function, it
transcribes exactly, and it passes. It passed **because it was written from the
intended algorithm** — so it contained the fix, silently, and agreed with
`np.cumsum` while the kernel did not. A transcription of an *algorithm* cannot
catch a defect in an *execution*; it reproduces the algorithm.

Defect 9's signature is worth keeping for its own sake: the error was a **zero,
not a small number**, because the branch never fired and the kernel simply never
wrote the strictly-lower triangle. A defect that produces an exactly-zero error
at the worst point is a defect that skipped work. A tolerance-based check with a
loose bound can hide that; a check that *reports where the worst point is and
what the two values were* cannot.

**The fix is `axis/simref.py`: a host-side emulator of `refkernels.cu`.** Not a
second implementation of the operators — a **phase-structured mechanical
transcription of the kernel**, with the properties that make it a check rather
than a second opinion:

- **shared memory goes through a buffer that only becomes visible at a barrier**,
  so `__syncthreads()` placement is load-bearing. Delete a barrier in the `.cu`,
  delete the matching `sync()`, and the emulator's answer changes — which is the
  point.
- **shuffles are evaluated across a whole warp at once.** A per-thread
  transcription cannot see a shuffle at all. The first version of the emulator
  added each thread's *own* value where the `.cu` adds its *partner's*, and
  returned a self-consistent wrong answer — the same failure mode as defect 8,
  one level up, committed by the very tool written to catch it.
- **every 16-byte access checks its flat index's alignment and names the defect
  if it fails**, which is the check that would have turned all three CUDA-error-
  716 faults (§5, above) into host-side errors.
- **every scan entry point is checked, each on a shape its own precondition
  admits.** `refkernels.cu` exports `axis_scan_scalar` and `axis_scan_v{1,2,4,8,16}`;
  `harness._scan_symbol` picks one by shape, so a run at `N=4096` exercises
  `axis_scan_v4` alone and says nothing about the other five. When this was
  first added it checked `SMALL_SHAPES["scan"]`, whose `N=64` routes to the
  **scalar** path — so the vectorised path, the one with the defect and the one
  the axis actually measures, was **never executed and the gate was green over
  it**. That is defect 5's lesson repeating one level up: *a mechanism check
  that exercises a different path than the measurement does not check the
  measurement*, and this time the mechanism check **was** the correctness gate.

**And the emulator's own falsifier**, because a gate that has never been observed
to fail is not known to discriminate. `verify.py` step 5 re-injects the
historical defects and requires the oracles to *reject* them, signature included:

```
[CAUGHT] scan: `run` reset to loop-local                 relerr 1.106e-01
         row 0: exact at 0-3 = True, wrong at 4-15 = True, 75.0% of elements off
[CAUGHT] trisolve: `c > i` (upper-triangular X)          relerr 9.599e-01
         worst at (6, 15, 2): got 0.000000 want 0.420140
[CAUGHT] trisolve: block-Schur sum from k = j+1          relerr 1.000e+00
         worst at (4, 56, 3): got -0.000000 want -0.437686
[CAUGHT] scan: drop the warp_tot offset term              relerr 8.373e-01
         row 0: exact at 0-3 = True, wrong at 4-15 = False, 87.5% of elements off
```

`[MEASURED]`, on this host, in ~15 s, with no GPU and no quota. The trisolve
mutant's signature — `got 0.000000` where a value was wanted — is the on-card
one, at a different index, which is what a faithful transcription of a defect
should produce. A mutant that still agrees fails the gate, because an
undetectable defect class means a gate that is not measuring what it says.

The claim being made is narrow and worth stating exactly: **this gate would have
caught defects 8 and 9 before either cost a GPU session.** It is not a claim that
the emulator models the GPU — it does not model occupancy, scheduling, register
pressure or memory coalescing, and it is not trying to. It is a claim about
*structure*, and structure is where all twelve defects were.

---

## 6. The reference kernels, and the checks that they are not strawmen

`axis/refkernels.cu`, compiled for sm_75 by the card's nvcc. A hand-written
reference that is weak makes the axis a measurement of the author, so two gates
run **on this host, with no GPU at all**:

**Gate 1 — instruction classes** (`axis/sass_census.py`, over a `cuobjdump -sass`
dump made in the repo's existing `archlab/nvcc:12.9` container):

```
== the two gate questions ==
  fp16 GEMM HMMA count      : 80   PASS -- tensor cores engaged
  fp32 GEMM FFMA=512 HMMA=0   PASS -- FFMA as expected for fp32 on sm_75
  sm_75-illegal instructions: 0
```

`[MEASURED]`. The fp16 rung really is a tensor-core rung, and the fp32 rung
really is FFMA-only on this ISA — which is what makes class (iv) a *hardware*
question for `gemm32` rather than a compiler one. And nothing in the cubin is
sm_75-illegal, which is `minecu_ref/build.sh`'s portability gate applied to new
code.

**Gate 2 — the SASS the author produced, and one correction.** The first fp16
GEMM came out at **4 096 SASS instructions** against the fp32 kernel's 976, with
16 B of spill. The cause was mine, not the hardware's: the shared-memory layout
was `[k][row]`, which forced eight separate `__ushort_as_half` extractions per
thread per k-step. Re-laid-out as `[row][k]` — which is also what
`wmma::row_major` wants for `matrix_a` — the loads became one 16-byte store each
and the kernel came out at **776 instructions, 80 HMMA, zero spills**
(`[MEASURED]`). Had the axis been built on the first version it would have been
measuring a layout mistake of mine and calling it a framework property.

The register and spill report from the same build (`ptxas -v`, `[MEASURED]`):
**every reference kernel compiles with 0 bytes of spill except `axis_gemm_f16`,
which spills 4 bytes** — the cost of the bounds guards its tile loads need in
order to be correct at shapes that are not multiples of 64, which is the price
of not being a kernel that only works at the shape it was measured at. Register
use: `axis_gemm_f32` 110, `axis_gemm_f16` 80, `axis_scan` 79, `axis_trisolve` 41
with 35 456 B of shared memory.

**Gate 3 — the reference's own behaviour, on this host** (`axis/simref.py`, run
by `verify.py`). A census says which *instructions* a kernel contains; it says
nothing about whether they are arranged correctly, which is exactly what defects
8 and 9 were. Gate 3 executes the reference's structure on the CPU and compares
it to the same float64 oracles the TinyGrad arm faces, so **both arms of the
comparison are now checked before the push, not one.** It is a genuinely third
gate because gates 1 and 2 answer different questions — gate 1 asks *is this the
right instruction class*, gate 2 asks *did the compiler like it* — and neither
can see a wrong index. §5 has the mechanism and its own falsifier.

---

## 7. The axis, and where it falls off

`.venv/bin/python axis/analysis.py out/axis/<stamp>/axis_report.json` prints the
table below, the controls, the designed contrasts, and the falloff correlates.
The reproduce-everything invocation is in the header.

### The measured ladder, `[MEASURED]`, one card, one run, one instrument

Device time, cuEvent on both sides; the cubin and the framework's replayed
compiled program. Floor and device properties measured in the same run.
**7/7 rungs OK**, version 19 build, `out/axis/20260929-200921/axis_report.json`.
Tesla T4, cc 7.5, 40 SMs, 4.2 MB L2, **measured** 1.590 GHz, 15 360 MiB, driver
580.159.04. Floor: fill **232.4 GB/s** (write only), copy **239.6 GB/s** (read
+ write), at n = 67 108 864.

| rung | demands | tg dev (ms) | tg e2e (ms) | ref (ms) | **tg/ref** | tg GB/s | % floor | kernels |
|---|---|---|---|---|---|---|---|---|
| `ew` | nothing (control) | 1.071 | 2.630 | 1.087 | **0.99×** | 251 | 105% | 1 |
| `gather` | one random address | 3.795 | 8.713 | 3.944 | **0.96×** | 31 | 13% | 1 |
| `red` | one reduction, no reuse | 1.012 | 3.235 | 0.527 | **1.92×** | 133 | 55% | 1 |
| `gemm16` | reuse, tensor core present | 2.607 | 7.849 | 1.415 | **1.84×** | — | — | 1 |
| `gemm32` | reuse, no tensor core on this ISA | 4.161 | 10.146 | 1.549 | **2.69×** | — | — | 1 |
| `trisolve` | a carried chain **with** a dense contraction | 73.69 | 2251.8 | 24.81 | **2.97×** | — | — | 585 |
| `scan` | a carried chain, no contraction | 21.46 | 32.57 | 0.827 | **25.96×** | 6 | 3% | 3 |

The `scan` ratio is **a representative value from a band, not a peak**: §9's width
sweep measures **22.0× – 35.8× for every `N ≥ 256`**, and the shape of that band is
set by the *reference's* efficiency curve, not the framework's — the framework's own
time is flat to 1.03% across a 32× range of `N`. And the `3%` of floor beside it is
not incidental: the window re-reads hit cache, so the framework's cost is
arithmetic rather than bandwidth, and **the lowest `%floor` and the highest ratio on
this table are the same fact seen twice.**

**A caveat on the `% floor` column, which is `axis_copy` and is therefore a
read-AND-write floor.** `refkernels.cu`'s own header states the rule this
violates if ignored — *a one-direction row must not be divided by a
two-direction one* — and the rule applies to the framework side too:

| rung | DRAM traffic, one element | is `% floor` meaningful? |
|---|---|---|
| `scan` | 1 read : 1 write | **yes** — the only genuinely copy-shaped rung on the ladder |
| `ew` | 3 reads : 1 write | **no** — >100% is expected, a read-only path beats a copy |
| `red` | 2 reads : 1 element per row | **no** — read-dominant; indicative only |
| `gather` | 2 sequential : 1 random, table 16 MB > 4.2 MB L2 | **no** — the table's effective traffic is not modelled by `bytes_moved` |

So `scan`'s **2% of floor** is the only percentage on this table that means what
it appears to mean, and it is the one that matters, because `scan` is the rung
with the largest gap. The others are printed because the byte counts are
derivable and the reader should see them, not because they are comparable to
each other.

**The `ew` row is the one that makes the rest readable.** 254 GB/s against a
measured copy floor of 239.3 GB/s — the framework is *above* the floor, because
`axis_ew` reads through the read-only path and the floor is a plain copy. The
point stands: **at the bandwidth end of the ladder TinyGrad is not behind a
hand-written kernel, and it is not ahead of one either. It is level.** A ladder
whose control rung showed a gap would be measuring the port, and this one does
not.

#### Both GEMM references are ~20% of their own roofline, and that has to be said

| rung | arm | ms | GFLOP/s | % of peak | peak, and where the number comes from |
|---|---|---|---|---|---|
| `gemm32` | ref | 1.549 | 1 387 | **17.0%** | 8.14 TFLOP/s = 40 SMs × 128 FLOP/cycle (FFMA) |
| `gemm32` | tg | 4.161 | 516 | **6.3%** | same |
| `gemm16` | ref | 1.415 | 12 145 | **18.6%** | 65.13 TFLOP/s = 40 SMs × 1024 FLOP/cycle (HMMA, fp32 accum) |
| `gemm16` | tg | 2.607 | 6 591 | **10.1%** | same |

The clock is **measured** (`cuDeviceGetAttribute`); the FLOP/cycle/SM figures are
architecture. Both peaks reproduce the T4's documented 8.1 and 65 TFLOP/s, which
is the check that the constants are the right ones for this part.

**This weakens the two GEMM ratios and the note says so.** A ratio against a
reference that is itself a fifth of the way to its own roofline is *partly* a
statement about the reference. `gemm32`'s 2.69× is a real gap and a
weaker-than-it-looks measurement at the same time: a better fp32 kernel —
vectorised shared-memory reads, fewer barriers per k-tile — would move the
denominator and shrink the ratio. The honest reading of the two GEMM rows is
therefore **"the framework is behind, and by less than the raw ratio suggests"**,
because the numerator beat a reference that had room to improve. `scan` and
`trisolve` have no such caveat: their references are at 68% and 4% of a
*measured* bandwidth floor in the direction that matters, and the `scan`
reference is 67% of the copy floor.

### Where the falloff is, and what it is correlated with

Ordering by ratio: `ew` 0.99, `gather` 0.96, `gemm16` 1.84, `red` 1.92,
`trisolve` 2.97, `gemm32` 2.69, `scan` 25.96.

`analysis.py` computes the three candidate correlates over all seven rungs and
**all three come out weak or negative**:

| correlate | r over n=7 | `analysis.py`'s verdict | what that means |
|---|---|---|---|
| **sequential depth** (carried dependent steps per element) | **+0.369** | weak association | the correlate the ladder was built around, and it is the *weakest* signal here |
| **contraction present** (a dense product inside the operator) | **−0.290** | no detectable association | the previous reading of this note — "the falloff tracks contraction" — is **refuted by the data that completed it** |
| **reuse** (each input byte read more than once) | **−0.290** | no detectable association | still collinear with contraction; not separated |

**This overturns the note's own earlier conclusion, and the correction is worth
more than the conclusion was.** With five rungs and the two sequential rungs
INVALID, the table said contraction was supported. With all seven measured, the
rung that is far off is `scan` — which has **no contraction** — and the rung
that *does* contract heavily, `trisolve`, is 2.97×, i.e. **9× closer than the
pure chain**. Contraction therefore does not explain the falloff; it very nearly
runs the other way.

Four things have to be said about that, and the first is the actual finding:

- **The structural correlates do not explain this ladder, and the thing that
  does is a per-operator property of the lowering.** `scan`'s 22–36× is not a
  consequence of "the operator has a carried chain"; it is a consequence of
  `Tensor.cumsum` being implemented as a hand-placed two-stage windowed
  re-summation. That is an **operator-specific** fact, and no correlate computed
  over *operator classes* can see it, because the class "carried chain" contains
  both a well-lowered and a badly-lowered instance. **A ladder over operator
  classes cannot find this class of explanation; only reading the generated
  program can.** The next subsection is that reading, and it cost no GPU quota.
- A correlation over seven points with three candidates, two of them collinear,
  and no held-out prediction is a **suggestion, not a result** — and
  `analysis.py` prints its own sample size for exactly this reason.
- "Contraction" and "reuse" remain **unseparated** by this ladder, and saying
  otherwise would be the `notes/06` C-16 rationalisation. A rung that reduces
  *without* reuse, or that gathers *with* reuse, would separate them.
- The one thing the ladder *does* establish cleanly is the negative result in
  §7's contrasts: the memory path and the addressing are **not** where the gap
  is. `ew` and `gather` are both level against hand-written kernels.

### Why the `scan` rung is the largest gap: the lowering, read on the host

A ratio says *how much*; it does not say *why*, and the four-way classification
cannot be closed without knowing which of the two surviving classes — (ii) *a
cost model could find it* and (iii) *not implemented* — a rung is in. That
question is about the generated program, not the API, so
`.venv/bin/python axis/lowering.py scan` captures it. **This costs no GPU
quota**, and on this study the two most informative facts about the largest gap
on the ladder both turned out to be readable from source and neither readable
from a timing.

`[MEASURED]`, tinygrad 0.14.0, this host, the bench shape `R=4096, N=4096`:
**3 kernels**, not `N`. That alone is decisive against the strongest reading of
`notes/26` §5c — the framework does **not** emit one kernel per scan step. The
captured bodies show what it emits instead:

```
kernel 0:  3 loops  (Lidx1 < 8192) (Lidx3 < 256) (Ridx0 < 64)     5 buffer refs
kernel 1:  2 loops  (Lidx1 <  512) (Lidx2 <  16)                   17 buffer refs
kernel 2:  3 loops  (Lidx0 <  512) (Lidx1 <  16) (Lidx2 < 64)  float4
```

Kernel 0's inner loop runs 64 times and reads **4** elements each — a
**256-wide window re-summed per output**. And the framework's own source says
what that window is, at `mixin/op.py:761-766`:

```python
def _split_cumalu(self, axis:int, op:Ops) -> Self:
    axis = self._resolve_dim(axis)
    ...
    # TODO: someday the optimizer will find this on its own
    # for now this is a two stage cumsum
    SPLIT = 256
    if not isinstance(s:=self.shape[axis], int) or s <= SPLIT*2: return self._cumalu(axis, op)
```

**So `Tensor.cumsum` is not a scan. It is a hand-placed two-stage windowed
re-summation**, and the comment attached to `SPLIT = 256` says in the framework's
own words that it is a placeholder for something a cost model was supposed to
derive. Stage 1 pools each 256-chunk with a 256-wide window — nominally
**256× the input traffic**, affordable only because the window fits in cache —
stage 2 scans just the **16** chunk totals, and stage 3 broadcasts them back.

This settles the rung's classification at the source, with no timing involved:

| class | `scan` | why |
|---|---|---|
| (i) cannot express | **closed** | `Tensor.cumsum` exists, `mixin/op.py:773`. The op is there. |
| (iv) does not exist for this architecture | **closed** | a prefix sum is architecture-neutral; nothing here is sm_75-specific. |
| (ii) a cost model could find it | **OPEN, and this is where it lands** | the op exists and is *correct*; the **decomposition is hand-placed**, carries the framework's own `TODO: someday the optimizer will find this on its own`, and `SPLIT = 256` is a literal. A cost model that chose the window — or a real scan — would find this. §9's width sweep confirms the class and corrects the mechanism: the window is `min(N, 256)` at *every* N, so there is no threshold to find, only a window width nobody chose. |
| (iii) not implemented | **closed** | the operator is implemented; it is the *strategy* that is not chosen. (ii) and (iii) are distinguished exactly here, and the source settles it. |

**And this is the same failure `notes/06` C-15 already recorded, in a different
operator.** That correction is that the study's KDA port implemented
`fla/ops/kda/naive.py`'s unrolled forward substitution — 4 032 sequentially
dependent scalar reductions — and **passed every test**. Here: a `min(N, 256)`-wide
windowed re-summation, and it passes `np.cumsum` to 1e-7 on the host.
*Correctness gates and correctness of strategy are different properties, and only
one of them is what a green gate certifies.*

**§9's width sweep then corrected the mechanism this section stated.** The claim
above was that the gap comes from the *two-stage* path and its `SPLIT*2 = 512`
threshold. The sweep shows the framework's per-element cost is **flat** across
N = 256 → 1024 — no step — because the *below*-threshold path is also a
`min(N, 256)`-wide re-summation. `SPLIT` governs how the window is composed and
why the kernel count is 1 vs 3; it does not govern how much is re-read. The
classification above is unchanged, the class is unchanged, and the mechanism is
now a measured window width instead of a threshold I asserted and the data
refuted.

The falsifier, if this reading is wrong: if the 22 ms were launch overhead the
ratio would be microseconds across 3 launches; if it were a weak reduction the
reference — same data, same card, one kernel — would be slow too. It is neither,
which is why the gap is attributed to the decomposition rather than to
scheduling. `analysis.py` prints the byte counts so the attribution is checkable
rather than narrated.

### The designed contrasts, and what each one attributes

Each pair differs by exactly one thing, which is what makes a difference
attributable. These are the load-bearing evidence in this section; the
correlations above are not.

- **(`ew`, `red`) — 0.99× vs 1.92×.** Same traffic class, one reduction added.
  The gap is **the reduction**, not the memory path: `ew` is level, and `ew` is
  what establishes the memory path is not the problem. Note this rung is **not**
  bandwidth-limited on either side, which is why the ratio is about the
  reduction's schedule and not about bytes: 134 MB of reads in 0.53 ms is
  255 GB/s, and a read-only-dominant stream is *not* comparable to a copy floor
  (see the caveat below), so the useful statement is the ratio and not the
  percentage.
- **(`ew`, `gather`) — 0.99× vs 0.96×.** Same traffic class, one random address
  into a 16 MB table against a 4.2 MB L2. **No gap at all.** The framework
  handles a missing-cache gather as well as a hand-written kernel, which
  contradicts the stronger reading of `notes/26` §3c (that a gather is a
  *barrier*) and supports the weaker one (that the cost is the index
  arithmetic, and there is little of it here). **This is the study's clearest
  negative result**, and it is negative because a control rung was measured on
  the same card in the same run rather than assumed.
- **(`gemm32`, `gemm16`) — 2.69× vs 1.84×.** The same contraction in two
  dtypes, one with a tensor core on this ISA and one without. The gap **shrinks by
  1.46× when the instruction class exists**, which is the signature of
  a scheduling problem rather than an expressiveness one — the framework can
  reach the tensor cores, and the default schedule does not get there on its own.
  It is also the rung where class (ii) and class (iii) are cleanly separable
  (§4), and the direction points at (ii). **Caveat, and it is not small:** both
  references sit near 20% of their own instruction-class peak, so this pair
  bounds the gap from above and the true gap is smaller than the ratio says.
- **(`scan`, `trisolve`) — 25.96× vs 2.97×, and the pair runs backwards.** A
  carried chain, then the same chain with a dense contraction at every step. The
  *deeper*, more contraction-heavy rung is **9× closer to the hand-written
  kernel than the simpler one.** This is the single most informative pair in the
  study, and what it says is that **sequential depth is not what costs** — a
  claim this ladder was built to test and which it has now refuted.
  What separates the two is the *quality of the lowering*, and the two rungs
  differ in it enormously: `trisolve`'s TinyGrad arm is a genuine two-level
  blocked substitution (right strategy) expressed as **585 kernels** because the
  framework has no loop construct; `scan`'s arm is **3 kernels** — and the
  strategy behind them re-sums a `min(N, 256)`-wide window per output. One
  strategy is right and pays in launch count; the other is one kernel and is
  simply the wrong decomposition.

  There is a real cost hiding in `trisolve` and it is not visible in the ratio:
  its end-to-end time is **2 252 ms against 73.7 ms of device time**, a 31×
  gap. 585 launches, each with its own Python-side dispatch, is where that goes.
  A framework with a persistent-kernel or loop fusion pass would collapse both
  the launch count and the device time. The ladder's device-time ratio does not
  see any of that, which is the reason the note keeps the end-to-end column
  beside the device column instead of discarding it.

`analysis.py` refuses to draw a contrast from a rung whose status is not `OK`,
which is why the status column is on every row.

---

## 8. The two existing subjects, placed

**`minecu_ref` — the DDA raymarcher (`Rung: ew`-class, demand: fusion-and-gather
with almost no arithmetic per element).** It is the low end of the axis: a
gather per step, a long dependent loop, and almost nothing else — the shape of
operator that `notes/19` and `notes/28` measure. It belongs on the axis as the
point furthest from any dense arithmetic, which is what makes the `gemm16` point
at the other end interpretable.

**KDA — the operator against flash-linear-attention (`Rung: trisolve`).** The
caveat the task requires, stated precisely: **fla's own reference does not run
on sm_75 and was never compiled here.** `notes/29` already records that triton
is not installed in this repo, so `fla/ops/kda/chunk_intra.py` could not be
AOT-compiled and **no hand-written SASS row for it exists anywhere in this
study**. What `axis_trisolve` is instead is a **hand-written sibling that
implements the same algorithm and does run on sm_75**: a two-level blocked
inverse at `BC=16` with the `k` sum starting at `j` (§5, defect 3 — the same
correction `notes/06` C-15 makes about the naive port). So the `trisolve` row is
**"TinyGrad vs a hand-written blocked inverse on a T4"**, and it is *not*
"TinyGrad vs flash-linear-attention". Any statement about the original kernel's
speed is `[UNVERIFIED]` and none is made.

What *is* verified is the algorithm, and it is verified twice, independently:
the TinyGrad arm against `np.linalg.inv` in float64, and the CUDA reference
against the same. The blocked form is the one `notes/25` measured off the pinned
source (`fla` @ `79d12e8`), not the `naive.py` form that `notes/06` C-15 shows
was the actual defect in the earlier port.

---

## 9. The controls, and what is still not measured

### The controls, `[MEASURED]`, same card, same run, same instrument

These are what make the axis a statement about the **framework** rather than
about a port. `notes/26` and `notes/31` isolated one habit — reducing to a
scalar inside a loop where a wide contraction belongs — and both notes are
honest that they could not say how much of their own gap it accounted for. Here
that price is a number, on the same card, in the same run, for two of the three
rungs that admit a control.

| rung | idiomatic arm | control | what the control measures |
|---|---|---|---|
| `red` | 0.993 ms — `(x*w).sum(-1)`, **1 kernel** | 2.245 ms — `K-1` dependent rank-1 updates | the C-18 habit costs **2.26×** here, at `R=65536, K=256` |
| `gemm32` | 4.481 ms — fp32, no tensor core | 2.523 ms — `ALLOW_TF32=1` | one shipped knob closes **43.7%** of the framework-side gap: 2.90× → **1.63×** |
| `scan` | 22.09 ms — `.cumsum(-1)`, **3 kernels** | 0.0696 ms at `N=256`, scaled 16× → **1.114 ms** `[MODELLED]` | the control **beats the idiomatic arm** — see below |

**The `red` control is the direct price of the habit, and it is 2.26×, not
~60×.** `notes/26` and `notes/31` report ~60× for the same habit. That is not a
contradiction, and the difference is worth stating exactly: the 60× was measured
against a **port** carrying the habit, and the 2.26× is measured against
TinyGrad's own idiomatic arm, which does not carry it. The habit is expensive
when the alternative is a good expression. Both numbers are true of what they
measured; they answer different questions, and only this one is about the
framework.

**The `gemm32` control is the strongest single piece of evidence in the note for
class (ii) on that rung.** One framework knob that already ships takes `gemm32`
from 2.90× to 1.63× of the hand-written kernel — nearly half the gap closed by
a setting that requires no new capability and no new operator: 2.90× → 1.63×.
That is what "a cost model could find it" predicts and what "not implemented"
does not. The
caveat is unchanged (§7): the reference is at 17% of FFMA peak, so the true gap
is smaller than either ratio.

**And the `scan` control inverts the note's own framing.** The naive spelling — a
Python loop of `N-1` unrolled steps, the C-18 habit in its purest form — comes
out at **1.114 ms scaled, against the idiomatic arm's 22.09 ms**. On this
operator at this shape, the framework's *idiomatic* `cumsum` is **20× slower
than writing the habit by hand.** This is `[MODELLED]`, from a measurement at
`N=256` scaled by 16; the harness caps the naive control at `N=256` because the
unrolled spelling at larger `N` already spent a whole 25-minute session once.

Three things follow, and the third is the one that matters:

1. **"Best TinyGrad" is not the same as "the idiomatic arm."** On `red` and
   `gemm32` the idiomatic arm is right and the habit is expensive. On `scan` the
   idiomatic arm is **20× worse than the habit**, so the ladder's own
   requirement — fix the scalar-reduction habit *everywhere* — would have made
   this rung *worse* had it been applied to the framework's own spelling. The
   requirement is sound as a rule about **ports**. On a rung where the framework
   is the port, following it blindly is a mistake, and this note is the place
   that mistake would otherwise have been made.
2. **This is `notes/06` C-15 for the third time.** A *correct* expression built
   on a bad decomposition — 4 032 dependent scalar reductions, a 256-wide windowed
   re-summation, a 585-kernel blocked substitution — passes every test and loses
   by one to two orders of magnitude. Three times in this study the defect was not
   a bug but a **decomposition**, and no correctness gate in existence can see
   one. Every gate in §5 is a correctness gate.
3. **So the note does not claim TinyGrad is "slow at scans."** It claims
   something narrower and much better supported: **for this operator at this
   shape, the framework's default lowering is worse than the spelling `notes/26`
   tells ports not to write** — a per-operator, per-shape claim with a mechanism
   in the framework's own source, and a falsifiable one (below).

### The falsifier, and it refuted the prediction twice — which is the point of running it

The `scan` account made one sharp quantitative prediction, written down here
**before** the sweep:

> `mixin/op.py:768` reads
> `if not isinstance(s:=self.shape[axis], int) or s <= SPLIT*2: return self._cumalu(axis, op)`
> with `SPLIT = 256`, so the framework switches decomposition at **N = 512** —
> below it, a single pooled reduction; above it, the two-stage 256-wide
> re-summation. **The operator's difficulty does not change at 512.** So the
> framework's cost per element should step up across 512.

**It did not step, and the reason is better than the prediction was.**

`[MEASURED]`, version 23, `ROWS × N` held constant at 2²² elements, so every point
moves the same data and the total time *is* the per-element cost up to one constant
divisor. **All seven points are `OK`** — the reference is correct at every one,
which required two fixes described below.

| N | framework (ms) | reference (ms) | ratio | ref kernel | tg kernels |
|---|---|---|---|---|---|
| 64 | 3.708 | 1.586 | 2.34× | `axis_scan_scalar` | 1 |
| 128 | 3.263 | 0.554 | 5.89× | `axis_scan_scalar` | 1 |
| 256 | 5.377 | 0.217 | 24.77× | `axis_scan_scalar` | 1 |
| 1024 | 5.562 | 0.155 | **35.80×** | `axis_scan_v1` | 3 |
| 2048 | 5.544 | 0.215 | 25.73× | `axis_scan_v2` | 3 |
| 4096 | 5.550 | 0.219 | 25.37× | `axis_scan_v4` | 3 |
| 8192 | 5.544 | 0.252 | 22.02× | `axis_scan_v8` | 3 |

**The framework's time saturates at `N = 256` and is flat forever after** — 5.377,
5.562, 5.544, 5.550, 5.544 ms across a 32× range of N, a **1.03× spread**. There is
no knee at 512, and there is no knee anywhere. The reason is in the source I
quoted: the *below*-threshold path is not a cheap single pass. It is `_cumalu`,
and `_cumalu` **pools with a window of width `N`** after padding the row by `N-1`.
So the below-threshold path re-reads `N` inputs per output and is quadratic in
exactly the same way.

**So the real mechanism is one thing with no threshold in it: `Tensor.cumsum`
re-sums a window of width `min(N, 256)` per output, at every N.** Below `SPLIT*2`
the window is `N`; above it, 256. The `SPLIT` switch changes only how the
256-wide window is *composed* — one pooled reduction, then a scan of the 16 chunk
totals, then a broadcast add — and not how much is re-read. `SPLIT = 256` is a
real structural fact and it is why the framework's kernel count goes 1 → 3, but
**it is not the cause of the gap, and this note said it was.** A knee at
`min(N, 256)` is what the data shows; a discontinuity at `SPLIT*2 = 512` is what
the note predicted. The note was wrong and the experiment was right.

**The N=64 and N=128 rows are the strongest evidence for the corrected account,
and the first reading of them missed it.** The framework is *faster* there —
3.708 and 3.263 ms against 5.4 ms above — and the only reason is that
`min(N, 256)` is `min(64, 256) = 64`: a 4× narrower window, 4× less work. **The
same defect that makes `cumsum` 26–36× slow above 256 makes it fast below 256**,
and a threshold-switch account predicts neither. The saturation at 256 is the
falsifier's positive result, and it is a stronger statement than the prediction
was because it has a *shape* and the prediction had only a location.

**The ratio's shape is the reference's, not the framework's, and saying so is the
second honest correction.** The ratio *peaks* at 35.8× and then falls to 22.0×,
and an earlier version of this note read a monotone rise (2.3× → 37.3×) off the
*previous* sweep — which was partly invalid (§ below). The rise is not the
framework's: its own column is flat to 1.03%. It is the **reference's efficiency
curve, which is U-shaped** — 1.586 ms at N=64 (the scalar kernel: one element per
thread, no `float4`, no warp shuffle), 0.155 ms at its best at N=1024, then back
up to 0.252 ms at N=8192 (longer serial chain per thread as `PT` grows). So the
honest reading of the ratio column is:

> **for every N ≥ 256 the framework is 22–36× behind, and the shape of that curve
> is set by which `axis_scan_*` specialisation the reference happens to use.** The
> framework's own cost is a step function of N: flat, and high, above 256.

That is a *weaker* and *truer* claim than a monotone growth law, and it is what
`analysis.py` prints. It also makes the ladder's single `scan` number honest to
state: **26.0× is a representative value from a band of 22–36×, not a peak.**

**The reconciliation with `%floor`, which is the check that the mechanism is
right.** §7 reports `scan` at **3% of the measured copy floor** while its ratio is
the largest on the ladder. Those are not two facts, they are one: the window
re-reads at these sizes are **cache hits, not DRAM traffic**, so the framework's
cost is arithmetic and not bandwidth. A window-width defect predicts exactly
that — high ALU cost, low DRAM cost — and a bandwidth defect would predict the
opposite. The note should have noticed the pairing two sections earlier.

**Two runs were lost to my own instrumentation, and both are the study's recurring
error.** They are worth recording because the second one nearly shipped a *false
result* rather than a missing one:

| what | symptom | why nothing else caught it |
|---|---|---|
| the sweep's "ns per element" column divided by `N` while the sweep held `ROWS × N` **constant** | a plausible 3.6× "discontinuity" at exactly the threshold the experiment was built to find | the column's units depend on a quantity the experiment was holding fixed, so it moved **in the direction the prediction pointed**. A normalisation is a claim about what is fixed. |
| a "fix" returned `axis_scan_v1` for every N that was a multiple of 1024, reasoning that one `float4` per thread was fastest | at N=4096 a block covers 1024 of the row's 4096 elements and reports **0.047 ms — faster than the correct kernel's 0.219 ms** | a kernel correct for a *smaller span* is not a faster kernel for a larger one; it is a wrong one, and it is faster **in proportion to how much of the answer it skipped**. `PT` is not a performance choice: with one block per row and no grid-stride loop, a block covers exactly `1024 × PT` elements, so `PT = N/1024` is *forced*. |
| one hardcoded `_ref_args` order for a rung whose two kernels declare **opposite** orders | INVALID for every `N ≤ 256`, OK above — the v22 pattern, and the reason those three rows were invalid | `axis_scan_v{PT}` is `(x, y, R, N)` and `axis_scan_scalar` is `(y, x, R, N)`: same arity, same types, opposite order. The harness's own type-pattern check passed throughout. |

**The sweep and the ladder check each other, which is the cheapest validation in
the note.** The ladder measures `scan` at `R=4096, N=4096` and gets a reference
time of **0.827 ms**; the sweep measures `N=4096` at `R=1024` and gets
**0.219 ms**. Same kernel, 4× the rows, and 4 × 0.219 = 0.875 ms against 0.827
measured — **within 6%**, in two separate launches, over a shape relationship
neither run was told about. Two independent measurements of the reference agreeing
to 6% at a shape the sweep did not visit is the strongest evidence available here
that the reference is not quietly shape-dependent, and it cost nothing beyond the
sweep the falsifier already required.

**And what the host gate can and cannot say about all of this, precisely.**
`verify.py` step 4b pins the span arithmetic from both sides — each entry point must
be correct at its own span, must **refuse** twice it, and the selector must name
the entry point the arithmetic implies. It prints the caveat it is entitled to
print: `simref.sim_scan` has an explicit span assertion and the CUDA has **none**,
so the CUDA truncates *silently*. The host gate checks the arithmetic; it does not
get credit for the device's silence, and only the on-card oracle caught that.

## 10. Does this framework have a capability envelope at all?

**Yes — but not the one the ladder was built to find, and that is the finding.**

The question this note set out to answer was whether the gap between TinyGrad
and hand-written CUDA on one T4 is organised by **operator class**: does the
framework have capability boundaries, so that some classes it does well, some it
cannot express, and some it expresses but schedules badly? The structure
supported that reading — no loop op and `UNROLL` capped at 32
(`READ-FROM-SOURCE`, `codegen/__init__.py:45`), no triangular solve and no
inverse anywhere (`READ-FROM-SOURCE`), and a working `cumsum`
(`READ-FROM-SOURCE`, `mixin/op.py:773`) are three genuinely different capability
boundaries, not three degrees of one thing.

**The measured ladder does not support the class-wise reading, and says so
itself.** All three structural correlates come out weak or *negative*
(§7): sequential depth **+0.369**, contraction present **−0.290**, reuse
**−0.290**, over n = 7. The decisive datum is the `scan`/`trisolve` pair, which
was designed to be the test: the rung with **more** sequential depth and **more**
contraction is **9× closer** to the hand-written kernel than the simpler one.
Sequential depth does not order this ladder. Contraction very nearly runs the
other way.

**What orders it instead is whether the operator's lowering is any good, and that
is a per-operator fact, not a per-class one.** The evidence is unusually direct
because both ends of the range are explained:

| | TinyGrad's own lowering | verdict |
|---|---|---|
| `matmul` fp16 | a real tiled TC kernel, 10.1% of HMMA peak, **1.84×** a hand-written one at 18.6% | **good.** A scheduling shortfall of under 2×, and the ceiling is set by a denominator with room to improve. |
| `cumsum` | a `min(N, 256)`-wide windowed re-summation instead of a carried running value, `SPLIT = 256` a hand-placed literal carrying the framework's own `TODO: someday the optimizer will find this on its own` | **bad.** ~256× the nominal work for an operator that needs 1×, and the re-reads hit cache, so the cost is arithmetic and not bandwidth. **22–36×** for every `N ≥ 256`, with the framework's own time **flat to 1.03%** (§9's sweep). |

**The same library ships both.** That is the answer, and it is a more useful one
than a class-wise envelope, for three reasons:

1. **It explains `notes/26` and `notes/31` without a separate theory.** Those
   notes found a ~60× habit — *reducing to a scalar inside a loop where a wide
   contraction belongs* — and could not say whether it was a framework limit or
   a port mistake. It is neither, in the end: it is a **lowering-quality** fact.
   TinyGrad will happily emit the bad spelling (`(x*w).sum(-1)` vs a Python
   unrolled loop) and will happily emit a good one, and the difference is 60×.
   "Best TinyGrad" is therefore a statement about *which lowering you reach for*,
   and the axis's job was to make that measurable rather than assumed — which is
   why the naive controls are timed in the same run as separate labelled
   measurements, and why `ew` and `gather` being **level** matters: it rules out
   a framework-wide handicap and isolates the variation to individual lowerings.
2. **It was falsifiable in one run, the run happened, and the stated mechanism
   was wrong.** The account said the gap comes from `cumsum` switching to a
   two-stage 256-wide re-summation at `N = SPLIT*2 = 512`. §9's width sweep finds
   **no step there at all**: the framework's time *saturates* at `N = 256` and is
   flat above it, because the *below*-threshold path re-sums a window of width
   `N` and is quadratic in the same way. Saturation at 256 is what a
   `min(N, 256)` window predicts; a switch at `SPLIT*2` would have produced a
   knee at 512 instead. **The class survived the falsifier and
   the mechanism did not**, which is exactly what a pre-registered prediction
   exists to produce, and the claim is narrower and better supported afterwards:
   a `min(N, 256)`-wide window at *every* N, with nobody having chosen the width.
   The sweep also produced the finding the single-shape ladder could not, and it is
   not the one the note guessed: **the framework's cost saturates at `N = 256` and
   is flat to 1.03% above it**, and the **gap is 22–36× for every `N ≥ 256`**. A
   single-shape ladder reports 26.0× and calls it the value; the sweep says it is a
   band, and that the band's *shape* belongs to the reference kernel's efficiency
   rather than to the framework.
3. **It moves the interesting question.** "Is TinyGrad slow?" is answered per
   operator and is not a property of the framework. "Which lowerings are good?"
   is a property of the framework, is enumerable, and has a per-operator
   measurement attached. `axis/lowering.py` is the tool for it: it captures what
   the framework *emits* for a given expression on this host, for no GPU quota,
   and it is what turned `scan`'s 26× from a measurement into a diagnosis.

**And the framework's own source agrees, in a way that is worth quoting rather
than paraphrasing.** The `cumsum` lowering is not merely worse than a real scan;
it carries a comment saying it is a placeholder for something a cost model was
supposed to derive. **Class (ii) — "a cost model could find it" — is the
framework's own account of its own largest gap on this ladder.** That is the
strongest form this classification can take: not that I inferred the class from a
timing, but that the maintainers wrote down the same thing next to the constant.

**And the controls sharpen it further, in a way that is uncomfortable for the
idiom this study started from.** The `scan` control (§9) shows the framework's
idiomatic `cumsum` is **20× slower than the naive unrolled spelling** on the same
data at the same shape. So on this operator the advice `notes/26` gives ports —
never reduce to a scalar inside a loop — is **wrong if applied to TinyGrad's own
lowering**, and a study that had followed its own rule here would have made the
rung worse. "Best TinyGrad" is therefore a *measurement* per operator, not a
style, and that is the practical form of this note's conclusion: **the ladder's
job is to tell you which spelling to reach for, and the answer differs by
operator.** `ew` and `gather` say "the idiomatic one, you are already there";
`red` says "the idiomatic one, and the habit costs 2.26×"; `gemm32` says "turn on
the knob, it recovers 44% of the gap"; `scan` says "**do not use `cumsum`**".

**What would change the answer.** A width sweep finding the ratio smooth across
N = 512 (§9) would refute the per-operator account and put the class-wise reading
back on the table. A `minecu` point coming back at 20× would refute the "level at
the memory path and at addressing" claim, and it is the single most likely thing
to change this note's conclusion — `minecu_ref` is a gather-and-fuse operator
with a long dependent loop, much closer to `scan` in shape than `ew` is, and the
axis's low end being level is the result most likely to be an artefact of
choosing `ew` as the control.

**A last word on the instrument, because the note's real subject is the
instrument.** Twelve defects were found while building this, and **not one of them
was found by the measurement**. They were found by gates, by an oracle, by a
falsifier that disagreed with what it was testing, or by reading a generated
program, and the pattern is uniform:

- **two were in the hand-written reference**, and neither was visible to a clean
  compile, a `cuobjdump` census, a shape check, or a *correct* Python
  transcription. One is now caught on the host in ~15 s by `axis/simref.py`,
  whose own falsifier re-injects the defect and requires the oracle to reject it.
- **one put a host-to-device copy inside a timed graph** and inflated `ew` by
  40×, in the direction that flatters the framework. A shape check, an oracle and
  a compile all passed; only noticing that 268 MB took 44 ms found it.
- **one lost an entire controls cell** to a missing `f` on a string — the
  notebook died on a usage error, printed `MISSING`, and shipped a clean-looking
  7/7 ladder beside it. The generator now refuses to build a notebook with an
  uninterpolated placeholder on a command line.
- **three came out of the width sweep**, and all three are worth naming because
  two of them were mine. A normalisation divided by `N` while the sweep held
  `ROWS × N` constant, manufacturing a 3.6× "discontinuity" — caught because the
  sweep printed the reference's own smoothness beside the framework's. A "fix" for
  that made the selector pick a kernel that was **faster because it was wrong**,
  writing a quarter of the answer — caught because the oracle rejected it. And a
  real one: one hardcoded argument order for a rung whose two kernels declare
  opposite orders, invisible to the harness's own type-pattern check, caught
  because the sweep's `INVALID`s fell in exactly the shape the selector bug
  predicted.

**Every one of those is the same shape as `notes/06` §5b: a check that passed,
beside a number that was wrong, on a comparison that looked finished.** That is
what an instrument has to be defended against, and it is why this one is four
gates — and why one of them has been *observed to fail*, which is the only
property that distinguishes a gate from a decoration.
