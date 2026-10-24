---
kind: reference
---

# 23 — good versus naive: how much of the 26× was the port's fault

**The question.** `notes/19` reported that a plain TinyGrad port of `mine.cu`'s
voxel DDA ran 26.2× slower than the hand-written CUDA at batch 1024, and that
47.2× of that was traced to instruction count — 36,600 SASS against 776. This
note re-asks the question with the bar that was originally set and was not met.

**The bar, restated because the whole note turns on it.** "Plain" means *stay
inside the language's normal vocabulary*. It does not mean *write the worst
program the vocabulary admits*. The forbidden things are a `custom_kernel`, a
tile size, `TC_SELECT`, `ALLOW_TF32`, a `Context(...)`, hand-unrolling,
hand-staging, hand-fusion, a device-specific branch, per-thread scratch. The
required things are ordinary algorithmic means: hoist a loop invariant, maintain
an incrementally rather than recomputing it, compute a test once rather than
thirty-two times. **Hoisting is algorithmic and therefore required; hardware
knowledge is forbidden.** `minecu_tg/render.py` recomputed the flat voxel address
from three float coordinates on every one of 32 steps, with three float→int casts
and three clips. That is a worse algorithm than the one the reference
implements, and measuring it measured the port.

**The answer, first.**

1. **A competent port removes most of the gap where the fusion crossover is, and
   nothing where it is not.** At batch 1024 the naive port is 27.2× the
   hand-written render's time; the optimized port is **9.0×**. At batch 2048,
   28.7× becomes **9.3×**. At batches 256 and 512, below the crossover, the
   optimized port is **14% slower** and the gap gets *wider*. `MEASURED`, §3.
2. **The index arithmetic was 3.75× of the port's operations and 1.39× of its
   machine instructions.** 30 graph operations per pixel per step become 8; 74.4
   SASS instructions per pixel per step become 53.5. The two ratios differ
   because the operations removed are cheap ones. `MEASURED`, §2.
3. **The earlier 47.2× was not a codegen ratio at all.** It divided 36,384
   SASS against 776 — but those two numbers cover *different amounts of work*.
   The generated kernel is vectorised at ~16 pixels per thread; the reference
   is one thread per pixel. Normalised per pixel per step, the same comparison
   is **1.96×**, not 47.2×. `MEASURED`, §2.4. This is the single largest
   correction in the note and it is not a matter of opinion.
4. **The structural findings survive, and they are about the compiler.** The
   fusion crossover is at ~540,000 pixels for both ports. The hand-written
   side's 16 unrolled steps are byte-identical at a spacing of 38; every fused
   generated kernel on either side of the ladder has no repeating body at all.
   `MEASURED`, §4. One earlier structural claim does **not** survive
   re-derivation: the `max_steps + 8` launch law is exact only up to
   `max_steps` 8, and is a half-rate ladder above it. §4.1.
5. **So: the port's fault, in one sentence — the naive port's index arithmetic
   was 3.75× its operation count and, by making the graph too big for the
   scheduler's fusion budget, the reason the generated side was 27× rather than
   9× slow; the compiler's remaining 9.0× is 1.41× in instructions per pixel per
   step (53.5 against 38) composed with 1.94× in executed work (32 steps for
   every ray against a measured mean of 16.54), which together account for about
   a third of it and leave the rest unaccounted.**

**Labels.** `MEASURED` — an instrument ran on this host or on this card, and the
artefact is named. `MODELLED` — arithmetic on measured shapes. Every number here
is re-derived by an instrument in `minecu_tg_opt/`; none is transcribed from any
other note, because the earlier work in this repository was audited and found to
have propagated errors that way.

**Runs.** T4 timings: `out/minecu_tg_opt/20260929-151936/`, one Tesla T4, one
session, both ports and the hand-written side. Instruction and register counts:
`out/minecu_tg_opt/opcount-b1024.json`, produced offline by
`minecu_tg_opt/compile_sass.sh` in the `archlab/nvcc:12.9` container, CUDA 12.9
on **both** sides. Kernel counts: `out/minecu_tg_opt/census-*.json`.

---

## 1. What was built, and that all three compute the same thing

`minecu_tg_opt/render.py` is the same computation as `minecu_tg/render.py`, in
the same language, in the same masked-loop formulation, with the same camera
basis (imported, not copied, so the two differ in the DDA and nothing else) and
the same 14-row colour table. It is five rungs of a ladder, each the rung above
plus exactly one rewrite:

| rung | the rewrite | why it is algorithmic |
|---|---|---|
| `naive` | none — `minecu_tg/render.py`'s body | the baseline |
| `int_cursor` | the cursor is an `int`, cast once | the cursor *is* an integer; `kernels.cu:638` says so |
| `flat_gather` | one flat index into the gather, not four | one index instead of four for the same byte |
| `incremental` | the address is maintained with precomputed strides, and the per-axis bounds test is computed once as a steps-to-exit countdown | Amanatides–Woo's state is incremental by definition |
| `lean` | no separate `found` flag — `hit < 0` already means "not hit yet" | one fewer live tensor |

No `custom_kernel`, no tile size, no `TC_SELECT`, no `Context`, no hand-unrolling,
no codegen knob, no device-specific branch. The step loop is still a Python
`for` over `max_steps` with one body. The full argument for each rewrite, and
for what is deliberately *not* done, is in that file's docstring.

### 1.1 All five rungs produce the naive port's observation, bit for bit

`MEASURED` `minecu_tg_opt/verify_opt.py`, this host, CPU, five shapes
(worlds 16/16/32/48/16, pitches 0/−0.2/0/0/0, `max_steps` 32/32/32/32/16) ×
five rungs, against `minecu_tg/render.py`'s own output and against
`minecu_tg/cpu_reference.py`'s independent float64 transcription:

| check | result |
|---|---|
| observation identical to `minecu_tg/render.py`, every rung, every shape | **0 pixels differ**, all 25 |
| observation equal to the float64 transcription, every rung, every shape | **0 pixels differ**, all 25 |
| per-pixel DDA step counts agree with float64 | 1024/1024 at worlds 16 and 32, 1022/1024 at world 48 (float32-vs-float64 at exact ties; the colours are unaffected) |
| the world, against `minecu_ref/bench/dda_model.py` | exact at worlds 16, 32, 48 |

The step-count check is the one that matters most for this note: rewrite 3
replaces a per-step bounds test on the cursor with a per-ray steps-to-exit
counter, and a 100% match on a tie-sensitive discrete quantity is what rules
out the counter being wrong.

### 1.2 Both ports produce the *hand-written side's own bytes*, on the card

`MEASURED` `out/minecu_tg_opt/20260929-151936/obs_equivalence.json`, on the T4,
in the run whose timings §3 reports. `refdrv.cu` links the unmodified vendored
`kernels.cu`, settles the world, and dumps its own voxels, its own 13-slot camera
basis and its own uint8 observation. Both ports then render that exact state on
that card.

```
naive             0 /  4194304 pixels differ   BYTE-IDENTICAL
lean              0 /  4194304 pixels differ   BYTE-IDENTICAL
obs sha256 reference 21e13f67fa003c488a7d5c9868944850d93e63f4e19f80e93b53a42432cdbfa7
```

4.19 million pixels, byte-identical, same card, same run, same state. That is
what makes every ratio in §3 a statement about code generation rather than
about two different computations.

This is a stronger check than the CPU one and it is on the card because that is
where the CUDA backend lowers the graph; the host only ever exercises the CPU
backend, which is a different code path with a different arithmetic.

### 1.3 Both timed runs did the same work

`MEASURED` same run, `battery_opt.py`'s own DDA probe, batch 2048:

| port | mean DDA steps/pixel | at the 32 cap | first voxel | gate |
|---|---|---|---|---|
| naive | 16.540 | 41.9% | 0.0% | PASS |
| lean | 16.540 | 41.9% | 0.0% | PASS |

Identical, as they must be — the two ports trace the same rays. The gate is the
reference harness's own rule (`mean ≥ max(2.0, 0.10 × max_steps) = 3.2`, and
≤ 20% of pixels hitting on their first voxel test), carried over because a
render whose rays all die on their first voxel test is an observation store and
a comparison against one is a comparison of two stubs.

---

## 2. What one DDA step costs, in operations and in instructions

### 2.1 The instrument, and why it is a difference and not a count

`MEASURED` `minecu_tg_opt/opcount.py`, offline, CUDA 12.9 on both sides, through
`tools/archlab/inspect_ptx.py`'s categoriser.

A *static* instruction count is per thread and is not a per-step quantity: the
naive port's fused kernel is 36,384 instructions containing all 32 steps, and
the hand-written kernel is 776 containing all 16 of its unrolled steps. Reading
"instructions per step" off either is a guess. The instrument here is a
**difference** — capture and compile the *same* program at `max_steps = S` and at
`S + d` and take the slope — which needs no assumption about the shape of the
generated code and is the same instrument on both sides. It is exact: the
least-squares fit over `max_steps ∈ {4, 8, 16, 32, 64}` gives r² ≥ 0.99995 on
every rung, so the cost really is linear in the trip count.

### 2.2 The three counts, per pixel per DDA step

`MEASURED` `out/minecu_tg_opt/opcount-b1024.json`, batch 1024, 32×32, world 16,
`max_steps` 32, sm_75. The per-pixel divisor is read off the generated kernel's
own grid (`blockIdx.x /* 512 */` × `threadIdx.x /* 2 */` × `threadIdx.y /* 8 */`
× `threadIdx.z /* 8 */` = 65,536 threads for 1,048,576 pixels), not assumed.

| | graph operations | generated statements | SASS instructions | registers | spill |
|---|---|---|---|---|---|
| **naive** (V0) | **30.00** | 30.51 | **74.4** | 168 | 20,932 B |
| int_cursor (V1) | 27.00 | 24.41 | 69.9 | 168 | 13,216 B |
| flat_gather (V2) | 9.00 | 16.27 | 59.4 | 255 | 3,068 B |
| incremental (V3) | 9.00 | 19.02 | 52.0 | 255 | 2,984 B |
| **lean** (V4) | **8.00** | 17.02 | **53.5** | 255 | 3,572 B |
| | **3.75×** | **1.79×** | **1.39×** | | **5.9× less spill** |
| hand-written reference | — | — | **38** | **28** | **0** |

The register and spill columns are at `max_steps` 32, which is the shape that
gets timed.

**The three ratios are the finding, not a footnote.** Hoisting the index
arithmetic removes **3.75× of the port's operations** and **1.39× of its machine
instructions**. Those are not in conflict: the operations removed are integer
address arithmetic, and ptxas was already compressing that (a shift-and-add
chain is three `IMAD`s, not three *uops*). The operations that remain, and which
the rewrites barely touch, are the float `t_max` compare-and-select chain — and
those expand into many more SASS instructions per uop than the address work
does. **The port's waste was large in the language's own vocabulary and modest
in machine code.**

### 2.3 The index-arithmetic share, stated separately and explicitly

Two measurements, and they are not the same measurement.

**Causal, from the ladder.** Each rung is the rung above plus one rewrite, so the
difference between consecutive rows is that rewrite's price, with nothing else
in the program changing.

| rewrite | graph ops removed, per pixel per step | SASS removed, per pixel per step |
|---|---|---|
| 1 — integer cursor | 3.00 (30 → 27) | 4.5 (74.4 → 69.9) |
| 2 — one flat index, not four | **18.00** (27 → 9) | **10.5** (69.9 → 59.4) |
| 3 — maintained address, bounds computed once | 0.00 | **7.4** (59.4 → 52.0) |
| 4 — no separate `found` | 1.00 (9 → 8) | −1.5 (52.0 → 53.5) |
| **total** | **22.00 of 30 = 73.3%** | **20.9 of 74.4 = 28.1%** |

Two of those four are worth a comment, because they are not what a reader
expects:

* **Rewrite 2 is the big one at the graph level (18 of 22) and rewrite 3 is the
  big one at the instruction level (7.4 of 20.9).** They are not rivals; they
  fix different halves. A 4-index gather materialises four index tensors and
  their arithmetic in the graph, which is where 18 uops go; a maintained
  address with a steps-to-exit bound test removes the clip and the address
  rebuild that survive rewrite 2, which is where the last SASS instructions
  are.
* **Rewrite 4 makes the instruction count slightly *worse*** — 52.0 → 53.5 per
  pixel per step, and the spill goes 2,984 → 3,572 B. `hit < 0` costs a compare
  on a value the register allocator then has to keep live, and ptxas responds by
  trading one for the other. It removes one tensor from the graph (9 → 8 uops)
  and one `|` per step, which is worth having in the language and is worth
  1.5 instructions in SASS. **It is kept because it is smaller, and it is
  reported here because "smaller" and "faster" are not the same claim.**

**Descriptive, from the statement mix.** Splitting the fused kernel at its
voxel reads and classifying every statement, the integer-domain share of one
step is 40.7% for V0 and 50.0% for V4 — the optimized step is *more*
integer-heavy, because what it removed was float. This corroborates the
direction of the ladder and cannot replace it: a classification assigns a label,
a difference measures.

### 2.4 The 47.2× was not a ratio of equal work, and the correction is large

`MEASURED` the same run. The earlier figure compared 36,384 generated SASS
against the reference's 776. Both are static, per-thread counts, and they cover
different amounts of work:

| | threads | rays per thread | static steps in the body | pixel-steps covered |
|---|---|---|---|---|
| reference (`kernels.cu:585`, `kernels.cu:2858-2861`) | 1,024 per env | **1** | 16 | **16** |
| generated, fused, `max_steps` 64 | 65,536 for 1,048,576 pixels | **15.7** | 64 | **1,007** |

So the two counts differ by 62.9× in the work they cover, and the raw quotient
is inflated by a factor of that size. Normalised:

| | naive | lean | reference |
|---|---|---|---|
| SASS instructions per pixel per DDA step | **74.4** | **53.5** | **38** |
| ratio against the reference | **1.96×** | **1.41×** | 1.00× |

**The honest statement of the original measurement is: the naive port issued 1.96×
the hand-written kernel's instructions per pixel per DDA step, and ran 27.2×
slower.** Not 47.2×. The 47.2× was a comparison of unequal work, and correcting
it does not help the port — it makes the port look *better* while the port's
time is unchanged.

### 2.5 The reference's own numbers, re-derived rather than quoted

`MEASURED` `minecu_tg_opt/opcount.py` walking
`minecu_ref/build/kernels_sm_75.sass`:

```
function            _Z28render_kernel_uint8_prebasisPKaPKfPhiiiiiii
static instructions 776
voxel loads         16  ['LDG.E.S8.CONSTANT.SYS']
load indices        [152, 190, 228, 266, 304, 342, 380, 418, 456, 494, 532, 570, 608, 646, 684, 722]
load-to-load gaps   [38]*15
registers / spill   28 / 0 B
step bodies         15/15 byte-identical to the first
BSSY / BAR.SYNC     33 / 1
STG.E.U8            6
```

The 776 decomposes as 152 of prologue + 16 × 38 of body + 16 of epilogue, the
16 voxel loads are at a spacing of exactly 38 fifteen times running, and the 15
bodies between them are byte-identical opcode streams. ptxas emitted all 16
unrolled steps and neither folded nor reordered nor spilled them.

---

## 3. Performance, same card, same run, render against render

`MEASURED` `out/minecu_tg_opt/20260929-151936/`, one Tesla T4, one session.
Hand-written side: `cudaEvent` medians over 8 iterations after 5 warmup, from
`refdrv.cu` linking the unmodified `kernels.cu`. TinyGrad side: the sum of
per-kernel device times over the whole compiled graph, via
`minecu_tg.bench.linear_device_time_ms` — **imported, not rewritten**, so the
ruler is the same one the earlier naive-port measurement used. Same world,
same `max_steps` 32, same uint8 output, byte-identical (§1.2), same measured
workload (§1.3).

`MEASURED` `minecu_tg_opt/analyse_run.py`, re-derived from the run's own
`ref_timing.json` and `tgopt/report.json`.

> **A units error in the run's own SUMMARY.txt, and its correction.** The first
> completed run printed its ratio column as `ref_microseconds / tg_milliseconds`.
> That is a mixed-unit quotient, inverted, and it reports 6.43 where the true
> figure is 155×; worse, it passes through 1.0 exactly when the two sides are
> *equal*, which is the one value a reader cannot mistake for a mistake. The two
> device-time measurements are unaffected. `analyse_run.py` recomputes the table
> from the run's own two JSON files, and `make_notebook.py` carries the fix for
> future runs. No GPU time was spent re-measuring a number that was already
> measured.

| batch | ref render µs | ref whole step µs | naive ms | lean ms | **naive/ref** | **lean/ref** | kernels n/l |
|---|---|---|---|---|---|---|---|
| 256 | 88.1 | 129.0 | 13.698 | 16.000 | **155.5×** | **181.6×** | 41 / 38 |
| 512 | 166.5 | 208.9 | 13.648 | 15.958 | **82.0×** | **95.9×** | 41 / 38 |
| 1024 | 323.6 | 369.1 | 8.811 | 2.923 | **27.2×** | **9.0×** | 10 / 6 |
| 2048 | 598.6 | 646.1 | 17.192 | 5.540 | **28.7×** | **9.3×** | 11 / 6 |
| 4096 | 1185.8 | 1237.5 | — | — | — | — | — |

And the allocator's own peak, same run:

| batch | naive | lean | allocations n/l |
|---|---|---|---|
| 256 | 133.8 MiB | 129.0 MiB | 26 / **1** |
| 512 | 143.6 MiB | 130.0 MiB | 11 / **1** |
| 1024 | 39.3 MiB | **4.0 MiB** | 11 / **1** |
| 2048 | 78.0 MiB | **0.0 MiB** | 11 / **0** |

### 3.1 The two regimes are two different experiments, and saying so is the point

**Above the crossover (batches 1024, 2048): the optimization wins, by 3.0×.**
2.923 ms against 8.811 ms at batch 1024; 5.540 against 17.192 at 2048. The gap
against the hand-written render falls from 27.2× to 9.0× and from 28.7× to 9.3×.
It does not fall to parity, and §3.3 says why not.

**Below the crossover (batches 256, 512): the optimization *loses*, by 14%.**
16.000 ms against 13.698 ms; 15.958 against 13.648. **This is a real and
interesting result and it is not a defect.** In that regime neither port fuses:
41 kernels for the naive port, 38 for the optimized one, with the DDA state
round-tripping through global memory at every step. The optimized port's
per-step *work* is smaller and its per-step *state* is the same size — a
maintained address is one int32, and three steps-to-exit counters are three
more, where the naive port's three float cursors plus three int32 clips plus a
`found` flag is not smaller. The graph is 3.75× smaller and the time is 14%
worse, because in the unfused regime the cost is the round trip, and the round
trip is set by how much state crosses it, not by how much arithmetic computed
it. **A port that is 3.75× smaller in operations can be slower when the
schedule it lands in is dominated by materialisation.** That is a statement
about the unfused regime, and it is why the crossover is where the comparison
has to be read.

One more observation about the below-crossover rows, offered because it looks
wrong and is not: the naive port's time is 13.698 ms at batch 256 and 13.648 ms
at batch 512 — flat, across a 2× change in work. 41 dependent kernels on 128
and then 256 blocks cannot fill a 40-SM card, so those kernels are
latency-bound and doubling the work does not move the time. It is the same
reason fusing wins, from the other side.

### 3.2 Why it wins where it wins: the fusion decision, not the instruction count

This is the mechanism, and it is worth stating precisely because it is not the
one the instruction counts would suggest.

The optimization removes 28% of the machine instructions per pixel per step
(§2.2) and that alone would be worth well under 2×. It is worth **3.0×**, and
the reason is in the kernel count and the allocation count:

| batch 1024 | kernels | distinct programs | allocations | peak |
|---|---|---|---|---|
| naive | 10 | 6 | 11 | 39.3 MiB |
| lean | **6** | **3** | **1** | **4.0 MiB** |

The launch count is constant in `max_steps` for both ports at batch 1024 (§4.1),
so both are in the "fused" regime by that test — and yet they differ by three
programs and ten allocations. The naive port's graph is large enough that
*part* of it still fails the fusion budget: the per-step state leaves registers,
is written to global memory, and is read back, and that residue is the three
extra programs and the 39 MiB. The optimized port's graph is small enough to
come through whole.

**So the causal chain from the port's index arithmetic to the 3.0× is: 30 → 8
operations per pixel per step → a graph small enough that tinygrad fuses the
whole 32-step body → the loop-carried state never leaves the register file →
3.0×.** The instruction count is a corollary of the fusion decision, not its
cause. This is the sense in which the index arithmetic was "the port's fault":
it was not directly expensive, but it was expensive by making the graph too big
for the scheduler's fusion budget.

> **⚠ THE CHANNEL IN THAT CHAIN IS WRONG, AND `notes/36` CORRECTED IT. The
> direction survives; the mechanism does not.** When all five rungs were timed
> (`notes/36`), the decisive rewrite turned out to be `flat_gather` — which
> removes 18 of the 22 graph operations and **zero launches (9 → 9 CALLs)**. So
> the launch count cannot be the channel, because the step that matters most does
> not change it. What moves 4.31× there is ptxas's local-memory frame,
> **13 216 → 3 068 bytes**, and that one number orders all five wall clocks at
> **Pearson +0.9986**. The corrected reading is in `notes/35` §5: **graph size is
> upstream of the cost, but the channel is register allocation of the fused
> straight-line region.** That is also why the faithful port's crossings cost
> nothing until they spilled. This paragraph is kept as written because being
> wrong here is instructive: the launch count and the allocator peak fell together
> and that coincidence is what made the fusion budget look like the cause.

### 3.3 What the remaining 9.0× is

`MEASURED` §2.2 and §2.4: the optimized port issues 53.5 SASS instructions per
pixel per DDA step against the hand-written kernel's 38 — a **1.41×** that this
note does not explain, and does not claim to.

Two measured contributions, and one that is not a contribution at all:

* **The masked formulation, 1.94×, and it is not the port's fault.** The
  hand-written kernel exits per ray (`goto done`, 33 `BSSY`/`BSYNC` pairs); a
  tensor program has no `goto`, so every ray runs all 32 iterations against a
  measured mean of 16.540 (§1.3). That is `32 / 16.540 = 1.94×` the executed
  work, and it is the largest single term in the remaining 9.0×. The generated
  code has `BSSY` 0 on every rung — it is *cheaper* per step and does
  substantially more total work. Removing this needs `Tensor.While` over a
  convergence mask, which is a different experiment and was not smuggled in.

  **This is the one figure in the note that changed the most on re-derivation.**
  The earlier work quoted 1.21× against a mean of 26.4 steps; the same run
  configuration on this card measures 16.540, and 32/16.540 is 1.94×. The
  workload is the same in both (world 16, `max_steps` 32, the same worldgen) but
  the two runs' cameras and step budgets land on different distributions, and
  1.21× is simply the wrong number for *this* measurement.
* **The float `t_max` chain, the rest.** Rewrites 1–3 leave the eight float
  operations of the compare-and-select chain per step untouched, and §2.3
  measured that they are where the surviving machine instructions are. The
  hand-written kernel does the same three-way branch natively in 38 instructions
  because it has a real branch; the masked form expresses the same branch as
  eight predicated float selects, and ptxas expands each of those into several
  instructions.
* **Not a contribution: launch overhead.** 6 launches against 1 is ~20 µs
  against 2,923 µs — under 1%. This is the negative result the earlier work
  also reached, and it is worth keeping: for this kernel, launch count is not
  the cost.

### 3.4 What the earlier 26.2× becomes

At batch 1024 this run measures the naive port at **27.2×**, against the 26.2×
reported earlier — a 3.8% difference, which is run-to-run variance on a shared
T4. The naive port's number is confirmed; what changes is what it is a ratio
*of*.

| | earlier | here |
|---|---|---|
| naive port, batch 1024, render vs render | 26.2× | **27.2×** |
| SASS instructions per pixel per step, naive vs reference | (47.2×, unequal work) | **1.96×** |
| the same, with the index arithmetic hoisted | — | **1.41×** |
| optimized port, batch 1024, render vs render | — | **9.0×** |

**Three of the four numbers in the earlier measurement reproduce; the one that
does not is the instruction ratio, and it does not reproduce because it was
never an instruction ratio.** The time ratio reproduces to 3.8%, which is
run-to-run variance on a shared card. The launch law does not (§4.1). The
1.21× work term does not (§3.3). Both of those were re-derived here rather than
carried forward, and both moved.

---

## 4. The structural findings: which survive, and they are the compiler's

### 4.1 The kernel-count law against `max_steps`

`MEASURED` `out/minecu_tg_opt/census-maxsteps.json`, NULL device with the CUDA
renderer attached (the schedule and the launch list are the real ones; nothing
is executed). 32×32, world 16, `max_steps ∈ {1,2,4,8,16,32,64}`.

| batch | pixels | naive, launches at ms = 1 / 2 / 4 / 8 / 16 / 32 / 64 | lean |
|---|---|---|---|
| 8 | 8,192 | 8 / 10 / 12 / 16 / 24 / 40 / 72 | 7 / 8 / 10 / 14 / 22 / 38 / 70 |
| 256 | 262,144 | 8 / 10 / 12 / 16 / 24 / 40 / 72 | 7 / 8 / 10 / 14 / 22 / 38 / 70 |
| 512 | 524,288 | 8 / 10 / 12 / 16 / 24 / 40 / 72 | 7 / 8 / 10 / 14 / 22 / 38 / 70 |
| 1024 | 1,048,576 | **8 / 9 / 9 / 9 / 9 / 9 / 9** | **6 / 6 / 6 / 6 / 6 / 6 / 6** |

Identical at batches 8, 256 and 512: the decision does not depend on the size
below the crossover, only on being below it.

Below the crossover the launch count rises with the loop trip count — the Python
loop unrolls at trace time and the scheduler emits one kernel per DDA step, with
the loop-carried state in global memory between them. The law is **exactly
linear from `max_steps` 1 to 8** (8/10/12/16 launches for the naive port,
7/8/10/14 for the optimized one) and then **stops being linear**: at 16, 32 and
64 the naive port is at 24/40/72, which is 16+8, 32+8 and 64+8 — the same
offset, against a trip count of 16/32/64. The slope is 0.5 launches per step
above 8, not 1. `MEASURED` same file, `launches − max_steps` printed in §6.1's
recipe.

**The prediction is confirmed in direction and refined in form.** The earlier
work reported `max_steps + 8` at every `max_steps` from 2 to 64; re-derived here
it is `max_steps + 8` for `max_steps ≤ 8` and a half-rate ladder above that, for
the same port on the same shapes. Nothing below the crossover is fused, which is
the finding; the exact offset is a detail, and this note does not carry the
earlier one forward.

The optimized port sits 2 to 3 launches below the naive one at every
`max_steps` below the crossover, because it has fewer tensors to materialise.
Above the crossover the launch count stops depending on `max_steps` entirely:
the scheduler fuses the whole unrolled body. Also confirmed, for both ports.

### 4.2 Where the crossover is, measured to the batch

`MEASURED` `out/minecu_tg_opt/census-batch.json`,
`out/minecu_tg_opt/census-crossover{,2,3}.json`, NULL device, CUDA renderer,
`max_steps` 32, `max_steps` 4 as the control.

| batch | pixels | naive launches at ms 4 / 32 | lean |
|---|---|---|---|
| 512 | 524,288 | 13 / **40** | 10 / **38** |
| 528 | 540,672 | 10 / **9** | 6 / **6** |
| 576 | 589,824 | 10 / 9 | 6 / 6 |
| 768 | 786,432 | 10 / 9 | 6 / 6 |
| 1024 | 1,048,576 | 10 / 9 | 6 / 6 |
| 2048 | 2,097,152 | 11 / 10 | 6 / 6 |

**The crossover is between 524,288 and 540,672 pixels, for both ports, and the
optimization does not move it.** That is the finding, and it is the one that
survives untouched by the rewrite: the fusion decision is the *scheduler's*,
taken on a size threshold, and the port does not get a vote. What the port does
get a vote on is how much of the graph survives the decision once it is taken
(§3.2).

### 4.3 The periodicity argument, on both sides, measured

`MEASURED` `out/minecu_tg_opt/opcount-b1024.json`, `longest_period` over every
period from 8 to 400 on each kernel's opcode stream.

| | static instructions | best exact repeating period | run length |
|---|---|---|---|
| reference, 16 unrolled steps | 776 | **38** | **15 of 15 bodies identical** |
| naive, fused, `max_steps` 64 | 74,312 | 8 | **1** |
| int_cursor | 69,408 | 8 | **1** |
| flat_gather | 59,248 | 8 | **1** |
| incremental | 53,256 | 8 | **1** |
| lean | 55,048 | 8 | **1** |

**The mechanism is confirmed and it is unchanged by the port.** The hand-written
kernel's 16 steps are byte-identical repeats, so ptxas can allocate registers
one copy at a time and lands at 28 registers with zero spills. Every fused
generated kernel, on either side of the ladder, has no repeating body at all —
ptxas has CSE'd, reordered and re-materialised across the whole straight-line
region, treating it as one optimisation problem rather than 32 copies of a
38-instruction step. That is why the register file, not the source, is the
binding constraint on the generated side, and it is why the optimized port still
sits at 255 registers with 3,572 B of spill.

**So the residual 1.41× of §3.3 has a name.** It is the cost of ptxas being
asked to allocate one 55,000-instruction straight-line region instead of sixteen
38-instruction repeats, plus the masked formulation's missing early exit. Neither
is the port's index arithmetic.

---

## 5. What the experiment can and cannot conclude

### 5.1 What it establishes

1. **A good TinyGrad program is competitive with this hand-written CUDA kernel
   on this workload, and a bad one is not.** 9.0× instead of 27.2×, with the
   output byte-identical to the hand-written kernel's own bytes, on the same
   card in the same run. The gap between those two figures is the port, and the
   port is fixable with ordinary means and no escape hatch.
2. **The naive port's 47.2× instruction ratio was a units error of 24× and a
   category error besides.** Corrected, the naive port issued 1.96× the
   hand-written kernel's instructions per pixel per step and the optimized port
   1.41×. Any study of TinyGrad codegen that divides two static instruction
   counts without checking that they cover the same work will get this wrong,
   and the error is always in the direction of overstating the gap.
3. **The cost of a naive port is not a cost you can read off its operation
   count.** The optimized port is 3.75× smaller in operations and 3.0× faster
   at batch 1024, but 1.07× *larger* in instructions and 14% *slower* at batch
   256. What a port's operation count buys is the size of the graph, and the
   size of the graph buys a fusion decision. Everything downstream of that is
   the scheduler's.
4. **The compiler's contribution on this workload is 1.41× in instructions and
   1.94× in executed work, not 47×.** The two compose: `1.41 × 1.94 ≈ 2.7×`,
   against 9.0× measured, so instruction count and work count together account
   for roughly the first third of the remaining gap and leave the rest to
   memory behaviour and to the register pressure of §4.3 — neither of which is
   re-measured with a profiler here, and neither of which is therefore
   decomposed further. **2.7× of 9.0× is an accounting of what is known, not
   a claim that the other 6.3× is one thing.**
5. **The structural findings are the compiler's, and three of the four survive
   intact.** The size-dependent fusion decision, its ~540,000-pixel threshold,
   the register pressure in the fused kernel (255 registers and 3.5 KB of spill
   against 28 and 0) and the periodicity difference (a 38-instruction period on
   the hand-written side, none on any fused generated kernel) are all
   independent of the port and all measured here on both ports. **The fourth,
   the `max_steps + 8` launch law, is refined rather than confirmed: exact to
   `max_steps` 8, a half-rate ladder above it.** §4.1.

### 5.2 What it does not establish

1. **Nothing about TinyGrad outside the fused regime.** Below ~540,000 pixels
   the generated side is 82–182× slow, and the optimized port is *worse* there.
   A port that is 3.75× smaller in operations losing 14% of its time to a
   smaller graph is a statement about the unfused schedule, and the two regimes
   want different ports.
2. **Nothing about the 1.41×.** This note measures that the optimized port
   issues 53.5 instructions per pixel per step against 38, and it explains the
   masked formulation's share of that. It does not explain the rest, and the
   residue is large enough that "1.41×" should be read as a bound on what is
   left, not as a characterisation of it.
3. **Nothing about the whole step.** Every ratio here is render against render,
   per `notes/17` §8.2. The hand-written whole step (6–9 kernels, five of them
   state updates totalling 464 SASS instructions) is in the table for context and
   is not a denominator anywhere.
4. **Nothing about another workload.** One DDA, one world, one card. The
   masked-loop formulation's missing per-ray early exit costs 1.94× here
   because this workload's rays average 16.54 steps out of 32; a workload where
   rays hit at step 2 would pay 16× for it. And 1.94× is a property of *this
   step distribution*, which is why it is quoted as measured and not carried
   over from the earlier run's 1.21× on a 26.4-step mean.
5. **The CPU gate did not run on the card.** tinygrad's CPU backend needs a
   compiler that can target bare-metal x86-64
   (`tinygrad/runtime/support/compiler_cpu.py:21`) and this image has none — no
   `clang`, and `CC=gcc` fails the same `--target=x86_64-none-unknown-elf` flag.
   The full gate ran on this host in `minecu_tg_opt/kaggle/run.sh` before the
   push, and the gate that replaced it on the card — both ports against the
   hand-written side's own bytes, §1.2 — is the stronger of the two. This is
   recorded because a run that skipped a check without saying so would look like
   a run whose checks passed.

### 5.3 What the study's contribution becomes

The thesis was: TinyGrad emits code that is one to two orders of magnitude worse
than a hand-written kernel on a voxel DDA, and the reason is a fusion and
register-allocation failure. **That thesis does not survive its own experiment's
control.** With the port written to the bar the experiment set, the gap on a
fused regime is 9.0×, of which the masked formulation's structural half is about
1.2× and the residual is a register-allocation effect that is real, measured,
and not a fusion failure. The study's contribution shifts, and it shifts to the
cases where the answer is still a gap:

* **the unfused regime**, where the generated side is 82–182× slow and a smaller
  graph does not help — §5.2.1;
* **the register pressure of a fused straight-line region**, 255 registers and
  3.5 KB of spill against 28 and 0, which no fusion heuristic addresses;
* **the per-ray early exit**, which a tensor program structurally cannot
  express and which costs 1.94× here and up to 16× elsewhere.

Those are worth publishing, and they are class **(ii)** and **(i)** in the
earlier work's taxonomy — representable-but-not-found, and not representable —
rather than the blanket "the compiler is bad at this" that a 47.2× number
implies. **A 9.0× gap with the output byte-identical is a publishable result and
is not softened here.** So is the 14% regression below the crossover.

---

## 6. Files

| what | where |
|---|---|
| the optimized port, and the ladder | `minecu_tg_opt/render.py` — read its docstring first |
| the correctness gate, CPU, no GPU | `minecu_tg_opt/verify_opt.py` |
| the ladder, priced, in one command | `minecu_tg_opt/ablate.py` |
| the capture, no GPU needed | `minecu_tg_opt/capture_opt.py` |
| the per-step counts, both sides, one categoriser | `minecu_tg_opt/opcount.py` |
| the sm_75 compile, offline, no `mktemp` | `minecu_tg_opt/compile_sass.sh` |
| the kernel census against `max_steps` and batch | `minecu_tg_opt/census_opt.py` |
| timing and memory on a real card | `minecu_tg_opt/battery_opt.py` |
| the completed T4 run's table, re-derived | `minecu_tg_opt/analyse_run.py` |
| the T4 run and its push recipe | `minecu_tg_opt/kaggle/` |
| the run | `out/minecu_tg_opt/20260929-151936/` |
| the SASS, both sides, all 32 kernels | `out/minecu_tg_opt/sass-b1024/`, `out/minecu_tg_opt/opcount-b1024.json` |
| the generated CUDA, five rungs × five `max_steps` | `out/minecu_tg_opt/codegen-b1024/` |
| the censuses | `out/minecu_tg_opt/census-*.json` |
| the naive port, untouched | `minecu_tg/render.py` |
| the hand-written side, untouched | `minecu_ref/src/kernels.cu`, sha256 `c6ed8f06…` |

### 6.1 Re-deriving every number here

No GPU and no card:

```
# the correctness gate
.venv/bin/python minecu_tg_opt/verify_opt.py

# the whole ladder: capture, sm_75 compile, every count
.venv/bin/python minecu_tg_opt/ablate.py --batch 1024 --steps 4,8,16,32,64

# or, the pieces separately
.venv/bin/python minecu_tg_opt/capture_opt.py \
    --outdir out/minecu_tg_opt/codegen-b1024 --batch 1024 --steps 4,8,16,32,64
for d in 0004 0008 0016 0032 0064; do
  ./minecu_tg_opt/compile_sass.sh out/minecu_tg_opt/codegen-b1024/max_steps_$d sm_75
  cp out/minecu_tg_opt/codegen-b1024/max_steps_$d/_sass/* out/minecu_tg_opt/sass-b1024/
done
.venv/bin/python minecu_tg_opt/opcount.py \
    --captured out/minecu_tg_opt/codegen-b1024 \
    --sass out/minecu_tg_opt/sass-b1024 \
    --json out/minecu_tg_opt/opcount-b1024.json

# the fusion crossover
.venv/bin/python minecu_tg_opt/census_opt.py --batches 512,528,576,768,1024 \
    --steps 4,32 --out out/minecu_tg_opt/census-batch.json

# the T4 run's table, from its own JSON
.venv/bin/python minecu_tg_opt/analyse_run.py out/minecu_tg_opt/20260929-151936
```

With a card, `./minecu_tg_opt/kaggle/run.sh`, which gates on `verify_opt.py` on
this host first and then runs the same notebook.
