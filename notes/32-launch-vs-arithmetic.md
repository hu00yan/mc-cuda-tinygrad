---
kind: reference
---

# 32 — the KDA UT transform, timed on a card: does it cost launches or arithmetic

**The gap this closes.** `notes/31` priced five spellings of the KDA UT
transform — the inverse of a `[64,64]` unit lower-triangular matrix — with an
offline schedule census, and then listed what it could not do. Two of its own
items are this note:

> **§8.2** *"Whether `C` or `D` wins, given that their lane-op ratio is 8.1× and
> their launch ratio is 0.17×. These point opposite ways and the crossover
> depends on the ratio of launch overhead to lane throughput on the specific
> card."*
>
> **§8.3** *"Whether `D2`'s 3-launch advantage survives being 24× the
> arithmetic. … The arithmetic is already known exactly, so only the per-launch
> cost is missing."*

**Why it was thought to need a bigger card.** Comparing against `fla` needs
sm_80+. Comparing **two of our own candidates** does not: C and D2 are both
TinyGrad implementations of the same operator, so timing them against each other
is a TinyGrad-internal comparison and it runs on the free Kaggle T4. The brief's
framing was right and the blocker was narrower than it looked.

**Instrument:** `kda/ut_time.py`. The candidate expressions, the problem
instance and the census are **imported from `kda/ut_price.py`, not
re-expressed** — two spellings that agree to 3.1e-08 can still differ in launch
count by 20×, and a hand-rewrite would change the quantity under test. What is
new is the clock.

```
python3 kda/ut_time.py --all                      # on a card
python3 kda/ut_time.py --from-json <ut_time.json> # re-derive §5 and §8 anywhere
```

**Run of record:** `out/gpush/runs/kaggle/ut-time-t4/20260929-155438-e0d520de90c9/`.
One run, one card, one session, 1,149.3 s of battery time. Payload sha256
`c2e8314f57701f7f`, 2 files, runner's payload verdict **MATCHED**.
`gpush kaggle run --dataset tenghuzhang/ut-time-payload --kernel
tenghuzhang/ut-time-t4 … -- python3 kda/ut_time.py --all`.

**Box** `MEASURED`: Tesla T4, 15,360 MiB, compute capability 7.5, driver
580.159.04, sm_75, 2 devices. python 3.12.13, tinygrad 0.14.0, numpy 2.0.2.
Knobs, read out of the installed tinygrad at runtime: `ALLOW_TF32=0 USE_TC=1
TC_SELECT=-1 TC_OPT=0 NOOPT=0 IMAGE=0 NOLOCALS=0 BEAM=0 SPLIT_REDUCEOP=1
CACHELEVEL=2 DEBUG=0`. A knob being on would invalidate every count here, for
the reason `notes/26` §7 gives, and none is.

**There is no timing comparison against the hand-written kernel in this note and
none may be inferred from it.** `fla` is not exercised by `kda/ut_time.py` at
all. Every number below is TinyGrad against TinyGrad, one framework, one card,
one instrument.

---

## 0. The answer, first

> **1. Launch overhead is 65% of candidate A's device time and 58% of
> candidate C's, so the honest answer to "which dominates" is *both, and it
> depends which candidate*. But for the pair the brief asks about, the ordering
> is decided, not a tie, and it is decided by the **arithmetic axis** — the
> launch axis predicted the wrong sign.**
>
> C is **1.77×** faster than D2 in device kernel time (840 µs against 1,483 µs),
> on a repeat-to-repeat spread of 5.9%, and it is 1.79× and 1.75× faster under
> the two constant spellings. D2 launches **3 fewer** kernels, so the launch axis
> predicted D2 ahead by 3 × 11.03 = **33 µs**; the card charged it **643 µs
> more**. The launch axis is wrong by 676 µs, in the wrong direction. The
> arithmetic axis predicted the right sign and over-shot the magnitude by 8.4×.
>
> **2. The offline census transfers to the card exactly.** All four candidates:
> 5,380 / 86 / 44 / 41 kernels, census equals device, difference zero, on every
> one. `notes/31`'s kernel column is a prediction about a real card and it is
> exactly right.
>
> **3. `A→B` is confirmed as a wall-clock result and it is the largest effect in
> the study of this operator: 62.6× fewer launches, 33.0× less device time,
> 46.7× less host time, for zero algorithmic change.**
>
> **4. The per-launch cost on this card, measured in this process with this
> instrument, is 11.033 µs** (R² = 0.9999 over 16 → 4,096 launches), and
> 11.460 µs for a kernel that moves a 16 KB buffer (R² = 1.0000). Not
> estimated from a vendor number.
>
> **5. The one-sentence verdict.** *For this operator the UT transform is a
> **cost-model** problem, not a scheduling problem and not an expressiveness
> problem: TinyGrad emits candidate C's 44 launches exactly as written and
> exactly as the offline census predicted, so its scheduling has nothing to get
> wrong — what it cannot do is price C against D2 without realizing both, and
> the quantity it optimises against, `ast.arg.estimates.ops`, prices this pair
> at 14.86× against a measured 1.77× while the `estimates.mem` field sitting in
> the same record prices it at 1.47×.*

The class-(ii) placement and its falsifier are §9. The falsifier was run. It
resolved, and not in the comfortable direction.

---

## 1. The instrument, and the three ways it was wrong before it was right

`notes/31` §1's rule is the reason this section exists: a census that does not
reproduce a known figure is a broken census, and a *timing* harness built on a
broken candidate set is worse than no timing harness. The census
(`kda/ut_price.py`) is re-run first, on the card, before anything is timed:

```
== self-test 1: the single-column chain of notes/26 section 3b ==
   C  steps  kernels  REDUCE  expect    lane_ops  verdict
   4      3        5       2       2          55  OK
   8      7       11       6       6         259  OK
  16     15       21      14      14       1,147  OK
  32     31       43      30      30       4,843  OK
  64     63       85      62      62      19,915  OK
== self-test 2: the port's 64 outer x (C-1) inner loop, 1 slab ==
   C  kernels  REDUCE  expect    lane_ops  verdict
   8       76      48      48       5,838  OK
  16      324     224     224      43,724  OK
  32     1316     960     960     460,302  OK
  64     5380    3968    3968   3,056,396  OK
```

`MEASURED` on the T4. The last row is `notes/31`'s own headline figure,
reproduced on the device that was not there when it was published.

Three instrument hazards were found by this harness being wrong first, not by
reading. All three are guarded in code and named in `kda/ut_time.py`.

### 1a. `Tensor.realize()` is a silent no-op after `schedule_linear()`

`MEASURED` — the first T4 run of this harness
(`out/gpush/runs/kaggle/ut-time-t4/20260929-151418-59aeedc47f08/`) reported
`t_realize = 0.127 ms` for candidate A. A is 5,380 launches. That is **24 ns per
`cuLaunchKernel`**, which is not a number, and the tell was that it was not a
number. B, C and D2 reported 0.078 / 0.076 / 0.066 ms for 86 / 44 / 41
launches — 0.9, 1.7 and 1.6 µs each, which are *also* wrong but wrong by a
factor of ten rather than a factor of a hundred, so the row that looked most
obviously absurd was A and the row that looked plausible was A's neighbours.

Mechanism, `READ-FROM-SOURCE`: `schedule_linear()` calls `transform_to_call` and
then `_apply_map_to_tensors(becomes_map, name="buffers")`
(`.venv/…/tinygrad/tensor.py:409-411`), which **assigns the buffers**. `realize()`
then filters on `not x.uop.has_buffer_identity()` (`tensor.py:422`), finds
nothing to do, and returns. It raises nothing. This is `notes/26` §6f, and the
harness walked into it because the shape of the call —
`build(); schedule_linear(); realize()` — reads like an end-to-end measurement
and is not one.

Fixed by taking the three host clocks on **three separate fresh graphs**, and by
**refusing** to print a timing whose realize is observed to have launched zero
kernels (`measure()` raises). The launch counts of the timed realizes are in
`ut_time.json` under `launched_in_timed_realize`; in the run of record they are
`[5380, 5380, 5380]`, `[86, 86, 86]`, `[44, 44, 44]`, `[41, 41, 41]`.

This is the brief's shape-assertion question asked of a *clock*: a timing
assertion that only checks that `realize()` returned would have passed on an
empty graph, exactly as a shape assertion would have passed on transposed data.

### 1b. `GlobalCounters.time_sum_s` is not device time

`MEASURED`, and it is the more consequential of the two. `time_sum_s` is
`sum(et for et in ets)` (`engine/realize.py:65`) **plus**, for every call whose
runtime returns no elapsed time, a host-side `perf_counter` delta taken after a
`Device.synchronize()` (`engine/realize.py:78-81`). `exec_copy` returns `[]`
(`realize.py:167-178`), so **every COPY is charged its host cost into a counter
whose name says device time.** On this card a COPY is a pinned-host allocation
(`runtime/ops_cuda.py:83` calls `cuMemHostAlloc` per copy-in), and it dominates:

| cand | kernel_us (real device time) | time_sum_s | the difference | COPYs | per COPY |
|---|---|---|---|---|---|
| A | 91,175 µs | 96,653 µs | 5,478 µs | 67 | 82 µs |
| B | 2,765 µs | 3,375 µs | 610 µs | 2 | 305 µs |
| C | 840 µs | 2,276 µs | **1,436 µs** | 11 | 131 µs |
| D2 | 1,483 µs | 2,446 µs | 963 µs | 6 | 161 µs |

For candidate C the counter reads **2.71× the device time**, and every one of
the eleven COPYs in it is host work. The headline device number in this note is
therefore `kernel_us` — the sum of the per-kernel CUDA-event times tinygrad
itself *prints* at DEBUG=2 (`realize.py:92-94`) — and `time_sum_s` is reported
beside it with the difference named rather than quietly dropped. Both are in
`ut_time.json`; `kernel_us` is a sum over the same lines the instrument printed,
so it is checkable by reading the log.

### 1c. The drain was itself a kernel

The first version drained the device with `Tensor(np.zeros(1)).realize()`, which
is a host-to-device COPY and therefore showed up in
`GlobalCounters.kernel_count` — the very number under report. `MEASURED` on this
host: with that drain, every candidate's device count came out **above** the
offline census, by 1 for B, C and D2 and by 9 for A. Replacing it with
`Device[Device.DEFAULT].synchronize()` — `cuCtxSynchronize`, no launch
(`runtime/ops_cuda.py:126-130`) — took every discrepancy to zero, and
`§4`'s table is the evidence.

Recorded because A's remaining gap of 8 was then a *real-looking* number that
would have been reported as a census error, and its actual cause turned out to
be something else entirely (§4): the schedule depends on the device string. Two
nearly-identical-looking discrepancies, one of them mine and one of them
TinyGrad's, and the only way to tell them apart was to fix mine first and see
whether the other one moved.

---

## 2. Correctness on the card, and a negative control for the gate

`MEASURED` on the T4, before any timing, at 2 slabs, `C=64`, against
`numpy.linalg.inv` in float64:

| cand | shape | max rel err |
|---|---|---|
| A | (2, 64, 64) | 3.108e-08 |
| B | (2, 64, 64) | 3.108e-08 |
| C | (2, 64, 64) | 3.108e-08 |
| D2 | (2, 64, 64) | 3.108e-08 |

Identical to `notes/31`'s `--verify` table to four significant figures, on a
different device. `VERIFY PASS`, and the run aborts if it does not.

**The gate has a negative control, because this operator has a documented
history of being ported from the wrong artifact while passing every test**
(`notes/06` §5b, ledger C-14 and C-15). The control is the transpose of a
correct result: same shape, different operator.

```
candidate C, as computed            relerr 3.013e-08  < tol
the SAME array transposed           relerr 5.246e-02  >= tol, REJECTED
the transposed array's SHAPE matches True  (so a shape assertion alone could not have caught it)
```

`MEASURED`. A shape assertion would have passed the transposed array. The
specific bug this is aimed at is `notes/31` §9: `Tensor.expand` left-aligns its
shape argument (`mixin/movement.py:162`, `_align_left`), so `expand` and
`[:, None]` disagree about which axis to widen — the result was wrong by 3.4×
and still passed a shape check. The gate in this harness is a **value** gate and
has been shown to reject the cheapest possible wrong operator.

---

## 3. The table

`MEASURED`, one run, one card, 64 slabs of `[64,64]` (`B·NT·HV = 2·8·4`), `C=64`.
`kernel_us` is the sum of the printed per-kernel CUDA-event times; min of 3
repeats, with the repeat-to-repeat spread. The three host columns are wall clock
on separate fresh graphs (§1a), min of 3.

| cand | launches | census | kernel_us | spread | copy_us | time_sum | ops(device) | t_build_ms | t_sched_ms | t_exec_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** naive 64×63 unrolled | **5,380** | 5,380 | **91,175** | 3.1% | 5,478 | 96,653 | 141,465,728 | 4,679.7 | 4,415.5 | 6,126.5 |
| **B** the same 64 solves, batched | **86** | 86 | **2,765** | 1.9% | 610 | 3,375 | 81,767,424 | 94.2 | 82.5 | 149.6 |
| **C** `BC=16` blocked | **44** | 44 | **840** | 5.9% | 1,436 | 2,276 | 22,160,128 | 63.8 | 57.1 | 118.7 |
| **D2** Neumann on the off-diagonal | **41** | 41 | **1,483** | 3.0% | 963 | 2,446 | 329,342,720 | 47.7 | 45.3 | 98.3 |

`MEASURED`. The offline axes, re-run in the same session, same expressions:

| cand | REDUCE (census) | DAG | alu_lane (census) | ops/device ÷ alu_lane | algorithmic MACs/slab | chain |
|---|---|---|---|---|---|---|
| A | 3,968 | 85 | 192,625,524 | **0.734×** | 391,168 | 63 |
| B | 62 | 85 | 79,888,114 | 1.024× | 391,168 | 63 |
| C | 20 | 34 | 22,081,474 | 1.004× | 66,012 | 17 |
| D2 | 22 | 34 | 17,731,413 | **18.574×** | 1,573,340 | 20 |

`RE-DERIVED` here, in the same process. Every census figure in
`notes/31` §3.1's `arange` column reproduces exactly on the card, including
`A`'s 192,625,524 — the number whose *larger-than-the-whole-operator* value
`notes/31` §3.5 used to disprove additivity.

**`ops(device)` is TinyGrad's own per-program op estimate**, summed over the
schedule that actually ran (`engine/realize.py:63`, `ast.arg.estimates`). It is
not this repo's census. Two different definitions of the same idea, side by side,
and their ratio to each other is not 1: A at 0.734× and D2 at **18.574×** while
B and C sit within 3% of 1. The non-additivity is not a footnote any more; it
is a column.

### 3a. `A → B`: the largest effect in this operator, now measured

`notes/31` §5's first decomposition — *"the 64 column solves are 64 independent
linear systems with the same matrix; batching them onto a leading axis changes
nothing about the work, the depth, or the arithmetic"* — was a census claim. On
the card:

| | A | B | ratio |
|---|---|---|---|
| device launches | 5,380 | 86 | **62.6×** |
| device kernel time | 91,175 µs | 2,765 µs | **33.0×** |
| host end to end (build+sched+exec) | 15,221.7 ms | 326.2 ms | **46.7×** |
| algorithmic MACs | 25,034,752 | 25,034,752 | **1.00×** |
| DAG depth (census) | 85 | 85 | 1.00× |

`MEASURED`. The 62.6× launch reduction is exactly the census's figure; the
device time falls 33.0× and the host time 46.7×, both **less** than 62.6× because
a per-realize fixed cost does not scale. The MAC count is identical, so none of
this is arithmetic. **The claim was right and it was a census; it is now a
clock.** For A specifically, 65.1% of its device time is launch overhead (§6),
which is the mechanism.

---

## 4. Does the offline census transfer to a card?

This is the check `notes/31` could not make, because it was priced on a host
with no CUDA device — `Context(DEV="CUDA")` makes `Device.DEFAULT` literally
read `"CUDA"` (`device.py:44`), so the census ran the real CUDA *scheduler* on a
machine with no CUDA *device*. The table above therefore cannot distinguish
"the census is right" from "the census is right about a scheduler that would
also be right here". So it is run three ways in the same session:

| cand | census with `DEV="CUDA"` | census with `DEV=` the timed device | kernels the device executed | difference |
|---|---|---|---|---|
| A | 5,380 | 5,380 | 5,380 | **0** |
| B | 86 | 86 | 86 | **0** |
| C | 44 | 44 | 44 | **0** |
| D2 | 41 | 41 | 41 | **0** |

`MEASURED`. Zero on every row. `notes/31`'s kernel column is a prediction about
a real card and it is exactly right, on all four candidates, including the
5,380 that `notes/13` and ledger C-18 disagree about.

One asymmetry is worth recording because it cost a diagnosis. On a **Metal** host
the two census columns *do* differ — by 8 kernels for A and 0 for B, C, D2,
independently of allocator state and of slab count. That is tinygrad's memory
planner reading the device string, not a census bug, and it is 0.15%. It is
recorded because a T4 number and a Metal number for the same graph are not the
same schedule, and a census run on one is a prediction about the other.

---

## 5. The per-launch cost, measured on this card

The brief is right that the census is not additive, so no instruction count is
converted to time by any factor here. Instead the per-launch cost is **measured
in this process on this card with this instrument**: a chain of *n* dependent
one-op kernels, each forced to materialise by a `.contiguous()`, timed exactly
as the candidates are. The x-axis is the number of kernels that **actually
launched**, read back from `GlobalCounters.kernel_count` — not *n* — so a fusion
surprise would show up as a wrong slope rather than as a silently wrong
constant.

| chain shape | 16 | 64 | 256 | 1,024 | 4,096 | fit | intercept | R² |
|---|---|---|---|---|---|---|---|---|
| `[1]`, 1 thread | 194 | 750 | 2,911 | 11,735 | 45,247 | **11.033 µs/launch** | 128 µs | 0.9999 |
| `[64,64]`, 16 KB | 195 | 731 | 2,975 | 11,700 | 46,954 | **11.460 µs/launch** | 6 µs | 1.0000 |

`MEASURED`, microseconds, at `C=64` and 64 slabs' worth of nothing. The two rows
differ by **3.9%**, which is the sensitivity check: a kernel that does nothing
but move one 16 KB buffer costs 11.46 µs on this card, and this operator's
widest intermediate is 16 KB for every candidate (`notes/31` §3.3). So the
per-launch cost is not sensitive to this operator's buffer sizes, and the
headline 11.033 µs can be used against all four candidates.

This is a number no note in this repository had, and it is the one
`notes/31` §8.3 said was "the only thing missing".

---

## 6. Two independent decompositions of the same total

### 6a. Subtract the measured launch cost

`kernel_us − launches × 11.033 µs`, at the bench shape. `MEASURED` on the left,
`RE-DERIVED` on the right.

| cand | launches | kernel_us | launch term | residual | residual share | ns per algorithmic MAC |
|---|---|---|---|---|---|---|
| A | 5,380 | 91,175 | 59,358 | 31,816 | **34.9%** | 1.271 |
| B | 86 | 2,765 | 949 | 1,816 | 65.7% | 0.073 |
| C | 44 | 840 | 485 | 355 | **42.2%** | 0.084 |
| D2 | 41 | 1,483 | 452 | 1,031 | **69.5%** | 0.010 |

**Launch overhead is the majority of A's device time, a bare majority of C's,
and a minority of D2's** — 65.1%, 57.7% and 30.5% respectively. So "does launch
overhead or arithmetic dominate" has no single answer for this operator: it
depends on which candidate, and A is the candidate `notes/31` called
catastrophic, for a reason that is now a subtraction rather than an adjective.
The launch term is charged against the *device* launch count, which includes the
COPYs; a COPY costs no device time at all (§1b), so this slightly over-charges
every row, by 3% for D2 and 25% for C, and moves nothing across a line.

The last column is the interesting one. `ns per algorithmic MAC` is the residual
divided by the algorithm's own multiply-add count. **A is 1.271 and D2 is
0.010** — D2's non-launch work costs 127× fewer nanoseconds per multiply-add
than A's. That is not a paradox, it is the whole shape of the operator: A
scatters a solved vector into `M` one column at a time (262,144 of its 391,168
MACs are the scatter), while D2's extra work is six `[64,64]@[64,64]`
contractions, which are the only thing in this operator that resembles a matmul.

### 6b. Sweep the shape and fit

If the schedule is a function of the graph and not the batch size, then
`launches` is flat in slabs and only the per-slab arithmetic grows, so
`kernel_us(slabs) = fixed + slope·slabs` separates the two axes without any
conversion factor. `MEASURED` at 1, 8, 16, 32, 64 slabs:

| cand | launches flat in slabs? | fixed_us | slope µs/slab | R² | fixed share at 64 slabs | slope / algorithmic MACs |
|---|---|---|---|---|---|---|
| A | **yes** | 68,681 | 427.9 | 0.8825 | 72.9% | 1.094 |
| B | **yes** | 1,668 | 20.7 | 0.6727 | 60.3% | 0.053 |
| C | **yes** | 566 | 5.1 | 0.7576 | 67.6% | 0.077 |
| D2 | **yes** | 588 | 14.7 | 0.9862 | 39.0% | 0.009 |

`MEASURED` and `RE-DERIVED`. The precondition — launches flat in slabs — is
**checked and reported per candidate**, not assumed, and it holds for all four.
The last column is the crux: **D2's marginal per-slab device cost is 2.9× C's
(14.7 against 5.1 µs/slab) while its algorithmic multiply-add count is 23.8×
C's.** D2's six `[64,64]@[64,64]` matmuls therefore run about **8× more
MAC-efficiently per unit of time** than C's 20 reduce kernels — six fat
contractions of 262,144 MACs each amortise a launch far better than twenty thin
ones of 3,301. That is why a 23.8× ratio in multiply-adds does not become a
23.8× ratio in time, and it is the mechanism behind §7.

The two decompositions agree on the conclusion and disagree on the split, and
the disagreement is informative rather than a nuisance: the sweep's intercept
(566 µs for C) is above the smallest measured point (528 µs at 1 slab), so a
line fitted through five points on a convex curve over-estimates the floor. §6a's
number is the one to use — it comes from a fit with R² = 0.9999 over a 256×
range of launch counts, and it is a direct subtraction of a measured constant
rather than an extrapolated intercept. The sweep's value is that it is
independent of §5 and it agrees on the sign and the ordering for all four rows.

---

## 7. C against D2: decided, and where the difference lives

| | C | D2 | ratio |
|---|---|---|---|
| device launches | 44 | 41 | 0.93× |
| **device kernel time** | **840 µs** | **1,483 µs** | **1.77×** |
| repeat spread | 5.9% | 3.0% | |
| `arange` spelling | 839 µs | 1,503 µs | **1.79×** |
| `numpy` spelling | 1,169 µs | 2,041 µs | **1.75×** |
| algorithmic MACs | 4,224,768 | 100,693,760 | 23.83× |
| ops/device (`estimates.ops`) | 22,160,128 | 329,342,720 | 14.86× |
| mem/device (`estimates.mem`) | 24,971,264 | 36,767,744 | **1.47×** |
| launch term, §6a | 485 µs | 452 µs | **0.93×** |
| residual, §6a | 355 µs | 1,031 µs | 2.90× |

`MEASURED`. **This is not a tie and no coin was flipped.** The 1.77× gap is
against a repeat-to-repeat spread of 5.9%, it reproduces at 1.79× and 1.75×
under the two constant spellings of `notes/31` §2 (which push the launch counts
from 44/41 to 71/64 — both up, the order preserved, so the launch axis still
favours D2 — and still leave C ahead by the same factor), and it reproduces in a
third measurement block, the shape sweep, where C at 64 slabs is 838 µs against
D2's 1,505 µs.

**Where the 643 µs lives.** `MEASURED`, the per-kernel DEBUG=2 lines split by
kernel class (`E_` elementwise, `r_` reduce):

| kernel class | C | D2 | delta |
|---|---|---|---|
| `E_` elementwise | 13 kernels, 275 µs | 13 kernels, 298 µs | **+23 µs** |
| `r_` reduce | 20 kernels, 567 µs | 22 kernels, 1,230 µs | **+663 µs** |
| total | 33 kernels, 840 µs | 35 kernels, 1,483 µs | +643 µs |

The whole difference is in the reduce kernels — D2's six `[64,64]@[64,64]`
Neumann products against C's 16 `[16,16]` merge products plus 4 14-step
substitutions. The elementwise tail, which is the block assembly and the
`· Diag(β)`, is the same work in both and costs the same to within 23 µs. And
the launch term moves the *other* way by 33 µs. So the 643 µs is arithmetic, and
it is 103% of the gap.

That is the answer to the brief's question, in its precise form:

> **Launch overhead is 65% of A's device time and 58% of C's, so the honest
> answer is "both, and it depends which candidate". But for C against D2 the
> launch terms are 485 µs and 452 µs — 33 µs apart — while the totals are 840 µs
> and 1,483 µs, 643 µs apart. The question is therefore decided by arithmetic,
> and the launch axis, which is the one `notes/13` and `notes/31` both lean on,
> gets it backwards by 676 µs.**

---

## 8. Which axis predicted it

| axis | predicts D2/C | measured | error | right sign? |
|---|---|---|---|---|
| **launch count** (`kernel_count`, exact) | **0.93×** | 1.77× | wrong sign, 676 µs | **no** |
| TinyGrad's own op estimate (`estimates.ops`) | 14.86× | 1.77× | 8.4× | yes |
| algorithmic MACs (`ut_price.macs`) | 23.83× | 1.77× | 13.5× | yes |
| TinyGrad's own mem estimate (`estimates.mem`) | **1.47×** | 1.77× | **1.20×** | yes |
| marginal per-slab cost (sweep slope) | 2.88× | 1.77× | 1.6× | yes |

The launch axis predicted D2 ahead by **33 µs** on an 840 µs budget. The card
charged it 643 µs more. That is not a tie and not a near-miss; the axis is
pointing the wrong way and its entire prediction is 5% of the error.

**The transferrable result is the row order, not any single number: for an
operator whose widest intermediate is 16 KB, the framework's launch count — the
one quantity it computes exactly, and the one `notes/13` and `notes/31` both
lean on — is not a cost proxy, and the quantity that predicts best is not
arithmetic either but bytes.** `estimates.mem` is a field TinyGrad already fills
in, already sums into `GlobalCounters.global_mem` (`realize.py:64`), already
prints — and that no optimisation reads. It prices this pair at 1.47× against a
measured 1.77×. `estimates.ops`, the field the framework is built around, is off
by 8.4× on the same pair.

This also settles a claim in the record. `notes/19` §6 row 9 recorded, for the
mine.cu DDA on this same card, that *"launch count is not the cost"* — 9
launches against 8,754 µs, under 1.5%. That is a true negative result and it
does not conflict: it is a **9-versus-1** comparison inside a graph whose
kernels each do milliseconds of work. This is a **5,380-versus-41** comparison
inside a graph whose kernels each do microseconds. **Both are right, and
together they say the launch axis is not a proxy at any fixed strength — it has
to be calibrated against the arithmetic the kernels actually do, and at
`[64,64]` fp32 the arithmetic is far too small to saturate a Turing SM.**

**Is the third possibility — bandwidth — ruled out? Yes, arithmetically.**
`GlobalCounters.global_mem` is TinyGrad's own estimate of bytes moved, `MEASURED`
here; the T4's 320 GB/s is a **vendor specification, not measured in this note**,
so this row is `MODELLED` and says so:

| cand | bytes moved (measured) | at 320 GB/s | kernel_us (measured) | bandwidth is |
|---|---|---|---|---|
| A | 119,439,872 B | 373 µs | 91,175 µs | 0.4% of the time |
| B | 90,174,208 B | 282 µs | 2,765 µs | 10.2% |
| C | 24,971,264 B | 78 µs | 840 µs | 9.3% |
| D2 | 36,767,744 B | 115 µs | 1,483 µs | 7.7% |

`MODELLED` from a measured byte count and a vendor bandwidth. Even if the
vendor figure is 2× optimistic nothing crosses 20%, so none of the four is
bandwidth bound and the "launch and latency" reading stands on the remaining
~90%. **This is the one place in this note where a hardware constant from a
datasheet enters an argument**, and it is flagged at the point of use because
`notes/10-t4-results.md` §8 exists because unflagged datasheet numbers outlive
their scope.

**A correction for whoever owns `notes/31`, flagged and not edited.** Its §3.3
says `max_out_bytes` is **16,384 B for all five** candidates at the 64-slab
bench shape. `RE-DERIVED` here with the same census on the same expressions:
16,384 B at **1** slab and **1,048,576 B at 64 slabs**, for all four candidates,
with `max_arg_bytes` identical to `max_out_bytes` in every case. So the figure
`notes/31` quotes is the one-slab number, and its conclusion — that the operator
is bandwidth-trivial and no candidate can win on bytes — survives at 64× the
size it was stated at: 1 MiB per buffer, moved 33 times, is 66 MiB against a
measured 840 µs, which is the 9.3% row above.

---

## 9. The class-(ii) verdict, and the falsifier that was run

**Placement.** `notes/19` §6's four classes: **(i)** cannot express, **(ii)**
can express but does not find, **(iii)** not yet implemented, **(iv)** does not
exist for this architecture. C and D2 are both expressed, both implemented, both
verified to 3.108e-08 against a float64 `numpy.linalg.inv` **on the card**, and
both run. So (i), (iii) and (iv) are excluded by measurement, not by argument.
What is at stake is (ii): which of two good implementations the framework's own
performance model would prefer.

**Measured answer to that question: it prefers the right one, and for the wrong
reason, by a factor of 8.4.** TinyGrad's only per-program cost estimate,
`ast.arg.estimates.ops`, ranks C above D2 by 14.86× when the card's ratio is
1.77×. The *other* quantity it maintains, `kernel_count`, is exact — and ranks
D2 above C by 7% when D2 is 77% slower. **The framework's two cost proxies
disagree with each other by 16.0× on this one operator** (14.86 ÷ 0.93, the
`ops` column against the `launches` column of §7's table), and the one that is
exact is the one that is wrong.

**The falsifier, stated so it can be run, and then run.**
`kda/ut_time.py --from-json` re-derives this section from the run's own
artefact, with no card and no retyping:

```bash
python3 kda/ut_time.py --from-json \
  out/gpush/runs/kaggle/ut-time-t4/20260929-155438-e0d520de90c9/artifacts/ut_time_out/ut_time.json
```

The test: *refit a linear cost model on the other candidates, predict the
held-out one, and ask whether the prediction lands inside that point's own
repeat-to-repeat spread.* `inside` would mean a cost model over the quantities
TinyGrad already computes is sufficient and the class-(ii) gap is a calibration
constant. `outside` means it is not.

| model | terms | R² on all 20 sweep points | held-out points inside their own spread |
|---|---|---|---|
| 1 | `a + b·launches` (4 points) | 0.999876 | **0 / 4** |
| 2 | `a + b·launches + c·ops` (20 points) | 0.979793 | **0 / 20** |
| 3 | `a + b·launches + c·ops + d·mem` (20 points) | — | **0 / 20** |

`RE-DERIVED` by `kda/ut_time.py`, from the artefact. Model 2 over-predicts D2 at
64 slabs by **11.7×** (17,568 µs predicted against 1,505 µs measured). Model 1
over-predicts A by 2.1×. **No linear form over TinyGrad's own scalars predicts
this operator's held-out cost.** The falsifier resolved against the comfortable
reading.

**So the class-(ii) claim survives only in its stronger form, and this is the
finding:** *the missing thing is not a number, it is a functional form.*
`estimates.mem` carries the right signal for the pair in question — 1.47×
against a measured 1.77×, where `estimates.ops` is 14.86× — and a *ratio* of the
two quantities is off by only 1.20×, while no global linear combination of them
extrapolates across candidates. What a cost model for this operator needs is the
one thing a scalar op count cannot carry: that a `[64,64]` fp32 tile is 16 KB,
occupies a fraction of one SM, and therefore converts estimated ops into time at
a rate that depends on the size of the tile, not on the count.

**The falsifier of the stronger form, stated for whoever picks this up:** build
the model with an explicit per-kernel size term —
`t ≈ Σ_k [ L·τ_launch + mem_k·τ_byte + ops_k·f(bytes_k) ]` with
`f` monotone and bounded, `τ_launch = 11.033 µs` measured here, and
`τ_byte, f` fitted on the 20 sweep points — and hold out a candidate. If that
predicts all 20 held-out points inside their spread, the class is (ii) and the
gap is a two-constant calibration nobody has written down. If it does not, then
the quantity a cost model for this operator needs is **not** a function of
launch count, estimated ops and estimated bytes, and the honest class for
"which of these two spellings should I write" drops to **(i)** — even though
both spellings exist, because the thing that would choose between them is not
expressible in the framework's own cost vocabulary. I do not know which, and
saying so is cheaper than guessing.

---

## 10. Hardware counters, and what survives without them

### 10.0 The result: not available, for two separate reasons

`ncu` **is** on the Kaggle image (Nsight Compute 2025.1.1.0,
`/usr/local/cuda/bin/ncu`). Two attempts, two distinct failures, both recorded
because both look like a permissions problem and neither is.

1. **Missing Python dependency.** The first attempt failed on all four
   candidates with `ModuleNotFoundError: No module named 'wrapt'`
   (`out/gpush/runs/kaggle/ut-time-t4/20260929-155438-e0d520de90c9/`, §6 of
   `ut_time.txt`). `ncu`'s front end imports `wrapt`; the Kaggle image does not
   ship it. Not `NVreg_RestrictProfilingToAdminUsers`, not a permission denial —
   the profiler never ran.
2. **The fix never reached the card.** `kda/ut_time.py` now installs `wrapt`
   (verified rc=0) and passes `--launch-count` as a flag rather than as a bare
   positional. The re-run
   (`out/gpush/runs/kaggle/ut-time-t4/20260929-161853-fcd71b6a2397/`) came back
   `COMPLETE` in **9 seconds** and its artefacts are a byte-identical copy of the
   previous run's, including the identical failure. Its
   `GPUSH-VERIFY.json` reports fingerprint **`e0d520de…`** while the run's own
   provenance says **`fcd71b6a…`**, its shipped attestation names the *previous*
   run's command (`--all`, not `--counters`), and its fetched `ut_time.py` is
   55,652 B where the pushed file was 74,175 B. **That run executed the
   previous run's payload.** The ncu fix is in the working tree and was not
   measured.

   This is a transport finding and it is `DECISIONS.md` **#18**'s shape again,
   one layer up: a run whose payload is not the payload that run pushed, reported
   as a clean success with a `MATCHED` verdict. The verdict is true — the tree
   the runner hashed did match the attestation *inside that tree* — and it is
   silent about the only question that matters, which is whether that tree is
   the one *this* invocation published. The local-side gates (published file set,
   published bytes re-hashed) passed, so the dataset was correct and the mount
   lagged it. `gpush/` is not mine and is not modified; flagged for whoever owns
   it. **The cheapest fix is a one-line comparison the record already has both
   sides of: `cmd_run` should refuse (or re-poll) when the fingerprint in the
   fetched `GPUSH-VERIFY.json` differs from `tree.fingerprint`, instead of
   printing `MATCHED`.**

   It does not affect the run of record. That run is self-consistent at every
   level: the runner re-hashed the tree it executed and got run 2's own
   per-file digests (`61d94ca8…` for `ut_time.py`, `d1a5dfab…` for
   `ut_price.py`), the fingerprint it reports is run 2's, and the fetched code
   carries the `kernel_us` schema, the `t_exec_ms` column and the
   silent-no-op guard that the previous run did not have.

**No counter in this note is `MEASURED`.** `ncu` is installed on the box, the
missing dependency is now handled, and one more `--counters` run would settle
it; that run is not part of this note and nothing below may be read as if it
were.

### 10.1 Which conclusions need counters and which do not

Stated before the result precisely so it cannot be read backwards.

| conclusion | needs a counter? | why |
|---|---|---|
| A is 33× slower than B, and 65% of that is launch overhead | **no** | the launch count is `GlobalCounters.kernel_count`, an exact count of executed kernels, and the per-launch cost is a CUDA-event fit at R² = 0.9999 |
| C beats D2 by 1.77×, decided by arithmetic | **no** | per-kernel CUDA-event times split by kernel class; the gap is 103% of one class |
| launches are flat in slabs, so the sweep fit separates the axes | **no** | it is a count, not an estimate |
| the offline census transfers to the card exactly, zero difference on all four | **no** | a count against a count |
| `A→B` is 62.6× launches / 33.0× device / 46.7× host for zero algorithmic change | **no** | all three are measurements in this note; the MAC count is re-derived from the algorithm |
| **the *reason* arithmetic does not dominate D2 — that its 23.8× MACs convert to time ~8× more efficiently than C's** | **partly** | the sweep's two slopes (5.1 vs 14.7 µs/slab) establish it *behaviourally*. `sm__inst_executed_pipe_fma.sum` would confirm it by counting *issued* instructions rather than timing them, and it is the one measurement that would rule out "the op ESTIMATES differ in ratio from the instructions the hardware actually issues" as an alternative explanation. **Not measured.** |
| **that none of these kernels is bandwidth bound** | **partly** | `MODELLED` above: ≤10% of measured time at a vendor 320 GB/s. A `dram__bytes.sum` counter would make it `MEASURED` and would also catch traffic TinyGrad's estimate misses. **Not measured.** |
| **that none of these kernels reaches the FMA pipe** (the occupancy claim in §8) | **yes** | `sm__throughput.avg.pct_of_peak_sustained_elapsed` and `sm__warps_active.avg.pct_of_peak_sustained_active`. **Not measured.** §8's closing paragraph is a *model* plus an *inference from the timings*, and is labelled as such where it is made. |

Everything above the last two rows stands on CUDA events and exact counts with
no counter at all. The two rows below them are the mechanism claims, and they are
the ones a reader should treat as the softest sentences in this note.

---

## 11. What this note does not say

1. **Nothing about the hand-written kernel.** `fla` is not imported, not
   installed, and not timed by `kda/ut_time.py`. Every ratio here is
   TinyGrad/TinyGrad. `notes/30`'s sm_75 configuration findings are a separate
   scope and are not touched here.
2. **Nothing about `kda/tiny_kda.py` as a whole.** A, B, C and D2 are scheduled
   **in isolation**, and `notes/31` §3.5 proved the isolated census is not
   additive: candidate A alone reports 192.6M ALU-lane ops against the whole
   operator's 57.8M. So the 5,380 launches here are a property of the
   standalone graph, not a measurement of the port. The port's own census is
   `notes/26`'s, and ledger C-18's "96 kernels" is a different graph. **The
   right way to use §3 is comparative, across these four candidates, and not as
   a share of any whole operator.**
3. **Nothing about sm_80.** A T4 is sm_75: no bf16 tensor cores, and `notes/31`
   §7 established that these fp32 shapes get no tensor core at any of them
   anyway, so the ranking is not expected to move. "Not expected" is a
   prediction, not a measurement, and it is labelled as one.
4. **Nothing about the `arange`/`numpy` fork being settled.** §7 shows the
   C-vs-D2 ordering is spelling-invariant. It says nothing about which spelling
   is better overall: `numpy` costs C 1,169 µs against `arange`'s 840 µs, a
   1.39× penalty, and buys nothing here.
5. **One run.** Every number is one session on one card. The repeat-to-repeat
   spreads (0.7% to 5.9%) bound within-session noise and nothing else: not
   run-to-run, not card-to-card, not session-to-session. The 1.77× gap is 30×
   the worst spread, which is why it is called decided; a 7% claim would not be.

---

## 12. Reproduce

```bash
# offline, no card: the census self-test and the gate's negative control
.venv/bin/python kda/ut_time.py --selftest

# on the free Kaggle T4 (sm_75). This is the run of record.
./gpush/gpush kaggle run \
  --dataset tenghuzhang/ut-time-payload --kernel tenghuzhang/ut-time-t4 \
  --kaggle-cli .venv/bin/kaggle \
  --include kda/ut_price.py --include kda/ut_time.py \
  --session 50 --queue-wait 40 --poll 20 --timeout 3600 \
  -- bash -c "python3 -m pip install --no-input --disable-pip-version-check \
             tinygrad==0.14.0 >/dev/null && python3 kda/ut_time.py --all --out ut_time_out"

# anywhere, no card: re-derive the launch calibration and §9 from the artefact
.venv/bin/python kda/ut_time.py --from-json \
  out/gpush/runs/kaggle/ut-time-t4/20260929-155438-e0d520de90c9/artifacts/ut_time_out/ut_time.json
```

No paid compute, no A100, no Lightning, no paid Colab, no Modal. Free Kaggle
T4; three sessions, of which one (§10.0) measured nothing.

**The artefacts of record:**

| what | where |
|---|---|
| the instrument | `kda/ut_time.py` — sections 0–7 ran on the card; `verdict` (§8/§9) and `--from-json` were added after the run and re-derive from the card's own JSON. The **exact file that executed** is the run directory's `artifacts/kda/ut_time.py`, 55,652 B, sha256 `61d94ca8…`; the working-tree file is 74,175 B and differs from it only by the additions made after the run. |
| **the measurement** | `out/gpush/runs/kaggle/ut-time-t4/20260929-155438-e0d520de90c9/` — `artifacts/ut_time_out/ut_time.txt` (the log), `ut_time.json` (every row, including all repeats and every `kernel_us_all` list), `gpush-run.txt` (raw subprocess stdout, including the census self-test that `ut_time.txt` does not capture), `provenance.json` (fingerprint, dataset, sha256, device, plan), `GPUSH-VERIFY.json` (the runner's own verdict on the bytes it executed) |
| the superseded run | `…/20260929-151418-59aeedc47f08/` — kept because §1a and §1b are bugs this harness had, and a reader who wants the broken numbers has them |
| the run that executed nothing | `…/20260929-161853-fcd71b6a2397/` — kept because §10.0's transport finding is the evidence, and the evidence is a 9-second `COMPLETE` with the previous run's payload |
