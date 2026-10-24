---
kind: reference
---

# 35 — How far a TinyGrad port gets, and what each step was worth

**This is the handoff document.** It is written so that someone can continue the
study without reading any code. Every number is `MEASURED` on a Tesla T4 (sm_75)
or is explicitly marked as not. The sources are cited so any number can be
re-derived rather than believed.

**The one-sentence answer.** A correct, carefully optimised TinyGrad port of a
hand-written CUDA raymarcher runs at **11% of the hand-written kernel's speed** in
the regime where the framework fuses the work, and at **1%** in the regime where
it does not. The reason is not arithmetic and not tuning: it is that TinyGrad's
decision to fuse a loop is a function of graph size, and there is a threshold the
port lands on one side of.

**What "95% of cuBLAS" would mean here.** If this study were a matmul result
rather than a raymarcher result, the headline number would be the percentage of
the reference library, the way a hand-written SGEMMM reaching 95% of cuBLAS is.
The equivalent number here is in §4. It is 11%, and the two regimes differ by an
order of magnitude, so **which regime a number comes from has to travel with it.**

---

## 1. The subject, and the benchmark

**The hand-written reference** is `minecu_ref/src/kernels.cu` — a single fused
voxel raymarcher that marches a ray through a grid, and the piece of
Infatoshi's `mine.cu`. It is hash-pinned to upstream (`sha256 c6ed8f06…`) and was
measured directly on this project's own T4:

- **7 198.9 µs per environment step, 4.55 M environment-steps/s, 26.416 DDA steps
  per pixel** — whole-step figures, 7 kernel launches (`out/minecu-t4/`).
- **Per render, per batch** — this is the denominator used everywhere below
  (`notes/23` §3, `notes/28`).

**The benchmark is that kernel.** Everything in §3–§4 is a ratio to it, measured
render-against-render at equal batch, both sides in the same run, on the same
card, in the same process.

---

## 2. The ladder, and the mistake each rung avoids

The port was written twice. The first version (`minecu_tg/`, rung `naive`) was
correct and passed every correctness gate. It was also **27× slower than the
reference**, and the reason turned out to be that the first version was not a
faithful translation of the algorithm. The second version walks a ladder, and
each rung is the one above plus exactly one rewrite.

| # | rung | what it does | the mistake it avoids | why it is not a hardware trick |
|---|---|---|---|---|
| 0 | `naive` | the baseline, `minecu_tg/render.py` | — | — |
| 1 | `int_cursor` | the voxel cursor is held as an integer and cast once, instead of as a float that is cast every step | the cursor **is** an integer; the hand-written kernel says so at `kernels.cu:638` | a type fact, readable in the reference |
| 2 | `flat_gather` | one flat index into the voxel array instead of a four-dimensional index | four index computations to fetch one byte | the array is flat; the reference advances a single address |
| 3 | `incremental` | the address is **maintained** with precomputed strides, and the per-axis bounds test becomes a steps-to-exit countdown computed once | recomputing from three float coordinates on every one of 32 steps | **Amanatides–Woo, the algorithm the reference implements, precomputes and then advances incrementally.** Recomputing is not that algorithm |
| 4 | `lean` | no separate `found` flag — `hit < 0` already means "not found yet" | one more live tensor across 32 steps | a strict simplification |

**None of these is a hardware hint.** There is no `custom_kernel`, no tile size,
no `TC_SELECT`, no `Context`, no hand-unrolling, no codegen knob, no
device-specific branch, and the step loop is still a plain Python `for` over
`max_steps` with one body. That constraint is what makes the result a statement
about a compiler rather than about a port — and it was **breached twice** during
this project in ways the corrections ledger records (C-14, C-16).

---

## 3. What each step was worth

**Graph operations per pixel per DDA step** — the number that turned out to
matter most (`notes/23` §2):

| step | graph ops | machine instructions per pixel per step |
|---|---|---|
| `naive` | 30 | 74.4 |
| + `int_cursor` | 27 | 69.9 |
| + `flat_gather` | **9** | 59.4 |
| + `incremental` | 9 | 52.0 |
| + `lean` | 8 | 53.5 |

**The middle column is the story and the right column is not.** Removing 22 of 30
graph operations (`flat_gather` alone removes 18) is worth an enormous amount of
wall clock; removing another 24% of machine instructions (`incremental`, then
`lean`) is worth almost nothing — `lean` is actually 1.5 instructions *worse* and
still the fastest version.

**Wall clock, same card, same run, render against render, every rung timed**
(`notes/36`; the reference re-measured **in the same run** at 241.1 µs for batch
1024 and 466.9 µs for 2048, which is the number to use — `notes/23`'s 323.6 µs is
a different session on a shared free-tier T4, and a ratio of two sessions is not a
ratio):

| rung | ms @256 | ms @512 | ms @1024 | ms @2048 | CALLs @1024 | ×ref @1024 |
|---|---|---|---|---|---|---|
| `naive` | 13.720 | 13.682 | **8.880** | 17.460 | 10 cold / 9 warm | 36.8× |
| `int_cursor` | 13.293 | 13.285 | **6.034** | 11.949 | 9 / 9 | 25.0× |
| `flat_gather` | 13.402 | 13.369 | **3.020** | 5.817 | 9 / 9 | 12.5× |
| `incremental` | 16.258 | 15.897 | **2.741** | 5.105 | 7 / 7 | **11.4×** |
| `lean` | 16.508 | 16.039 | **2.950** | 5.512 | 6 / 6 | 12.2× |

Every rung was gated on the card before any of them was timed: all five produce
byte-identical output to the hand-written driver's own bytes over 1 048 576 pixels,
and all five ran the same 16.5384 DDA steps per pixel.

**Two things this table says that the previous version of this report could not.**

1. **The best rung is `incremental`, not `lean`.** `lean` is 7.6% slower, and
   consistently so. The rung this study submitted as its optimised port is not its
   fastest port. That is in the record rather than tidied away.
2. **Below the crossover the ladder inverts.** At batches 256 and 512 the two
   *smallest* graphs are 16–20% **slower** than the naive one, and no currency
   measured here predicts it. So the ladder is not monotone, and a report that
   presented it as a climb would be wrong.

---

## 4. The gap to the benchmark, absolute and in percentage

Batch 1024, **both sides measured in the same run on the same card** — which is the
only basis on which this ratio means anything:

| | absolute | ratio | as % of the reference |
|---|---|---|---|
| hand-written reference | **241.1 µs** | 1.00× | 100% |
| best TinyGrad port (`incremental`) | **2.741 ms** | 11.4× | **8.8%** |
| `lean` (the rung submitted) | 2.950 ms | 12.2× | 8.2% |
| faithful port, crossings included | 3.012 ms | 12.5× | 8.0% |
| first port (`naive`) | 8.880 ms | 36.8× | 2.7% |
| **what the framework charges for the gap** | **2.50 ms** | | **91.2 points of the 100** |

**The absolute number to carry: 2.50 ms of unexplained cost per render at batch
1024.** That is the size of the prize. It is a measured duration difference between
two kernels computing the same thing, on one card, in one process, with both sides
gated byte-identical first.

**In the language of the cuBLAS comparison this study is imitating: the best port
reaches 8.8% of a hand-written CUDA kernel, from a correct-but-first-try 2.7%.**
The honest reading of that 6.1-point improvement is that **most of what the study
set out to find was found by writing better code, not by finding a compiler
flaw** — and that the remaining 91 points are the part that belongs to a compiler.

---

## 5. What the mechanism is, and what rules it out

**There is a threshold, and the port lands on one side of it.** Above ~1 M pixels
TinyGrad fuses the whole 32-step body into one kernel; below it, it does not, and
launches one kernel per DDA step. The crossover is pinned to **524,288 – 540,672
pixels** and holds for both the first port and the optimised one.

**The consequence is visible in the allocator, same run, endpoints only:**

| batch | `naive` peak | `lean` peak | allocations naive/lean | kernels naive/lean |
|---|---|---|---|---|
| 256 | 133.8 MiB | 129.0 MiB | 26 / 1 | 41 / 38 |
| 512 | 143.6 MiB | 130.0 MiB | 11 / 1 | 41 / 38 |
| 1024 | 39.3 MiB | **4.0 MiB** | 11 / 1 | 10 / 6 |
| 2048 | 78.0 MiB | **0.0 MiB** | 11 / 0 | 11 / 6 |

**The mechanism, corrected by the data that finished measuring it.** An earlier
version of this report attributed the win to crossing the fusion budget, on the
evidence of launch count and allocator peak falling together. **The per-rung
timings refute that channel and keep the direction.** Excess of each rewrite over
what its instruction count predicts, batch 1024:

| rewrite | graph ops removed | launches removed | excess over the SASS prediction |
|---|---|---|---|
| `int_cursor` | 3 | 1 | 1.383× |
| **`flat_gather`** | **18** | **0 — 9 CALLs either side** | **1.697×** |
| `incremental` | 0 | 2 | 0.964× |
| `lean` | 1 | 1 | 0.956× |

**The decisive step removes 18 of the 22 graph operations and not a single launch.**
All 2.16× of the ladder's unexplained gain (3.011× measured against 1.391× predicted
by instructions) sits in the first half; rewrites 3 and 4 contribute nothing.

**So the channel is not the launch count.** What moves 4.31× at that step is
ptxas's local-memory frame, **13,216 → 3,068 bytes**, and that one number orders all
five wall clocks at **Pearson +0.9986** (batch 1024) and **+0.9985** (batch 2048).

> **⚠ §5c: THIS SENTENCE IS FALSIFIED. Both the attribution and the number are
> wrong. `notes/38` built the case §5 said did not exist, and it breaks the claim.**

**1. The "frame" was never the frame.** 13 216 → 3 068 is **spill stores plus spill
loads**; the identity holds for all five rungs. The actual stack frame is
**5 472 → 1 312 bytes**, and **the two quantities rank the rungs differently.** So
the number §5 correlates against wall clock is the right *shape* of quantity —
spill is what a register-allocation story needs — but it is mislabelled, and it is
not the quantity the mechanism names.

**2. The mechanism is falsified, not merely unproven.** Holding everything else
fixed and changing only one thing:

| `flat_gather`, batch 1024 | stock | `NOLOCALS=1` |
|---|---|---|
| wall clock | 3.0151 ms | **57.2876 ms (19.0×)** |
| launches | 9 | 9 |
| output bytes differing from the reference | 0 / 1 048 576 | 0 / 1 048 576 |
| op count (statements) | 8 891 | 8 890 |
| **frame / spill / registers / SASS / LDL / STL** | 1160 / 1152·1152 / 255 / 4096 / 287·287 | **identical** |

**The spill did not move. The time moved by an order of magnitude.** So spill does
not explain the time, and §5's reading — "the channel is register allocation" — is
**false**, not merely under-determined.

**3. The alternative is not rescued either, and that matters more.** `NOLOCALS=1`
sets `__launch_bounds__` 128 → 1, and `dont_use_locals` **deletes the local
dimension** (`gpudims.py:61-62`). So the real variable is **work per thread**: a 128×
increase in per-thread work, of which the time captures **0.148**. Reading B
("graph size causes both") therefore also fails — what the knob actually moved was
neither graph size nor spill.

**What survives is the part §5 got right for the wrong reason.** The launch count
is genuinely not the channel, and the graph operations that `flat_gather` removes
are genuinely what separate the two regimes. **The mechanism connecting them is
unidentified.** The honest statement is now:

> **The launch budget is not the channel, and neither is spill. The variable that
> actually moves the time 19× with a byte-identical spill profile is work per
> thread, and the framework's own work-per-thread choice — not its register
> allocation — is the open question.**

**4. One incidental result worth more than the falsification.** `-maxrregcount` is
**inert on any kernel TinyGrad emits**: `__launch_bounds__(128)` becomes `.maxntid`,
and ptxas then ignores the flag. Nineteen settings produce **one** cubin; delete the
`__launch_bounds__` line and ten settings produce **ten** cubins. So the knob is
reachable only by removing the line the framework emits, which is a small,
concrete gap rather than an inference.

**Honest limits on all of the above, from `notes/38`:** the ideal experiment — an
identical instruction stream with the frame moved by an order of magnitude — **did
not run.** Nine `-maxrregcount` cells all recorded `ptxas_calls = 0`, so the
compiler hook never fired and all nine timed the stock binary; that is an
**instrument failure, not a null result, and nothing was imputed from it.** And the
two sources are **not** byte-identical, because removing the local dimension changes
the index arithmetic throughout — equal operation count is not an identical program,
and the register reading rests on a single reader (ptxas), `cuFuncGetAttribute`
having returned `None` in both cells.

**And the correlation was re-derived here rather than quoted, including a check
`notes/36` does not report. (Read this paragraph as evidence that the *correlation*
is real; §5c below shows it is not causal, and that the quantity is mislabelled.)**
`notes/36` does not report.** The five frames are 20 932, 13 216, 3 068, 2 984,
3 572 against wall clocks 8.880, 6.034, 3.020, 2.741, 2.950. Recomputed Pearson on
the raw values is **+0.9986** — reproduced exactly. A pairwise sign test over every
pair whose difference is resolvable finds **7 of 7 agreeing**: a larger frame is
always slower, with no counterexample.

**But the check also bounds what the correlation can mean.** Restricting to the
three clustered frames near 3 KB gives **r = +0.40 on n = 3** — nearly nothing — and
the two large frames give r = +1.0 trivially, because two points always do. So
**the correlation is carried by the two points where graph size also differs most**,
which is exactly the configuration in which "the frame is the channel" and "the
frame is a symptom of graph size" are indistinguishable. **The one pair that
disagrees in sign — `flat_gather` at 3 068 B / 3.020 ms against `lean` at 3 572 B /
2.950 ms — is below the resolution of both measures**, 588 B on a ~3 KB frame and
0.070 ms on a ~3 ms time, so it is a noise floor rather than a counterexample.
**The causal ambiguity is untouched by the correlation; it needs a same-graph-size
case, which is what `notes/38` is for.**

**And it is the same law §6 found on the other side of the abstraction.** There,
crossing the boundary was free until it spilled and cost 2.4× once it did. Here, the
best port's residual *is* a spill. **One mechanism covers both the faithful port's
penalty and the clean port's remaining gap: local memory.**

Per-rung allocator peaks for the middle three rungs are **not** measured: TinyGrad's
CUDA allocator caches device buffers, so only the first graph built at a shape
reaches `cudaMalloc`, and getting the rest would have cost a second T4 session.
That is a measurement-order artefact, not a rung allocating nothing. The blanks are
blanks.

**Three things this rules out, each with its evidence:**

1. **Not a tuning failure.** Every codegen knob the framework exposes was swept
   one at a time. **Six of seven emit byte-identical CUDA**; the seventh
   (`TRANSCENDENTAL`) makes the hot kernel 3.8% larger and changes nothing
   measurable; the only one that reached the scheduler (`NOLOCALS`) emitted a
   one-thread block and cost nothing in wall clock. And `OPT_LEVEL` does not
   exist. So the gap **cannot be closed through the exposed knobs** (`notes/27`).
2. **Not the memory path.** On an independent seven-rung ladder measured on a T4
   in one run (`notes/33` §7), TinyGrad is **0.99×** the hand-written kernel on
   streaming bandwidth and **0.96×** on a random-access gather. It is *not slower*
   on either, on a card, with both arms in one process and one `cuEvent` path. So
   a gap this size cannot be the memory system. *(These were the two figures
   §7 previously had to flag as unconfirmed; the T4 run resolved them and they
   held. The same ladder's two GEMM ratios did **not** hold and are reported as
   weakened upper bounds — see §7.)*
3. **Not arithmetic.** Per pixel per DDA step the optimised port issues **1.41×**
   the reference's instructions. It runs **9.0×** slower. **Arithmetic accounts
   for 1.41 points of a 9.0-point gap.**

**What is left is class (ii): expressible, and findable by a cost model that does
not exist.** With the port already correct, the residue is **795.8 SASS per step
against the reference's 48.5 — 16.4×** (`notes/27`).

---

## 6. The one result that generalises past this workload

A third arm ports the hand-written kernel's techniques **as faithfully as the
framework permits**, crossing into the framework's own knobs where the kernel's
technique has no expression otherwise. It is *slower* than the clean port, and the
reason is the same in both subjects studied:

| batch | reference µs | clean ms | faithful ms | faithful/clean |
|---|---|---|---|---|
| 512 | 146.3 | 16.115 | 17.200 | **1.067×** |
| 1024 | 288.8 | 3.012 | 10.442 | **3.466×** |
| 2048 | 561.9 | 5.642 | 20.715 | **3.672×** |

**In the unfused regime the crossings cost 1.067× against a static instruction
ratio of 1.044 — within 2.3%. In the fused regime the clock sees 2.4× more than
the instruction stream does, and the difference is entirely register spill
(9.47×), caused by one crossing.**

> **⚠ ALSO FALSIFIED, by the same case as §5c.** The cost of the abstraction
> boundary is not the boundary, and it is not the register pressure the crossing
> induces. `notes/38` moved a live-range knob (`NOLOCALS=1`) and got **19.0×** on
> the clock with a **byte-identical** frame, spill, register, SASS and LDL/STL
> profile. So the variable this law names did not move while the effect it predicts
> did. **The one surviving part is the negative half: crossings really are free when
> nothing changes — and "nothing changes" is now the thing that is hard to
> arrange, not a property of crossings.**

The KDA arm carried the same law independently — a faithful port at 2.7× the clean
port's instructions with 52,332 bytes of spill against 0 for clean — and **that
corroboration is now also suspect, for the same reason:** the lane showed on the
DDA subject that a byte-identical spill profile can accompany a 19× time change, so
co-varying spill and time is not evidence that spill causes time. **The KDA numbers
stand; the inference drawn from them does not.**

---

## 7. What is not measured, and what it would take

Stated plainly, because a report that hides its holes cannot be continued from.

- **The five-rung wall clock is measured** (§3) and **§5's mechanism is falsified**
  (§5c). `NOLOCALS=1` moves the clock **19.0×** with a byte-identical frame, spill,
  register and SASS profile, and the variable it actually moves is work per thread,
  not graph size either. **What remains unidentified is the mechanism**, and the
  ideal experiment — an identical instruction stream with the spill moved by an
  order of magnitude — **did not run**, because the compiler hook never fired in
  nine attempts. That is an instrument failure and nothing was imputed from it
  (`notes/38`).
- **The two GEMM rungs and the reduction rung are upper bounds**, not findings:
  their hand-written references sit near 20% of their own instruction-class peak.
  The DDA ladder, the scan restatement (§8b) and the reference-quality argument
  (§39) have no such caveat.
- **The submitted optimised rung is not the fastest one.** `incremental` beats
  `lean` by 7.6% consistently, so the headline gap is stated against the better of
  the two and `lean` is reported anyway.
- **The ladder is not monotone.** Below the crossover the two smallest graphs are
  16–20% *slower* than the naive port, and nothing measured here predicts it. Any
  summary of this study that presents the ladder as a climb is wrong.
- **`C-25` is resolved: the reference's mean is 16.54 DDA steps per pixel**, not
  26.416. Two independent measurements now agree on 16.5384 with byte-identical
  gating, against one earlier figure of 26.416 that no re-run has reproduced. The
  early-exit term is 32/16.54 = **1.94×** and the 1.21× figure is withdrawn.
- **The control rungs are now T4-measured and they held** (§5 item 2). The two
  GEMM ratios from the earlier, unconfirmed set did **not** hold: 4.2× and 1.3×
  became 2.69× and 1.84×, so the "fp32 versus fp16 differ by 3.2×" reading of the
  dtype contrast is **withdrawn**, and both GEMM ratios are **upper bounds** because
  their hand-written references sit near 20% of their own instruction-class peak.
  They are reported as weakened, not as findings (ledger C-26).
- **The sequential-depth half of the ladder was `INVALID` and carried no ratio; it
  now measures** — 2.97× for a solve, and a bare chain that **rests from 25.96× to
  34×** against the reference once its own reference hole below N=513 was found and
  the cost decomposed into two nested windows rather than one (§8b). Tuning the one
  literal recovers 4.5× and 7.5× survives it, so this rung is the study's clearest
  result: a self-admitted defect, priced rather than inferred.
- **The cliff's reference is missing where it matters.** The hand-written scan has
  no correct kernel for 257 ≤ N ≤ 1023, which is exactly the band where the
  decomposition switches — so a rung whose reference is absent in the mechanism's
  transition band cannot price the mechanism (ledger C-34).
- **Whether the subject's own reference can be replaced is settled, and the answer
  is no** (`notes/34`). A sibling operator runs unnarrowed on sm_75, but it has no
  delta rule — no β, no WY representation, no `C×C` triangular inverse — so
  substituting it would delete the hardest thing the compiler is being asked to do.
  The subject's reference runs, degraded to 4 of 12 autotune configurations, and
  that is the comparison to keep. **One further finding is free: the reference
  should be run in fp16, not bf16** — 7.148e-04 against 5.714e-03 relative error on
  the same card, which is a dtype choice and not a change of operator.
- **There are no hardware counters anywhere.** Kaggle exposes neither `ncu` nor
  `nsys` and a container cannot grant the privilege. Every conclusion here rests
  on wall-clock ratios from one clock, plus a memory floor measured rather than
  quoted.
- **Two claims are in direct conflict and are unresolved:** the reference's mean
  DDA steps per pixel is 26.416 in one record and 16.54 in another (ledger C-25).
  The early-exit term depends on it and is currently 1.93× or 1.21× depending on
  which is right.
- **The 9.0× is not fully decomposed.** 1.41× (instructions) × 1.93× (work) ≈
  2.7× of it. The remaining ~6.3× is measured and unattributed. It was left that
  way rather than filled with an estimate.

## 8. What this study says about the framework, in one paragraph

Across four subjects and four subsystems, the finding is the same shape: **the
information needed to make the right decision is computed and then not used.**
TinyGrad fills in a memory-cost estimate that is accurate to 1.20× and that no
optimisation reads; its fusion decision is a function of graph size with a
threshold the port crosses for better reasons than arithmetic; its per-launch cost
model gets the *sign* wrong on a real card by 676 µs, so it would prefer the worse
of two correct implementations; and its cumulative sum lowers a windowed
re-summation through a hardcoded `SPLIT = 256` beside the comment *"TODO: someday
the optimizer will find this on its own"* (`tinygrad/mixin/op.py:764-766`).
**An earlier version of this sentence said that literal was "the entire mechanism
behind the one cliff". That was wrong, and `notes/37` refuted it — see §8b. The
literal is causal and correctly located, but it accounts for about 4.5× of a
larger cost, and the two-stage form is a *mitigation* rather than the injury.
The four-way classification this project set out to produce is therefore
**dominated by class (ii) — expressible, and findable by a cost model that is not
there** — with a smaller and precisely located set of
genuine expressiveness failures: per-operation precision inside one kernel,
token-parallel versus contracted forms being indistinguishable, and a blocking
strategy that is a tensor-core blocking which costs 4.1× when copied into a
framework with no tensor core on that path.

**The shape of the envelope, from the seven-rung ladder on one T4** (`notes/33`):

| | ratio | |
|---|---|---|
| streaming bandwidth | **0.99×** | at parity |
| random-access gather | **0.96×** | at parity |
| reduction | 1.92× | |
| contraction, fp16 / fp32 | 1.84× / 2.69× | upper bounds, weakened |
| sequential solve + contraction | 2.97× | |
| **bare sequential chain** | **25.96×** as first measured — **restated to 34× against the reference, or 50× the copy floor, once the reference's own hole below N=513 was found and the cost was decomposed** (§8b) | **the cliff, and the only rung with a self-documented TODO next to it** |

## 8b. The cliff was measured wrong, and the error was mine

**`notes/37` tested the `SPLIT = 256` claim that §8 used to make, and the claim
failed — in three separate ways, one of them a sign error.** The record is here
because the corrected version is more useful than the one it replaced, and because
a report that hid this would be worse than one that had made the mistake.

**1. The two-stage form is a mitigation, not a cost.** At a fixed shape on the same
card, the framework's **one-stage** path runs at **329.17 ms** and its **two-stage**
path at **56.52 ms**. The one-stage form is **5.8× slower**. I had described the
decomposition as something the code pays *for*; it is something the code pays *to
avoid*. Cross-check: at `SPLIT=4096` the switch never fires, so `cumsum` *is*
`_cumalu`, and the ladder measures 329.59 ms against the explicit call's 329.17 ms
— agreement to 0.04%, so the two arms really are the same program.

**2. The literal is causal and correctly located.** Predicting the switch point
structurally as `2·SPLIT+1` gives 129 / 257 / 513 / 1025 / 2049 for `SPLIT` of
64 / 128 / 256 / 512 / 1024, and the measured values are **exactly those five**,
read two independent ways. So the constant governs where the switch happens. That
part of the original claim was right and it is now established rather than inferred.

**3. The cost is two nested windows, and I counted one.** `cumsum` re-sums in
chunks of `SPLIT` *and then* re-sums the chunk totals in chunks of `N/SPLIT`, so
the per-row cost is `N·SPLIT + (N/SPLIT)²`, minimised at `SPLIT³ = 2N` — about 20
at N=4096 — and **diverging at `SPLIT=1`**, which is the worst point on the ladder
because at a window of one the two-stage form degenerates into the one-stage form.

| | device ms | × copy floor | × hand-written reference |
|---|---|---|---|
| installed `SPLIT = 256` | 28.08 | 50× | **34×** |
| best on the ladder (`SPLIT = 64`) | 6.24 | 11× | **7.5×** |

> **Tuning the one literal recovers 4.5×. 7.5× survives it, and 4.5× is itself a
> lower bound** — the ladder has 8 and 64 with nothing between, while the work
> model puts the optimum near 20. The ladder spans 52.8× and the worst within-arm
> spread is 0.11%.

**So the cliff stands, restated and larger: 34× against the reference rather than
25.96×.** What was wrong was my attribution — the literal is a real, causal,
self-admitted defect worth about 4.5×, and it is not the whole cliff.

**Class (ii) survives this and is now *priced* rather than inferred**, which is the
study's actual thesis: the gap is expressible, and a constant plus a two-term work
model finds 4.5× of it. What the test also exposed is a gap in the *reference*:
**the hand-written scan has no correct kernel for 257 ≤ N ≤ 1023, and that hole is
precisely why `notes/33`'s grid could not see its own falsifier.** A rung whose
reference is missing in the band where the mechanism switches cannot price the
mechanism.

A methodological note from that lane, carried because it generalises: **five times
a first measurement of a shape was wrong** — twice because TinyGrad's program
cache returned the *previous* configuration's program, twice from probe deaths on
missing keys, and once from `realize()` timing a warm-up. The program cache
returning a previous program is a **new instance of the stale-result family** and
the most dangerous of the set, because nothing about the returned object looks
wrong.

**And the correlate that ladder was built to test is refuted by its own data.** The
rung with *more* sequential depth and *more* contraction came back **8.7× closer**
than the bare chain. So the gap is not a function of sequential depth, and the
honest description of the envelope is: **at parity where the framework only moves
data, about 2× where it contracts, and one cliff at a literal with a TODO next to
it.**

**The limit of that description, which is a claim about seven rungs and not about
both subjects** (`notes/39`). Placing the two subjects on the envelope:

- **The DDA subject is inside it** — 11.4× against a worst rung of 25.96×, 0.44×
  the cliff.
- **The KDA subject is outside every shape measured, by 1.08× to 3.92×** — and it
  *closes* on the envelope as the shape grows, the wrong direction for a fixed
  implementation property.
- **But that placement is not valid, and the reason is the reference, not either
  implementation.** `fla`'s T4 configuration spills **12 806 / 14 588 bytes** in
  19 328 SASS at 4 of 12 autotune configurations, while the axis's fp16 GEMM
  reference is 776 instructions, 80 HMMA and **zero spills**. A ratio against a
  spilling reference and a ratio against a non-spilling one are not the same
  measurement. **So the KDA subject's deficit is unquantified against a competent
  reference on this card, not 28–102×.**

**This is what makes a non-T4 card load-bearing rather than optional.** On `cc=89`
the same Triton source emits 24 B of spill instead of 12 806, so a 4090 does not
merely make class (iv) observable — it is the first card on which the KDA subject
has a valid measurement at all. Free T4 time cannot substitute for it: the blocker
is 65 536 B of shared memory against 101 376 B required, measured, in
`fla/utils/_device.py:181-185`.

**Update, and it closes the free route — with a sharper reason than first stated.**
This section recommended buying an RTX 4090, then revised to say a Kaggle L4 might
make it free. **The L4 route was tested and does not work on this account.** The
accelerator is *specified*, not drawn — `kaggle kernels push --accelerator` exists
and Kaggle CLI's docs give `NvidiaL4` — so restarting notebooks was never going to
help.

Seven sessions, one kernel slug, one probe each, so the shape string is the only
variable (`notes/40`):

| requested | GPU attached | cc | devices |
|---|---|---|---|
| `NvidiaL4` (v1, v3) | **none** | — | 0 |
| `NvidiaTeslaT4` (v2, v4, v5) | Tesla T4 ×2 | 7.5 | 2 |
| `NvidiaL4` **+ CUDA image pinned** (v6) | **none** | — | 0 |
| `NvidiaRtxPro6000` (v7) | **none** | — | 0 |

Pinning the CUDA image on v6 tested the obvious mechanism — that an L4 request gets
a non-CUDA image — and the pin worked (torch became `+cu128`, `nvcc 12.8` appeared)
with `device_count` still 0. **So the blocker is the accelerator attach, not the
container.**

**This is not a capacity stockout, and that distinction is the useful part.** A
read-only behavioural census found a public kernel reporting `NvidiaL4` as run
**2026-09-29 08:06 UTC**, roughly 19 hours before the census; `NvidiaRtxPro6000`
likewise. **Other accounts are being granted these cards and this one is not.** So
the free route is closed by an account-level grant, not by the platform — which
also means **whether payment is that grant is not established**, and a shape
appearing in a metadata field is not a policy. The SDK's own docstring lists only
three shapes and none of them is an L4, so trusting the docstring would have ended
this before it started and wrongly.

> **This is the project's recurring failure in its worst form yet: a flag that reads
> as though it worked.** A wrong `machine_shape` does not fail — it produces a
> completed run, an exit code of 0, a full log, and a CPU. Any timing taken under
> that configuration would be a number with no device behind it, and nothing in the
> harness would object. It was caught only because the probe reads the card and
> compares it against what was *requested* — a discipline every other lane here had
> to learn the hard way.

One cost worth recording: **GPU quota moved 17.02 h → 17.20 h across the seven
sessions, including the four that ran on a CPU box.** That is a coarse inference
and is labelled as one, but if it holds, **a downgraded request spends quota and
returns nothing.**

**So the KDA subject's status is unchanged: its deficit is unquantified against a
competent reference on a T4, and the scope label rides on every KDA number.** What
would open it remains a card this account is not granted, or one from outside
Kaggle's free tier.

A second, cheaper finding from the same runs: `torch.cuda.get_arch_list()` on the
T4 lists **`sm_120`**, so the installed torch is built for Blackwell and can emit
for architectures this account cannot run. Anything that infers the device from
torch's build rather than from the device is wrong here.

**And the ladder is missing a dimension that is a count rather than a shape.** Its
own `kernels` column refutes launch count as a predictor — the 585-kernel rung
comes back **8.7× better** than the 3-kernel one. What it does price is *one*
instance of the rank-1 habit (`red`, 2.26×); the KDA port contains 4 032 of them.
A rung of one operator repeated past the framework's program-size budget is the
cheapest available extension, needs only a T4, and would say whether the silence
about program size is a limit of the rungs chosen or a property of the framework.

**Next move, in order: (1) the five-rung timings, (2) the T4 control rungs, (3) the
sequential-depth references, (4) the 26.4-versus-16.54 conflict.** Items 1 and 2
are blocked only on Kaggle's two-concurrent-session cap, not on anything missing.

## Sources

| claim | source |
|---|---|
| reference timings, whole-step figure, crossover | `notes/23` §3–§4, `out/minecu-t4/` |
| ladder definitions and per-rung ops/instructions | `notes/23` §1–§2, `minecu_tg_opt/rungs.py` |
| knob sweep, residue, four-way classification | `notes/27`, `minecu_tg_knobs/` |
| faithful port and the spill law | `notes/28`, `minecu_tg_faithful/` |
| control rungs and the general ladder | `notes/33`, `axis/` |
| corrections, including the ones that voided earlier numbers | `notes/06-corrections-ledger.md` C-01…C-27 |
| the four-way classification and its falsifiers | `notes/04-where-compilers-stop.md`, `research/TAXONOMY.md` |
